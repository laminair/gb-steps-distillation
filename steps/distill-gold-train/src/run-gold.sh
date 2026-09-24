#!/bin/bash
# distill-gold-train step entrypoint, INSIDE the container.
#
# Deliberately thin: every decision that can be wrong in a way that costs a multi-node
# training run lives in render_gold_config.py, which is unit-tested. This script renders
# the config, works out which topology the config implies, and either launches it or
# refuses with a reason.
#
# WHAT THIS IS NOT. It is not a port of the reference on-policy launcher script it was
# modeled after. That script runs inside an LSF allocation and uses LSB_HOSTS + blaunch to
# split one allocation into a vLLM server host and trainer hosts. A granite.build step does
# not get an LSF allocation; it gets a container (Docker) or a Skypilot task. The split-role
# multi-node shape therefore does not translate, and that is tracked as D3 rather than
# papered over -- see the SPLIT-ROLE section below.
set -uo pipefail

# ---------------------------------------------------------------- locating our own parts
# STEP_HOME is derived from where THIS FILE lives, not hard-coded to /opt/<step-name>.
# In the image the two are identical -- the Dockerfile COPYs steps/distill-gold-train/src/
# into /opt/distill-gold-train/ -- so this changes nothing about a container run. What it
# buys is that the RENDERED step command is runnable against the checkout, which is the
# only way this step gets tested before there is an image (containerization is a separate,
# later piece of work). Confirmed directly why: render-step-command.py rewrites the one
# `/opt/<step>/` prefix it can see in the command text, so `bash .../src/run-gold.sh`
# resolved fine and then the script looked for /opt/distill-gold-train/render_gold_config.py
# and died with rc=2 before reaching anything under test.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STEP_HOME="${STEP_HOME:-$HERE}"
# Repo root when running from the checkout (steps/<name>/src -> ../../..); meaningless in
# the image, where every candidate built from it simply does not exist and is skipped.
CHECKOUT_ROOT="$(cd -- "$STEP_HOME/../../.." 2>/dev/null && pwd || echo "")"

# first_dir <candidate>... -- echoes the first candidate that is a directory. Both
# consumers below fall back to the IMAGE path when nothing matches, so a genuinely broken
# image still fails naming the path it expected rather than an empty one.
first_dir() { local c; for c in "$@"; do [[ -d "$c" ]] && { (cd "$c" && pwd); return 0; }; done; return 1; }

# The distillation package: /opt/<step>/vendor/... in the image (and on PYTHONPATH there),
# src/gb_steps_post_training/... in the checkout.
if [[ -z "${GOLD_SRC:-}" ]]; then
  GOLD_SRC="$(first_dir \
    "$STEP_HOME/vendor/gb_steps_post_training/distillation" \
    "${CHECKOUT_ROOT:-/nonexistent}/src/gb_steps_post_training/distillation" \
  )" || GOLD_SRC="$STEP_HOME/vendor/gb_steps_post_training/distillation"
fi
# The shared launcher library: /opt/<step>/lib/ in the image, a companion lib/ tree
# alongside this repo's own tooling when run from a checkout. Resolving it in both places
# matters -- it holds the hub-kernel preflight, and a checkout run that silently skipped
# that preflight would be testing a different script than the image runs.
if [[ -z "${LIB_DIR:-}" ]]; then
  LIB_DIR="$(first_dir \
    "$STEP_HOME/lib" \
    "${CHECKOUT_ROOT:-/nonexistent}/lib" \
  )" || LIB_DIR="$STEP_HOME/lib"
fi
# render_gold_config.py imports the package (gb_steps_post_training.distillation.
# tokenizer_identity -- deliberately shared with distill-tokenizer-align so the side that
# WRITES a tokenizer identity and the side that READS it cannot drift). The Dockerfile puts
# the package root on PYTHONPATH with an ENV line, which a checkout run does not get, so
# the import is resolved here from GOLD_SRC instead: the package root is two levels above
# the distillation directory (/opt/<step>/vendor in the image, <repo>/src in a checkout).
# In the image this prepends a path that is already on PYTHONPATH, which costs nothing.
GOLD_PKG_ROOT="$(cd -- "$GOLD_SRC/../.." 2>/dev/null && pwd || echo "")"
[[ -n "$GOLD_PKG_ROOT" ]] && export PYTHONPATH="$GOLD_PKG_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYBIN="${PYBIN:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"

# Defaults mirror the step template; every one is overridable by a flag.
STUDENT="" ; TEACHER="" ; TEACHER_TOK="" ; CORPUS="" ; OUTPUT_DIR="output"
LMBDA="0.3" ; BETA="0.5" ; TEMPERATURE="1.0" ; VLLM_TEMPERATURE="1.0"
LOSS_ARM="jsd" ; MAX_LENGTH="4096" ; VLLM_MODE="server" ; VLLM_NUM_SERVERS="1"
# Both default to render_gold_config.py's own argparse defaults (0.0 / ""), so a template
# that omits them renders exactly what it renders today: gamma 0.0 disables the auxiliary
# hidden-state loss, and an empty layer list means last-layer-only when it is enabled.
HIDDEN_LOSS_GAMMA="0.0" ; HIDDEN_LOSS_LAYERS=""
PDTBS="1" ; GAS="8" ; LR="1e-6" ; EPOCHS="1" ; SAVE_STEPS="100" ; SEED="42"
# auto = gold.py's own presence-based resume (gold.py:507-510), unchanged. See
# render_gold_config.py's RESUME_MODES for why never/require exist.
RESUME="auto"
LAST_MESSAGE_ONLY="0" ; DS_CONFIG="" ; EXTRA_YAML="" ; GPN="8" ; NODES="2" ; HF_HOME_ARG=""
# Tracking. Empty is the OFF state for all five, and empty is what the step template passes when
# its `tracking:` block is left at its defaults -- so the untracked run stays the zero-config one.
# These are forwarded verbatim to render_gold_config.py, which emits only the non-empty ones into
# the rendered config; resolution (which backend is on, whether a key exists, what the run is
# named) happens once, in tracking.py, and not here. See step-template.yaml's `tracking:` comment.
# TRK_ prefix, deliberately: unprefixed, these hold the flag values under the EXACT names the
# two libraries read from the environment. They are shell locals and never exported, so nothing
# reads them today -- but this script exports HF_HOME and HF_HUB_OFFLINE fifty lines down, and one
# stray `export` or `set -a` would hand ClearML a raw, unvalidated project name, bypassing the
# partial-config abort in tracking.py that is the whole point of routing through it. The prefix
# makes that impossible rather than unlikely. tracking-config.sh asserts the env vars are set in
# tracking.py alone, which is the same rule stated from the other side.
TRK_CLEARML_PROJECT="" ; TRK_CLEARML_RUN_NAME=""
TRK_WANDB_ENTITY="" ; TRK_WANDB_PROJECT="" ; TRK_WANDB_RUN_NAME=""

while (( $# )); do
  case "$1" in
    --student-model-path)   STUDENT="$2"; shift 2 ;;
    --teacher-model-path)   TEACHER="$2"; shift 2 ;;
    --teacher-tokenizer-path) TEACHER_TOK="$2"; shift 2 ;;
    --corpus-path)          CORPUS="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
    --lmbda)                LMBDA="$2"; shift 2 ;;
    --beta)                 BETA="$2"; shift 2 ;;
    --temperature)          TEMPERATURE="$2"; shift 2 ;;
    --vllm-temperature)     VLLM_TEMPERATURE="$2"; shift 2 ;;
    --loss-arm)             LOSS_ARM="$2"; shift 2 ;;
    --hidden-loss-gamma)    HIDDEN_LOSS_GAMMA="$2"; shift 2 ;;
    --hidden-loss-layers)   HIDDEN_LOSS_LAYERS="$2"; shift 2 ;;
    --max-length)           MAX_LENGTH="$2"; shift 2 ;;
    --vllm-mode)            VLLM_MODE="$2"; shift 2 ;;
    --vllm-num-servers)     VLLM_NUM_SERVERS="$2"; shift 2 ;;
    --per-device-train-batch-size) PDTBS="$2"; shift 2 ;;
    --gradient-accumulation-steps) GAS="$2"; shift 2 ;;
    --learning-rate)        LR="$2"; shift 2 ;;
    --num-train-epochs)     EPOCHS="$2"; shift 2 ;;
    --save-steps)           SAVE_STEPS="$2"; shift 2 ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --resume)               RESUME="$2"; shift 2 ;;
    --last-message-only)    LAST_MESSAGE_ONLY="1"; shift ;;
    --deepspeed-config)     DS_CONFIG="$2"; shift 2 ;;
    --extra-config-yaml)    EXTRA_YAML="$2"; shift 2 ;;
    --gpus-per-node)        GPN="$2"; shift 2 ;;
    --nodes)                NODES="$2"; shift 2 ;;
    --hf-home)              HF_HOME_ARG="$2"; shift 2 ;;
    --clearml-project)      TRK_CLEARML_PROJECT="$2"; shift 2 ;;
    --clearml-run-name)     TRK_CLEARML_RUN_NAME="$2"; shift 2 ;;
    --wandb-entity)         TRK_WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-project)        TRK_WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-run-name)       TRK_WANDB_RUN_NAME="$2"; shift 2 ;;
    *) echo "FATAL: unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "=== distill-gold-train ==="
date; hostname
echo "step home : $STEP_HOME"
echo "gold src  : $GOLD_SRC"
echo "lib dir   : $LIB_DIR"
echo "pkg root  : ${GOLD_PKG_ROOT:-<unresolved>}"

# An empty hf_home means "leave the image's own default alone", which is what lets a
# profile omit it. Setting HF_HOME="" would instead point the cache at the process CWD.
if [[ -n "$HF_HOME_ARG" ]]; then
  export HF_HOME="$HF_HOME_ARG"
  echo "HF_HOME   : $HF_HOME"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Same precondition as the reference launchers this was modeled after, same reason: gold.py
# resolves the Mamba2 hub kernels at import time, and with HF_HUB_OFFLINE=1 a cold cache is
# fatal AFTER the GPUs are allocated. In a step the fix is baked into the image, so a miss
# here means the image was built wrong -- still worth catching before the launch.
if [[ -f "$LIB_DIR/gold-kernels.sh" ]]; then
  source "$LIB_DIR/gold-kernels.sh"
  require_hub_kernels "$PYBIN" || exit 1
fi

if [[ ! -f "$STEP_HOME/render_gold_config.py" ]]; then
  echo "FATAL: render_gold_config.py not found under STEP_HOME=$STEP_HOME." >&2
  echo "       In the image it is COPYd there from steps/distill-gold-train/src/; in a" >&2
  echo "       checkout STEP_HOME is that directory. Set STEP_HOME explicitly if this" >&2
  echo "       script was moved away from it." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" || { echo "FATAL: cannot create output_dir: $OUTPUT_DIR" >&2; exit 1; }
RENDERED="$OUTPUT_DIR/gold-config.rendered.yaml"

LMO_FLAG=()
[[ "$LAST_MESSAGE_ONLY" == "1" ]] && LMO_FLAG=(--last-message-only)

"$PYBIN" "$STEP_HOME/render_gold_config.py" \
  --student-model-path "$STUDENT" \
  --teacher-model-path "$TEACHER" \
  --teacher-tokenizer-path "$TEACHER_TOK" \
  --corpus-path "$CORPUS" \
  --output-dir "$OUTPUT_DIR" \
  --lmbda "$LMBDA" --beta "$BETA" \
  --temperature "$TEMPERATURE" --vllm-temperature "$VLLM_TEMPERATURE" \
  --loss-arm "$LOSS_ARM" --max-length "$MAX_LENGTH" \
  --hidden-loss-gamma "$HIDDEN_LOSS_GAMMA" \
  --hidden-loss-layers "$HIDDEN_LOSS_LAYERS" \
  --vllm-mode "$VLLM_MODE" --vllm-num-servers "$VLLM_NUM_SERVERS" \
  --per-device-train-batch-size "$PDTBS" \
  --gradient-accumulation-steps "$GAS" \
  --learning-rate "$LR" --num-train-epochs "$EPOCHS" \
  --save-steps "$SAVE_STEPS" --seed "$SEED" \
  --resume "$RESUME" \
  "${LMO_FLAG[@]}" \
  --deepspeed-config "$DS_CONFIG" \
  --extra-config-yaml "$EXTRA_YAML" \
  --gpus-per-node "$GPN" --nodes "$NODES" \
  --clearml-project "$TRK_CLEARML_PROJECT" \
  --clearml-run-name "$TRK_CLEARML_RUN_NAME" \
  --wandb-entity "$TRK_WANDB_ENTITY" \
  --wandb-project "$TRK_WANDB_PROJECT" \
  --wandb-run-name "$TRK_WANDB_RUN_NAME" \
  --out "$RENDERED" || exit $?

# ------------------------------------------------------------- allocation assertion
# step-template.yaml claims gpus_per_node is "asserted against the actual allocation
# rather than trusted". It was not: render_gold_config.py only range-checks it (1..8).
# The claim is now true, because the untrue version was worse than no claim -- a reader
# who believes the check exists stops looking for the failure it was supposed to catch.
#
# What this catches, concretely: a site profile that grants ACCELERATORS "H100:2" while
# every distillation recipe carries GPUS_PER_NODE: 8. Without this check
# accelerate launches 8 processes onto 2 devices and dies deep inside DeepSpeed's
# initialisation, after the allocation is already held.
#
# CUDA_VISIBLE_DEVICES wins over nvidia-smi when set, because a scheduler that gives a
# job 2 of a host's 8 GPUs restricts it that way -- nvidia-smi -L would report all eight
# and the assertion would pass on an allocation that cannot run.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE=$(awk -F, '{n=0; for(i=1;i<=NF;i++) if(length($i)) n++; print n}' <<<"$CUDA_VISIBLE_DEVICES")
  VISIBLE_SRC="CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  VISIBLE_SRC="nvidia-smi -L"
else
  VISIBLE=""
  VISIBLE_SRC=""
fi

if [[ -n "$VISIBLE" ]]; then
  echo "allocation: $VISIBLE GPU(s) visible (via $VISIBLE_SRC), gpus_per_node=$GPN"
  if (( VISIBLE != GPN )); then
    echo "FATAL: gpus_per_node=$GPN but $VISIBLE GPU(s) are visible (via $VISIBLE_SRC)." >&2
    echo "       Training at a different width than the config claims is silent: the loss" >&2
    echo "       curve looks plausible and the effective batch size is wrong by a factor" >&2
    echo "       of $GPN/$VISIBLE. Fix the profile's ACCELERATORS or the recipe's" >&2
    echo "       GPUS_PER_NODE so the two agree -- do not just lower one to make this pass" >&2
    echo "       unless that is the width you meant to train at." >&2
    exit 1
  fi
else
  # Not fatal: a CPU-only render or a dry inspection has no GPUs and no business failing
  # here. Loud, though -- an unasserted allocation is exactly what this block exists to
  # stop being invisible.
  echo "WARNING: cannot count GPUs (no CUDA_VISIBLE_DEVICES, no nvidia-smi)." >&2
  echo "         gpus_per_node=$GPN is UNVERIFIED against the real allocation." >&2
fi

# ------------------------------------------------------------------ topology dispatch
# lmbda decides the SHAPE, not just the loss. Three cases, and only two of them are
# expressible as a single container today.
ON_POLICY=$("$PYBIN" -c "import sys; print(1 if float(sys.argv[1]) > 0.0 else 0)" "$LMBDA")

if (( ON_POLICY == 0 )) || [[ "$VLLM_MODE" == "colocate" ]]; then
  # All-trainer: off-policy (the student never generates, so there is no server) or
  # colocate (vLLM shares the trainer's GPUs). Both fit one container.
  NUM_PROCESSES=$(( NODES * GPN ))
  if (( NODES > 1 )); then
    # Multi-node all-trainer needs rendezvous values this step is not given: a step gets
    # no host list. Refuse rather than launch a single node while reporting NODES.
    echo "FATAL: nodes=$NODES with an all-trainer topology needs a multi-node rendezvous" >&2
    echo "       (main_process_ip/port and a machine_rank per host) that this step is not" >&2
    echo "       handed. Use nodes=1 here, or run it through a launcher that supplies" >&2
    echo "       the rendezvous values inside a real multi-node allocation. Tracked as D3." >&2
    exit 1
  fi
  echo "topology  : all-trainer, 1 node x $GPN GPU(s)  (lmbda=$LMBDA vllm_mode=$VLLM_MODE)"
  cd "$GOLD_SRC" || exit 1
  "$ACCELERATE" launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --config_file "$DS_CONFIG" \
    gold.py --config "$RENDERED" --output_dir "$OUTPUT_DIR"
  RC=$?
else
  # ---------------------------------------------------------------- SPLIT-ROLE (D3)
  cat >&2 <<'D3EOF'
FATAL: this configuration needs the split-role multi-node topology, which is not yet
       expressible as a single granite.build step. This is decision D3, open on purpose.

  You asked for lmbda > 0.0 with vllm_mode='server'. That shape is: N nodes in ONE
  allocation, vllm_num_servers of them running vLLM serving the STUDENT, the rest running
  the trainer, with the trainer learning each server's address at run time.

  Why it does not reduce to "start two containers":
    - Both roles must share a single allocation, because the working implementation uses
      LSF blaunch, and blaunch can only reach hosts inside its own allocation.
    - The trainer cannot be configured with the server addresses ahead of time. They are
      whatever hosts the scheduler picked, discovered from LSB_HOSTS after the job starts.
    - The launcher waits on each server's /health before fanning out the trainer, so the
      two roles are ordered, not merely co-scheduled.

  What to do today -- this is a real path, not a placeholder: a reference LSF-allocation
  launcher script (outside this repo) is the tested route for the on-policy arm. This step
  covers the off-policy and colocate arms; the on-policy server arm stays on that reference
  launcher until D3 is resolved with a real multi-node step contract.

  Resolving D3 means answering: which environment_config expresses one allocation with
  two heterogeneous roles, and how does the second role learn the first's address? Nothing
  in recipes/_profiles/ expresses that today, which the plan doc flags as a Phase 3 gap.
D3EOF
  exit 1
fi

echo
echo "trainer rc=$RC  output=$OUTPUT_DIR"
date
echo "=== done (rc=$RC) ==="
exit "$RC"
