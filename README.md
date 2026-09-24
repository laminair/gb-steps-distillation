# gb-steps-distillation

Source for the seven knowledge-distillation steps [granite.build](https://github.com/ibm-granite/granite.build)
(`ibm-granite/granite.build`) uses to distill a smaller Granite student from a larger
Granite teacher: `distill-sft`, `distill-tokenizer-align`, `distill-corpus-prep`,
`distill-logit-precompute`, `distill-gold-train`, `distill-hf-export`, `distill-eval`.

**This repo is source only.** It has no CLI, no orchestrator, and nothing you run
standalone — granite.build clones it automatically and runs its entrypoints as steps.
You never need to clone or touch this repo by hand; see [Using this with
granite.build](#using-this-with-granitebuild) below.

## Quickstart

You run a distillation recipe from **granite.build**, not from here:

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build
# install the gb CLI and configure a SkyPilot/LSF environment — see granite.build's
# own docs/getting-started.md and docs/environments/ for that one-time setup.

gb build start \
  -f recipes/granite4-gold-distillation/lsf/gold-smoke/build.yaml \
  --space <your-space>
```

`parameters.yaml` next to that `build.yaml` supplies the defaults (a small teacher/student
pair, a tiny corpus slice, a handful of training steps), so this runs end-to-end in
minutes rather than hours. Override any value with `--param KEY=VALUE`.

**Done** looks like: `gb build status` reports the target `SUCCESS`, and a `checkpoint`
artifact (type `model`) is registered — the trained student's weights, on whatever
filesystem or object store the environment resolves `env://` / step output URIs against.

## Pipeline structure

The steps chain into one model-agnostic pipeline. Each stage's output is the next
stage's input:

```
distill-sft              (optional but recommended)
       │  chat-template warm-up: teaches a raw base student the teacher's ChatML
       │  turn markers before any distillation happens. Output: a checkpoint.
       ▼
distill-tokenizer-align
       │  retags the student's tokenizer onto the teacher's vocabulary/turn markers,
       │  and builds tokenizer "overlays" (fast-tokenizer-only directories) for both
       │  sides. Output: a retagged student + teacher/student tokenizer overlays.
       ▼
distill-corpus-prep
       │  filters and normalises a raw conversation dataset against the ALIGNED
       │  student's tokenizer: length limits, completion-boundary checks, role
       │  normalisation. Output: a training corpus + a manifest describing it.
       ▼
distill-logit-precompute   (optional — only for the precomputed-logits KD arm)
       │  runs the teacher once over the corpus and stores its top-K logits per
       │  assistant token, so a later training run doesn't need the teacher loaded.
       │  Output: a sharded logit/index fileset.
       ▼
distill-gold-train
       │  the actual GOLD (generalized JSD) knowledge distillation: trains the
       │  student against the teacher, off-policy or on-policy. Output: a checkpoint.
       ▼
distill-hf-export
       │  selects/prunes a training checkpoint into a publishable, HF-native model
       │  directory (no format conversion — DeepSpeed ZeRO-3 already saves HF-native
       │  weights; this step picks the right checkpoint-N and strips training-only
       │  artifacts). Output: a publishable HF model directory.
       ▼
distill-eval
       │  measures how far the student's output distribution has moved toward the
       │  teacher's (JSD, KL, entropy) on a held-out corpus — the "did transfer
       │  actually happen" question capability evals like BFCL can't answer.
       ▼
   (scored student)
```

`distill-sft` and `distill-logit-precompute` are both optional stages a recipe can skip;
the rest form the load-bearing spine. granite.build's own
`recipes/granite4-gold-distillation/lsf/distill-pipeline-smoke/` and
`recipes/granite4-350m/lsf/distill-smoke/` build.yamls wire all of this together
end-to-end, including an `INCLUDE_SFT` switch that routes `distill-sft`'s checkpoint into
`distill-gold-train` as the student instead of skipping straight to alignment.

## Step reference

- **`distill-sft`** — plain SFT on a chat corpus, no teacher. Its primary role is the
  chat-template warm-up above; with `precomputed_logits_dir` set it doubles as a
  forward-KL distillation arm in its own right, or (left empty) as the baseline other
  arms are measured against. Entrypoints: `steps/distill-sft/src/run-sft.sh`,
  `render_sft_config.py`, `check_weight_residency.py`; trainer at
  `src/gb_steps_post_training/distillation/sft.py`. Wired in granite.build as
  `space://steps/distill-sft`.

- **`distill-tokenizer-align`** — retags a student's tokenizer onto a ChatML teacher's
  vocabulary and builds fast-tokenizer-only overlays for both sides, so a class-identity
  bug in `transformers` (`tokenizer_class: "GPT2Tokenizer"` silently overriding the
  trained `pre_tokenizer`) can't reach either model. Entrypoint:
  `steps/distill-tokenizer-align/src/run-align.sh`, backed by
  `src/gb_steps_post_training/distillation/{retag_student,build_overlay,tokenizer_identity,fast_tokenizer}.py`.
  Wired as `space://steps/distill-tokenizer-align`.

- **`distill-corpus-prep`** — filters and normalises a conversation dataset against the
  aligned student's tokenizer, with a verified (not assumed) completion-boundary check.
  Entrypoints: `steps/distill-corpus-prep/src/prep_corpus.py` and `merge_shards.py`
  (shard/merge for parallel prep), backed by
  `src/gb_steps_post_training/distillation/{masking,step_state}.py`. Wired as
  `space://steps/distill-corpus-prep`.

- **`distill-logit-precompute`** — runs the teacher once over the corpus, keeps the
  top-K logits per assistant token, so training reads them off disk instead of holding
  the teacher in memory. Entrypoint: `steps/distill-logit-precompute/src/run-precompute.sh`,
  backed by `src/gb_steps_post_training/distillation/precompute_logits.py`. Wired as
  `space://steps/distill-logit-precompute`.

- **`distill-gold-train`** — renders the GOLD trainer's config and launches it; the
  trainer itself (generalized JSD distillation, off-policy or on-policy via vLLM) is
  `src/gb_steps_post_training/distillation/gold.py` and `custom_gold_trainer.py`.
  Entrypoints: `steps/distill-gold-train/src/render_gold_config.py` and `run-gold.sh`;
  the DeepSpeed ZeRO-3 config it needs lives at
  `steps/distill-gold-train/configs/deepspeed/accelerate_deepspeed_zero3.yaml`, and an
  optional Hub-kernel warm-cache preflight at
  `steps/distill-gold-train/src/lib/gold-kernels.sh`. On-policy runs additionally use
  `run_vllm_serve.py` to serve the student under vLLM. Wired in granite.build as
  `space://steps/distill-gold` (the trainer's `gold-train` name shortens on the
  granite.build side) and, for on-policy serving, `space://steps/vllm-server`.

- **`distill-hf-export`** — selects the right `checkpoint-N` from a training run and
  prunes it into a publishable HF model directory; no weight conversion, since ZeRO-3
  already writes HF-native safetensors. Entrypoint:
  `steps/distill-hf-export/src/export_hf_model.py`. Wired as
  `space://steps/distill-hf-export`.

- **`distill-eval`** — measures JSD/KL/entropy between the student's and teacher's
  output distributions on a held-out corpus. Entrypoint:
  `steps/distill-eval/src/run-eval.sh`, backed by
  `src/gb_steps_post_training/distillation/{divergence,run_divergence}.py`. Wired as
  `space://steps/distill-eval`.

Every entrypoint above imports from the shared package at
`src/gb_steps_post_training/distillation/` — the modules not named individually
(`utils.py`, `render_common.py`, `tracking.py`, `checkpoints.py`, and similar) are
helpers those entrypoints pull in.

## Using this with granite.build

You don't clone this repo yourself. Each of the seven steps' `step-template.yaml` in
granite.build has a `code_config` block that clones this repo automatically, over
unauthenticated HTTPS (it's public, so no credential is needed):

```yaml
code_config:
  code_dir: ""
  repo: "https://github.com/laminair/gb-steps-distillation.git"
  ref: "a5d59bc45524a8d75706e20d44ae1a254f273f23"
  expect_ref: "a5d59bc45524a8d75706e20d44ae1a254f273f23"
```

`repo`/`ref` are what's actually cloned and checked out; `expect_ref` is a belt-and-braces
check the step re-verifies before running, so a moved or wrong checkout fails loudly
instead of silently training against the wrong code. Bumping this repo means bumping
both `ref` and `expect_ref` together, in every step's template, to the same commit.

To run a distillation job against **your own** teacher/student models and dataset,
override a recipe's parameters rather than editing anything here:

```bash
gb build start \
  -f recipes/granite4-gold-distillation/lsf/gold-smoke/build.yaml \
  --space <your-space> \
  --param TEACHER_MODEL=/path/to/teacher \
  --param STUDENT_MODEL=/path/to/student \
  --param TRAINING_DATASET=/path/to/corpus.jsonl
```

or copy one of granite.build's own recipes — `recipes/granite4-gold-distillation/lsf/`
and `recipes/granite4-350m/lsf/` both have working `build.yaml` + `parameters.yaml`
pairs — and edit the copy's `parameters.yaml` directly.

## Provenance

Some code comments reference companion checks, launchers, and scripts that aren't
included in this repo — they document engineering rationale and validation history from
the environment this code was developed and run in, not runtime dependencies. Every file
here runs as part of a granite.build step; none of them depend on anything outside this
repo and the third-party packages each step's `pyproject.toml` declares.
