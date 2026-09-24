"""Turn a GOLD training output directory into a publishable HF model directory.

This step is a SELECTOR/PRUNER, not a format converter. That is a verified property, not
an assumption: the DeepSpeed config used by `distill-gold-train` sets
`zero3_save_16bit_model: true`, so `Trainer.save_model` already writes HF-native
`model.safetensors` at the checkpoint root. Confirmed directly on job 1136274's
checkpoint-25 -- a 704786224-byte `model.safetensors` (350M params x 2 bytes) sitting
beside the DeepSpeed `global_step25/` shard tree.

So there are exactly four jobs here, and none of them is a weight conversion:

  1. SELECT   which checkpoint-N is the release. The trainer does not decide this.
  2. PRUNE    the resumable-run state (DeepSpeed shards, optimizer, scheduler, RNG). This
              is also nearly all of the size difference -- ZeRO optimizer state is several
              times the weights.
  3. NORMALISE the DEFAULTS the trainer saved for its own use, of which there are two and
              both are about what a consumer gets when it asks for nothing:
                - `gold.py:345` sets `padding_side="left"` because the trainer generates.
                  That is right for generation and wrong as a published default, since a
                  right-padding consumer trusting `tokenizer_config.json` will silently pad
                  on the wrong side.
                - `chat_template.jinja:13` defaults `enable_thinking` to True, so
                  `apply_chat_template(..., add_generation_prompt=True)` hands out the
                  REASONING prompt. For a student distilled on a corpus with no think traces
                  that is the prefix the weights never saw. `--chat-template-thinking default-off` flips
                  it; see THINKING_POLICIES, which also records why this step refuses to
                  delete the empty `<think></think>` blocks themselves.
              Both are OPT-IN or reported, never silent: this step's contract is that the
              export manifest says exactly which defaults it changed.
  4. ASSERT   the pruned directory actually loads.

Deliberately NOT a job here: grafting in a tokenizer. `gold.py:454` passes
`processing_class=tokenizer` and loads it from the *student* path (`gold.py:341`), so each
checkpoint is self-describing and carries the retagged tokenizer the corpus was built
with. The step confirms that rather than repairing it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# The shared step-completion contract, vendored into this step's image (see Dockerfile). A hard
# import and not a try/except: a missing gate would silently turn resume off for this step, and a
# step that re-exports a 700 MB directory every time it is asked is the cheaper failure to notice
# than one that skips work it should have done.
from gb_steps_post_training.distillation import export_state, step_state

# Files that make up a published model. An explicit KEEP list rather than a DROP list:
# with a DROP list, anything a future transformers/TRL release starts writing would be
# published by default, and this directory is the thing users download.
KEEP_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",  # present only when the weights are sharded
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
)

# Sharded weights: model-00001-of-0000N.safetensors. Matched by prefix+suffix rather than
# listed, since N is not known ahead of time.
KEEP_SHARD_PREFIX = "model-"
KEEP_SHARD_SUFFIX = ".safetensors"

# Resumable-run state. Named explicitly so the step's own report can say what it dropped
# and why, rather than "everything not in KEEP_FILES".
PRUNE_KNOWN = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
    "latest",
)
PRUNE_DIR_PREFIXES = ("global_step",)

# Load-time kwargs that transformers persists into tokenizer_config.json. Neither is
# tokenizer configuration; both were observed in job 1136274's checkpoint-25
# (`local_files_only` and `is_local`), and `is_local` was already present in the student
# source. Publishing a model whose tokenizer config pins local_files_only=true is wrong.
STRIP_TOKENIZER_KEYS = ("local_files_only", "is_local")

PADDING_SIDES = ("right", "left", "keep")

# The published chat template's THINKING POLICY.
#
# `chat_template.jinja:13` reads
#   {%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
# so a consumer that calls apply_chat_template(..., add_generation_prompt=True) and says
# nothing about thinking gets the REASONING generation prompt:
#   enable_thinking=True   ->  '<|im_start|>assistant\n<think>\n'      (template line 191)
#   enable_thinking=False  ->  '<|im_start|>assistant\n<think></think>' (template line 193)
#
# For a student distilled on a corpus with no think traces those are not equivalent, and the
# DEFAULT is the wrong one. Every labelled assistant span in that corpus begins with the
# literal `<think></think>` -- template line 90 prepends it to any assistant turn carrying no
# think markers, which job 1141677 measured as 120/120 spans. So the model was trained to emit
# `</think>` immediately after `<think>`, and was never once conditioned on `<think>\n`: the
# newline between the two markers does not occur anywhere in its training data. `default-off`
# flips line 13 so the published default is the shape the weights actually saw.
#
# WHY THERE IS NO `strip-injection` POLICY. The obvious reading of "strip the empty think
# block" is to delete line 90's injection, and that would be a defect. `<think></think>` is
# written at FIVE sites in this template, and they are not one feature:
#
#   line  90  INJECTOR     prepends it to a plain assistant turn with no think markers.
#   line 113  CANONICALISER tool-call turn, truncated history: rebuilds the turn with an empty
#                          block in place of the real trace.
#   line 119  INJECTOR     tool-call turn whose content is not a string: emits a bare block.
#   line 147  CANONICALISER plain turn, truncated history: same collapse as 113.
#   line 193  PROMPT       the enable_thinking=False generation prompt.
#
# `truncate_history_thinking` also defaults True (line 18), so 113 and 147 fire on their own
# for every assistant turn before the last user turn. Deleting only line 90 therefore renders
# ONE conversation two ways -- older turns keep the empty block via 113/147, the newest turn
# loses it, and the generation prompt still emits it at 193 -- which is a prefix shape no
# training example had. The empty block is load-bearing structure in this template, not
# decoration, so this step will not remove it. It changes which DEFAULT the consumer lands on
# and records that it did.
THINKING_POLICIES = ("keep", "default-off")

# NOT to be confused with distill-corpus-prep's `think_policy` (keep | strip | require),
# which is a decision about the TRAINING DATA -- whether reasoning traces are kept in the
# supervised targets at all. This one edits a DEFAULT in the published chat template and
# touches no data. They are related in one direction only: a corpus prepared with
# think_policy=strip produces a student that should be published with
# chat_template_thinking=default-off. The flag is spelled in full for that reason -- two
# neighbouring step keys called `thinking` and `think_policy` is how a recipe author sets
# the wrong one.

# Matched after stripping surrounding whitespace, and required to occur exactly once. A
# substring match on `enable_thinking` would also hit line 190's `{%- if enable_thinking %}`,
# and this repo has already been bitten by identifying a thing by substring.
_THINKING_LINE_ON = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
)
_THINKING_LINE_OFF = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}"
)

# Invariant guard for the five sites above: `default-off` must not change how many times the
# empty block is written. If a future edit to this function starts deleting them, this is what
# says so, at export time, instead of the published model doing it quietly.
_EMPTY_THINK = "<think></think>"


class ExportError(Exception):
    """Raised for any condition under which this step must not publish."""


def select_checkpoint(train_output_dir: Path, which: str = "latest") -> Path:
    """Pick the checkpoint to publish.

    `which` is "latest", a bare "checkpoint-N", or a path. "latest" means highest step
    number, NOT newest mtime: a resumed run rewrites older checkpoints' mtimes, so mtime
    ordering can select a checkpoint that is not the furthest along.
    """
    if which and which not in ("latest",):
        cand = Path(which)
        if not cand.is_absolute():
            cand = train_output_dir / which
        if not cand.is_dir():
            raise ExportError(f"requested checkpoint does not exist: {cand}")
        return cand

    if not train_output_dir.is_dir():
        raise ExportError(f"train_output_dir is not a directory: {train_output_dir}")

    found = [p for p in train_output_dir.glob("checkpoint-*") if p.is_dir()]
    if not found:
        raise ExportError(
            f"no checkpoint-* directory under {train_output_dir}. Nothing to export -- "
            "either training never reached save_steps, or output_dir is wrong."
        )

    def step_of(p: Path) -> int:
        tail = p.name.split("-", 1)[1]
        if not tail.isdigit():
            raise ExportError(
                f"cannot order checkpoint directory {p.name!r}: expected checkpoint-<int>."
            )
        return int(tail)

    return max(found, key=step_of)


def _is_kept(name: str) -> bool:
    if name in KEEP_FILES:
        return True
    # Sharded weights, e.g. model-00002-of-00003.safetensors.
    return name.startswith(KEEP_SHARD_PREFIX) and name.endswith(KEEP_SHARD_SUFFIX)


def classify(checkpoint: Path) -> dict[str, list[str]]:
    """Split a checkpoint's entries into keep / prune / unknown.

    "unknown" exists so that a file neither list anticipated is SURFACED rather than
    silently published or silently dropped. Both silent outcomes are wrong for a directory
    users download.
    """
    keep: list[str] = []
    prune: list[str] = []
    unknown: list[str] = []
    for entry in sorted(checkpoint.iterdir(), key=lambda p: p.name):
        name = entry.name
        if entry.is_dir():
            if any(name.startswith(pfx) for pfx in PRUNE_DIR_PREFIXES):
                prune.append(name)
            else:
                unknown.append(name)
        elif _is_kept(name):
            keep.append(name)
        elif name in PRUNE_KNOWN:
            prune.append(name)
        else:
            unknown.append(name)
    return {"keep": keep, "prune": prune, "unknown": unknown}


def normalise_tokenizer_config(
    raw: dict[str, Any], *, padding_side: str = "right"
) -> tuple[dict[str, Any], list[str]]:
    """Return (normalised config, list of human-readable changes).

    Returns changes rather than logging them so the caller can record exactly what it did
    in the export manifest -- "record that it did" is part of this step's contract.
    """
    if padding_side not in PADDING_SIDES:
        raise ExportError(
            f"padding_side={padding_side!r} is not one of {list(PADDING_SIDES)}."
        )

    out = dict(raw)
    changes: list[str] = []

    for key in STRIP_TOKENIZER_KEYS:
        if key in out:
            del out[key]
            changes.append(f"stripped {key} (a load-time kwarg, not tokenizer config)")

    if padding_side != "keep":
        before = out.get("padding_side")
        if before != padding_side:
            out["padding_side"] = padding_side
            changes.append(
                f"padding_side {before!r} -> {padding_side!r} "
                "(the trainer sets 'left' because it generates; see gold.py:345)"
            )

    return out, changes


def normalise_chat_template(
    text: str, *, chat_template_thinking: str = "keep"
) -> tuple[str, list[str]]:
    """Return (normalised template, list of human-readable changes).

    `chat_template_thinking` is one of THINKING_POLICIES; see the block comment on THINKING_POLICIES for
    what each one means and, more importantly, for the policy that is deliberately absent.

    Refuses rather than repairs. A template whose `enable_thinking` default line is missing or
    duplicated is not a template this function understands, and editing it on a guess would
    change the published model's generation prompt -- the one string every consumer's first
    token is conditioned on.
    """
    if chat_template_thinking not in THINKING_POLICIES:
        raise ExportError(
            f"chat_template_thinking={chat_template_thinking!r} is not one of "
            f"{list(THINKING_POLICIES)}."
        )
    if chat_template_thinking == "keep":
        return text, []

    lines = text.splitlines(keepends=True)
    hits_on = [i for i, ln in enumerate(lines) if ln.strip() == _THINKING_LINE_ON]
    hits_off = [i for i, ln in enumerate(lines) if ln.strip() == _THINKING_LINE_OFF]

    if len(hits_on) + len(hits_off) != 1:
        raise ExportError(
            f"chat_template.jinja: expected exactly one `enable_thinking` default line, "
            f"found {len(hits_on)} defaulting True and {len(hits_off)} defaulting False. "
            "This step will not guess which one sets the published generation prompt. "
            "Expected, ignoring indentation:\n"
            f"  {_THINKING_LINE_ON}"
        )
    if hits_off:
        return text, [
            f"enable_thinking already defaults to False (line {hits_off[0] + 1}); "
            "policy 'default-off' had nothing to change"
        ]

    i = hits_on[0]
    before = lines[i]
    indent = before[: len(before) - len(before.lstrip())]
    newline = before[len(before.rstrip("\r\n")):]
    lines[i] = indent + _THINKING_LINE_OFF + newline
    out = "".join(lines)

    # The five-site invariant. `default-off` edits one default; it must not add or remove any
    # empty think block.
    got, want = out.count(_EMPTY_THINK), text.count(_EMPTY_THINK)
    if got != want:
        raise ExportError(
            f"internal: 'default-off' changed the number of literal {_EMPTY_THINK} sites "
            f"from {want} to {got}. That policy is only allowed to flip one default; the "
            "empty block is load-bearing structure in this template (see THINKING_POLICIES)."
        )

    return out, [
        f"enable_thinking default True -> False (line {i + 1}); the generation prompt a "
        "consumer gets without asking becomes '<|im_start|>assistant\\n<think></think>', "
        "which is the prefix every labelled span in the training corpus began with"
    ]


def export(
    checkpoint: Path,
    dest: Path,
    *,
    padding_side: str = "right",
    chat_template_thinking: str = "keep",
    allow_unknown: bool = False,
) -> dict[str, Any]:
    """Copy the publishable subset of `checkpoint` into `dest`, normalising as it goes."""
    if not checkpoint.is_dir():
        raise ExportError(f"checkpoint is not a directory: {checkpoint}")

    parts = classify(checkpoint)
    if "model.safetensors" not in parts["keep"] and not any(
        n.startswith(KEEP_SHARD_PREFIX) and n.endswith(KEEP_SHARD_SUFFIX)
        for n in parts["keep"]
    ):
        raise ExportError(
            f"{checkpoint} has no model.safetensors (nor sharded model-*.safetensors). "
            "Either zero3_save_16bit_model was not set on the training run -- in which "
            "case the weights exist only inside global_step*/ and this step genuinely "
            "would need a converter -- or this is not a checkpoint directory."
        )
    if parts["unknown"] and not allow_unknown:
        raise ExportError(
            f"unrecognised entries in {checkpoint}: {parts['unknown']}. This step refuses "
            "to guess whether they belong in a published model. Add them to KEEP_FILES or "
            "PRUNE_KNOWN in export_hf_model.py, or pass --allow-unknown to drop them."
        )

    dest.mkdir(parents=True, exist_ok=True)
    changes: list[str] = []
    for name in parts["keep"]:
        src = checkpoint / name
        if name == "tokenizer_config.json":
            raw = json.loads(src.read_text())
            norm, tc_changes = normalise_tokenizer_config(raw, padding_side=padding_side)
            (dest / name).write_text(json.dumps(norm, indent=2, ensure_ascii=False) + "\n")
            changes.extend(f"tokenizer_config.json: {c}" for c in tc_changes)
        elif name == "chat_template.jinja":
            # Written rather than copy2'd only when the policy actually changes something, so
            # that `keep` leaves a byte-identical file with the checkpoint's own mtime -- a
            # rewritten-but-unchanged template would make `diff -r` against the checkpoint
            # report a difference this step did not make.
            raw_t = src.read_text()
            norm_t, ct_changes = normalise_chat_template(
                raw_t, chat_template_thinking=chat_template_thinking)
            if norm_t == raw_t:
                shutil.copy2(src, dest / name)
            else:
                (dest / name).write_text(norm_t)
            changes.extend(f"chat_template.jinja: {c}" for c in ct_changes)
        else:
            shutil.copy2(src, dest / name)

    manifest = {
        "source_checkpoint": str(checkpoint),
        "kept": parts["keep"],
        "pruned": parts["prune"],
        "dropped_unrecognised": parts["unknown"] if allow_unknown else [],
        "normalisations": changes,
        "padding_side_policy": padding_side,
        "chat_template_thinking_policy": chat_template_thinking,
    }
    (dest / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def verify(dest: Path, *, expect_tokenizer_from: Path | None = None) -> list[str]:
    """Load the exported dir and confirm it works. Imports transformers LAZILY so that
    classify/normalise/export stay unit-testable on a CPU box with no transformers."""
    notes: list[str] = []
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415

    AutoConfig.from_pretrained(str(dest))
    tok = AutoTokenizer.from_pretrained(str(dest))
    notes.append(f"tokenizer loaded: {type(tok).__name__}, vocab={len(tok)}")

    if expect_tokenizer_from is not None:
        ref = AutoTokenizer.from_pretrained(str(expect_tokenizer_from))
        probe = "<|im_start|>user\nhello<|im_end|>\n"
        got = tok(probe, add_special_tokens=False)["input_ids"]
        want = ref(probe, add_special_tokens=False)["input_ids"]
        if got != want:
            raise ExportError(
                "exported tokenizer does not agree with the expected (retagged student) "
                f"tokenizer at {expect_tokenizer_from}: {got[:12]} != {want[:12]}. The "
                "corpus was tokenized with one tokenizer; publishing another silently "
                "changes what the model was trained to expect."
            )
        notes.append(f"tokenizer identity matches {expect_tokenizer_from}")
    return notes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-output-dir", required=True)
    p.add_argument("--dest", required=True)
    # Empty string is accepted as a synonym for 'latest' so the step-template can pass
    # every flag UNCONDITIONALLY. The alternative -- wrapping optional flags in jinja
    # {% if %} inside a backslash-continued shell command -- puts a line continuation
    # next to a conditionally-empty line, where one flag going empty silently swallows
    # the following flag. Templates should not have to be clever about whitespace.
    p.add_argument(
        "--checkpoint", default="latest",
        help="'latest' or '' (highest step number), a bare checkpoint-N, or a path.",
    )
    p.add_argument("--padding-side", default="right", choices=PADDING_SIDES)
    # Defaults to 'keep': which generation prompt a published model hands out is a decision
    # about the artifact, and this step's job is to make it explicit and recorded, not to make
    # it. 'default-off' is the right value for a student distilled on a corpus with no think
    # traces -- see THINKING_POLICIES for the measurement and for why "just strip the empty
    # block" is not on this list.
    p.add_argument("--chat-template-thinking", default="keep",
                   choices=THINKING_POLICIES)
    # BooleanOptionalAction, for the same reason: it gives --allow-unknown/--no-allow-unknown
    # and --verify/--no-verify, so a template renders one or the other and never omits
    # the line. It also makes an explicit `--no-verify` in a recipe readable as a
    # decision rather than as an oversight.
    p.add_argument("--allow-unknown", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--expect-tokenizer-from", default="",
        help="Retagged student dir. If given, --verify asserts token-id agreement with it.",
    )
    p.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=False,
        help="Load the exported dir (needs transformers).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ckpt = select_checkpoint(Path(args.train_output_dir), args.checkpoint)
        dest = Path(args.dest)
        ref = Path(args.expect_tokenizer_from) if args.expect_tokenizer_from else None

        # THE RESUME GATE. Inline rather than in a wrapper script because this step has no
        # wrapper -- both launchers in step-template.yaml invoke this file directly -- so the
        # gate can only drift out of step with the export it guards if it is somewhere else.
        #
        # classify() is called here AND again inside export(). That is one listdir, deliberately
        # not factored out: the alternative is to pass parts into export() and have its own
        # contract depend on a caller doing the classification, which is worse for a function
        # that is also called from tests.
        parts = classify(ckpt)
        expectation = export_state.expectation(
            checkpoint=ckpt,
            keep=parts["keep"],
            dropped_unrecognised=parts["unknown"] if args.allow_unknown else [],
            padding_side=args.padding_side,
            chat_template_thinking=args.chat_template_thinking,
            verify=args.verify,
            expect_tokenizer_from=ref,
        )
        declared = export_state.declared_outputs(parts["keep"])
        decision = step_state.decide(dest, export_state.STEP_NAME, expectation, declared)
        for line in decision.lines:
            print(f"  {line}")
        if decision.kind == step_state.SKIP:
            # Exit 0, not 64. The 64 is step_state's internal vocabulary; the contract with the
            # launcher is that a successful no-op returns 0 so the LLMB_ARTIFACT_ID line after
            # this call still runs. A step that skips and exits non-zero starves its consumer.
            print(f"nothing to do: {dest} already holds this export.")
            return 0
        if decision.kind == step_state.REFUSE:
            print(f"FATAL [{export_state.STEP_NAME}]: refusing to overwrite {dest}. "
                  "See the key named above.", file=sys.stderr)
            return 2

        manifest = export(
            ckpt, Path(args.dest),
            padding_side=args.padding_side,
            chat_template_thinking=args.chat_template_thinking,
            allow_unknown=args.allow_unknown,
        )
        print(f"exported {ckpt} -> {args.dest}")
        print(f"  kept   : {len(manifest['kept'])} file(s)")
        print(f"  pruned : {manifest['pruned']}")
        for c in manifest["normalisations"]:
            print(f"  normalised: {c}")
        if args.verify:
            for note in verify(dest, expect_tokenizer_from=ref):
                print(f"  verified: {note}")
        # Marked LAST, after verify(). An export that cannot be loaded must not be recorded as
        # complete -- otherwise the next run SKIPs and the broken directory becomes permanent.
        # The declared outputs are re-derived from the MANIFEST's kept list rather than reused
        # from `declared` above, so that what is recorded is what export() actually wrote.
        step_state.write_marker(dest, export_state.STEP_NAME, expectation,
                                export_state.declared_outputs(manifest["kept"]))
        print(f"  marked : {step_state.MARKER_NAME} written ({len(manifest['kept']) + 1} outputs)")
    except ExportError as exc:
        print(f"FATAL [distill-hf-export]: {exc}", file=sys.stderr)
        return 2
    # No LLMB_ARTIFACT_ID line here: the step-template's launcher already emits it, and this
    # script printing it too meant ONE run announced the same id TWICE. Artifact emission
    # belongs to the launcher, because that is where the declared name lives -- keeping the
    # echo next to the outputs block is what stops the two from drifting apart. (Same
    # reasoning removed a duplicate from build_overlay.py, where the problem was first
    # measured: LSF job 1137372.)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
