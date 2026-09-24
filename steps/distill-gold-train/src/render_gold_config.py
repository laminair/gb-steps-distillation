"""Validate distill-gold-train's step config and render it as a GOLD YAML file.

WHY THIS IS A MODULE AND NOT INLINE IN run-gold.sh. Everything here is a decision that
can be wrong in a way that costs a full multi-node training run -- an invalid loss-arm
combination, a lmbda/topology contradiction, a corpus built with a different tokenizer
than the student. Those are exactly the checks worth testing, and a heredoc inside a
launcher cannot be tested. run-gold.sh is deliberately thin around this.

The output is consumed by trl's TrlParser via `gold.py --config <file>`, so the dataclass
(CustomGOLDConfig) stays the single authority on the full key surface: anything this
renderer emits that the dataclass does not accept is a hard parse error there rather than
a silently ignored key. This module's job is only to (a) map the step's explicit contract
onto dataclass field names, (b) reject combinations the dataclass would reject late or,
worse, accept and train wrongly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gb_steps_post_training.distillation import render_common, tokenizer_identity
from typing import Any

import yaml

# ---------------------------------------------------------------- loss arms
#
# Loss arms, expressed as the boolean sets CustomGOLDConfig actually takes.
#
# The step exposes ONE enum rather than a dozen booleans because the booleans are not
# orthogonal, and their non-orthogonality has two very different flavours:
#
#   (a) COMBINATIONS THAT CRASH. CustomGOLDConfig.__post_init__ rejects several. A config
#       setting two of them is not "more distillation", it is a ValueError several minutes
#       into a run that has already allocated its nodes.
#
#   (b) COMBINATIONS THAT SILENTLY TRAIN THE WRONG OBJECTIVE, which is worse, because the
#       run SUCCEEDS. `use_liger_fused_jsd` sets use_liger_gkd_loss
#       (custom_gold_trainer.py:1535) and that gates an ENTIRELY SEPARATE branch at :3304
#       which never reaches the primary-loss chain at :3452. The liger loss object is
#       constructed with beta, alpha, temperature and use_kl_interpolation ONLY
#       (:1541-1551). So liger together with use_ce_loss, use_distillm2,
#       use_distillm2_like, use_adaptive_kld or use_reversed_distillm2_like ignores the
#       second flag entirely and says nothing. __post_init__ catches exactly ONE of those
#       five (sampled_opd, via an explicit incompatibility); the other four pass validation
#       and train fused generalized JSD while the config claims another objective.
#
# An enum cannot express either kind of invalid combination. That is the whole argument for
# the enum, and (b) is why this table is the ONLY place these booleans are ever set.
#
# WHERE THE PRIMARY LOSS IS ACTUALLY DECIDED, read off custom_gold_trainer.py so this table
# cannot drift from it silently. A companion check asserts every line of this map against
# the dataclass and against the trainer source:
#
#   if   use_liger_gkd_loss      -> :3304  fused liger path; honours beta/alpha/temp/kl_interp
#   elif use_ce_loss             -> :3452  cross entropy only, no teacher term
#   elif use_sampled_opd_loss    -> :3457  REINFORCE-style on-policy distillation
#   elif use_distillm2           -> :3487  DistiLLM-2, comparative (per-TOKEN policy mask)
#   else generalized_jsd_loss    -> :3496, which sub-dispatches on a WHOLE-MICROBATCH
#        scalar (`on_policy=any(inputs["on_policy"])`, :3510 -- note the any(), it is not
#        the per-token mask distillm2 gets):
#          if   use_distillm2_like          -> :2983
#          elif use_reversed_distillm2_like -> :3003
#          elif use_adaptive_kld            -> :3048
#          else  beta == 0                  -> pure FORWARD KL
#                beta == 1                  -> pure REVERSE KL
#                use_kl_interpolation       -> convex (1-beta)*FKL + beta*RKL
#                otherwise                  -> mixture-distribution generalized JSD
#
# WHAT IS DELIBERATELY NOT AN ARM, because `beta` already reaches it: forward KL is `jsd`
# at beta 0.0, reverse KL is `jsd` at beta 1.0. Arms for those would give two ways to say
# one thing and let them disagree.
#
# ULD IS NOT IN THAT CHAIN AT ALL. use_uld_loss selects a separate path (:3105, :3534)
# comparing SORTED logit distributions, so teacher and student need not share a vocabulary.
# It is the only arm here that works for a genuinely cross-tokenizer pair -- the arm to
# reach for when pairing Granite with a non-Granite teacher. For a Granite/Granite pair it
# buys nothing over an aligned comparison; see
# recipes/granite41-3b/distill-uld-granite42-30b/README.md.
#
# EACH ARM CARRIES ITS OWN REQUIREMENTS rather than having them coerced. Two kinds appear:
# rules __post_init__ enforces (so failing here only saves the allocation), and rules that
# exist because outside them the arm is not a distinct objective at all -- it silently
# becomes another arm. The second kind is the one nothing downstream would catch.
LOSS_ARMS: dict[str, dict[str, Any]] = {
    "jsd": {
        "flags": {},
        "doc": "mixture-distribution generalized JSD; beta 0.0 = forward KL, 1.0 = reverse KL",
    },
    "kl_interpolation": {
        "flags": {"use_kl_interpolation": True},
        "doc": "convex (1-beta)*FKL + beta*RKL instead of the mixture JSD",
        # Not a hard requirement, but the arm is pointless outside it: at beta 0 or 1 the
        # chain short-circuits to pure FKL/RKL BEFORE reaching the interpolation branch
        # (:3089-3092 precede :3095), so the flag is dead and this is exactly `jsd`.
        "requires": {"beta_strictly_between": (0.0, 1.0)},
        "why": (
            "the interpolation branch sits AFTER the beta == 0 and beta == 1 "
            "short-circuits, so at those values the flag is never read and this arm is "
            "byte-identical to loss_arm='jsd'"
        ),
    },
    "adaptive_kld": {
        "flags": {"use_adaptive_kld": True},
        "doc": "adaptive KL divergence weighting inside generalized_jsd_loss",
    },
    "liger_fused_jsd": {
        "flags": {"use_liger_fused_jsd": True},
        "doc": "same JSD math via Liger's fused linear kernel; lower peak memory",
    },
    "liger_fused_kl_interpolation": {
        "flags": {"use_liger_fused_jsd": True, "use_kl_interpolation": True},
        # The ONE composition with liger that is genuinely honoured: the liger loss object
        # takes use_kl_interpolation as a constructor argument (:1550). Every other
        # secondary flag is silently dropped by that path, which is why no other liger
        # combination is offered as an arm.
        "doc": "fused liger kernel computing the convex FKL/RKL interpolation",
        "requires": {"beta_strictly_between": (0.0, 1.0)},
        "why": (
            "LigerFusedLinearSkewedJSDLoss receives beta at construction (:1544) and the "
            "interpolation is only meaningful strictly between the two pure divergences"
        ),
    },
    "distillm2": {
        "flags": {"use_distillm2": True},
        "doc": "DistiLLM-2, comparative: reverse KL on on-policy tokens, forward KL on off-policy",
        # Both terms are guarded by `.any()` (:2889, :2901), so outside 0 < lmbda < 1 this
        # does not crash -- it degenerates. At lmbda 0.0 every token is off-policy and the
        # loss IS forward KL (= jsd at beta 0.0); at 1.0 every token is on-policy and it IS
        # reverse KL (= jsd at beta 1.0). A silent equivalence, not an error, so nothing
        # downstream would report it.
        "requires": {"lmbda_strictly_between": (0.0, 1.0)},
        "why": (
            "the comparative loss needs BOTH policy classes present in a batch; at "
            "lmbda 0.0 it degenerates to forward KL and at 1.0 to reverse KL, either of "
            "which loss_arm='jsd' expresses directly at beta 0.0 / 1.0"
        ),
    },
    "distillm2_like": {
        "flags": {"use_distillm2_like": True},
        "doc": "DistiLLM-2 shape, non-comparative; policy class decided per MICROBATCH",
        "requires": {"lmbda_strictly_between": (0.0, 1.0)},
        "why": (
            "the on/off-policy branch is what distinguishes this arm; outside 0 < lmbda < 1 "
            "one side is unreachable and the arm reduces to a single fixed divergence"
        ),
    },
    "reversed_distillm2_like": {
        "flags": {"use_reversed_distillm2_like": True},
        # A SIBLING elif of distillm2_like (:3003), not a modifier of it -- setting both
        # would make this one dead code, which is why it is a separate arm with the flag
        # alone rather than a pair.
        "doc": "distillm2_like with the on/off-policy divergence roles exchanged",
        "requires": {"lmbda_strictly_between": (0.0, 1.0)},
        "why": (
            "same reason as distillm2_like: the exchanged roles are only observable when "
            "both policy classes occur"
        ),
    },
    "sampled_opd": {
        "flags": {"use_sampled_opd_loss": True},
        "doc": "REINFORCE-style policy gradient on sampled tokens; optional truncated IS",
        # These are __post_init__'s rules (custom_gold_config.py:307-312), surfaced rather
        # than coerced: quietly rewriting a user's lmbda to satisfy a loss arm would change
        # the experiment without saying so.
        "requires": {"lmbda_eq": 1.0, "last_message_only": True},
        "why": "CustomGOLDConfig.__post_init__ demands both (custom_gold_config.py:307-312)",
    },
    "uld": {
        "flags": {"use_uld_loss": True},
        "doc": "Universal Logit Distillation over sorted logits; the cross-tokenizer arm",
    },
    "ce": {
        "flags": {"use_ce_loss": True},
        # An SFT control that runs INSIDE this trainer, on the same data path, collator and
        # masking as every distillation arm. That is its whole value over
        # distill-sft-baseline: the only difference from a jsd run is the loss term, so a
        # gap between them cannot be a data-pipeline artefact. It is not a replacement for
        # distill-sft-baseline, which trains without a teacher resident at all.
        "doc": "cross entropy only -- in-trainer SFT control, teacher still loaded",
        "requires": {"lmbda_eq": 0.0},
        "why": (
            "with no teacher term in the loss, on-policy generation would train the "
            "student on its own samples with no teacher signal at all -- and a control "
            "that trains on its own output is not a control. Note the cost this arm pays "
            "regardless: use_ce_loss does not unload the teacher "
            "(custom_gold_config.py:27-30), so it holds teacher memory it never reads"
        ),
    },
}


# Requirement kinds validate() knows how to enforce. Named here so loss-arms.py can assert
# that no arm declares a requirement the loop would skip -- see the comment in validate().
_REQUIREMENT_KINDS = frozenset(
    {"lmbda_eq", "lmbda_strictly_between", "beta_strictly_between", "last_message_only"}
)


# Resume modes. gold.py auto-resumes on the mere PRESENCE of output_dir/checkpoint-*
# (gold.py:507-510) -- there is no flag there to ask for anything else. That default is
# right for a preempted run (proven directly, having resumed across two preemptions)
# and wrong for a recipe: re-running with a changed hyperparameter into the same output_dir
# silently restores the OLD optimizer and scheduler state under the NEW config, which looks
# like a successful run.
#
# All three modes below are enforced as PREFLIGHT VALIDATION, so none of them requires
# patching gold.py -- we only ever decide whether to start it, never what it does:
#   auto    -- gold.py's own behaviour, unchanged. Resume if a checkpoint is there.
#   never   -- FAIL if a checkpoint is present. For a fresh experiment: makes the operator
#              repoint output_dir rather than get a silent hybrid. Deliberately does NOT
#              delete the checkpoint -- destroying training output to satisfy a config flag
#              is not this step's decision to make.
#   require -- FAIL if a checkpoint is ABSENT. Turns "the checkpoint I meant to resume from
#              was missing, so I trained from scratch for six hours" into an immediate error.
RESUME_MODES = render_common.RESUME_MODES

# Shared, not local. WHY: distill-sft-baseline is the CONTROL for this step, and a control
# validated by a second, slightly different copy of these rules is not a control. The two
# steps must stay separate (separate images, separate configs -- a control that shares a step
# with the treatment is a control you can silently mis-configure into the treatment), but the
# guards they share must be one implementation. They live in the package, which
# steps/*/src can import (see the import above); the aliases below keep this module's
# call sites and its tests unchanged.
ConfigError = render_common.ConfigError
_corpus_tokenizer_identity = render_common.corpus_tokenizer_identity
_student_tokenizer_identity = render_common.student_tokenizer_identity
_existing_checkpoints = render_common.existing_checkpoints


def validate(args: argparse.Namespace, *, check_paths: bool = True) -> None:
    """Raise ConfigError on any config this step must not run."""
    if args.loss_arm not in LOSS_ARMS:
        raise ConfigError(
            f"loss_arm={args.loss_arm!r} is not one of {sorted(LOSS_ARMS)}."
        )

    if not 0.0 <= args.lmbda <= 1.0:
        raise ConfigError(f"lmbda must be in [0, 1]; got {args.lmbda}.")
    if not 0.0 <= args.beta <= 1.0:
        raise ConfigError(f"beta must be in [0, 1]; got {args.beta}.")

    # lmbda is a topology switch, not only a loss weight. Catch the contradiction here,
    # where it costs nothing, rather than in a launcher that has already been allocated.
    if args.lmbda == 0.0 and args.vllm_num_servers > 0 and args.vllm_mode == "server":
        raise ConfigError(
            f"lmbda=0.0 is off-policy: the student never generates, so no vLLM server is "
            f"needed -- but vllm_num_servers={args.vllm_num_servers} with vllm_mode="
            f"'server' asks for {args.vllm_num_servers} node(s) that would each be taken "
            "away from the trainer. Set lmbda > 0.0 for the on-policy arm, or set "
            "vllm_num_servers=0 for the off-policy one."
        )
    if args.resume not in RESUME_MODES:
        raise ConfigError(
            f"resume={args.resume!r} is not one of {list(RESUME_MODES)}."
        )

    if args.lmbda > 0.0 and args.vllm_mode == "server" and args.vllm_num_servers < 1:
        raise ConfigError(
            f"lmbda={args.lmbda} is on-policy with vllm_mode='server', which requires at "
            "least one server node; got vllm_num_servers=0."
        )
    if args.vllm_mode not in ("server", "colocate"):
        raise ConfigError(f"vllm_mode must be 'server' or 'colocate'; got {args.vllm_mode!r}.")

    # Per-arm requirements, enforced from the arm table rather than written out per arm.
    #
    # WHY GENERIC AND NOT ONE BLOCK PER ARM. There are eleven arms and six of them carry a
    # requirement. Written by hand that is six near-identical blocks, and the failure mode
    # of six near-identical blocks is that a SEVENTH arm gets added with a "requires" entry
    # and no block to read it -- a requirement that exists in the table, reads as enforced,
    # and is not. Driving the loop off the table itself makes that impossible: an arm cannot
    # declare a requirement this loop does not check. loss-arms.py asserts the converse,
    # that every requirement kind named in the table is one this loop understands.
    #
    # NOTHING HERE IS COERCED. Every requirement is reported and refused. Two of them are
    # rules CustomGOLDConfig.__post_init__ would raise on anyway, so failing here only saves
    # the allocation; the other four exist because outside them the arm SILENTLY BECOMES A
    # DIFFERENT ARM, which __post_init__ has no reason to object to and no downstream step
    # could detect. That second class is the reason this loop is worth its length.
    _arm = LOSS_ARMS[args.loss_arm]
    _req = _arm.get("requires", {})
    _why = _arm.get("why", "")
    _tail = f" WHY: {_why}." if _why else ""
    for _kind, _bound in _req.items():
        if _kind == "lmbda_eq" and args.lmbda != _bound:
            raise ConfigError(
                f"loss_arm={args.loss_arm!r} requires lmbda={_bound}; got {args.lmbda}."
                f"{_tail} Set lmbda explicitly rather than having it changed for you."
            )
        elif _kind == "beta_strictly_between" and not (_bound[0] < args.beta < _bound[1]):
            raise ConfigError(
                f"loss_arm={args.loss_arm!r} requires {_bound[0]} < beta < {_bound[1]}; "
                f"got beta={args.beta}.{_tail}"
            )
        elif _kind == "lmbda_strictly_between" and not (_bound[0] < args.lmbda < _bound[1]):
            raise ConfigError(
                f"loss_arm={args.loss_arm!r} requires {_bound[0]} < lmbda < {_bound[1]}; "
                f"got lmbda={args.lmbda}.{_tail}"
            )
        elif _kind == "last_message_only" and bool(args.last_message_only) != bool(_bound):
            raise ConfigError(
                f"loss_arm={args.loss_arm!r} requires last_message_only={_bound}.{_tail}"
            )
        elif _kind not in _REQUIREMENT_KINDS:
            # Unreachable if loss-arms.py is green. Present anyway: a typo'd requirement key
            # would otherwise be a requirement that silently does not apply, which is the
            # exact failure this loop was written to make impossible.
            raise ConfigError(
                f"internal: loss_arm={args.loss_arm!r} declares requirement {_kind!r}, "
                "which validate() does not implement. Fix the arm table or this loop."
            )

    # Hidden-state matching is NOT an arm and is checked separately, because it is the one
    # loss term that COMPOSES: it is added on top of whatever the primary loss returned
    # (custom_gold_trainer.py:3515-3520), weighted gamma * (1 - current_lmbda). Making it an
    # arm would have forced one arm per (primary x hidden) pair.
    if args.hidden_loss_gamma < 0.0:
        raise ConfigError(
            f"hidden_loss_gamma must be >= 0; got {args.hidden_loss_gamma}. Use 0.0 to "
            "disable the auxiliary hidden-state loss."
        )
    # Parsed here, not in render(), so a typo is a ConfigError naming the flag rather than a
    # bare ValueError from a list comprehension one function later.
    for _tok in (t for t in args.hidden_loss_layers.split(",") if t.strip()):
        try:
            if int(_tok) < 0:
                raise ValueError
        except ValueError:
            raise ConfigError(
                f"--hidden-loss-layers must be comma-separated non-negative decoder layer "
                f"indices; got {_tok!r} in {args.hidden_loss_layers!r}."
            ) from None
    if args.hidden_loss_layers.strip() and args.hidden_loss_gamma <= 0.0:
        raise ConfigError(
            f"--hidden-loss-layers {args.hidden_loss_layers!r} was given with "
            f"hidden_loss_gamma={args.hidden_loss_gamma}, which disables the hidden-state "
            "loss entirely -- so the layer list would be silently ignored. Set a positive "
            "gamma or drop the layer list."
        )
    if args.hidden_loss_gamma > 0.0 and args.loss_arm.startswith("liger"):
        # The liger path computes its loss inside a fused kernel and captures only the LAST
        # decoder layer via an lm_head pre-hook (:3358). It does honour use_hidden_loss
        # (:3360), but hidden_loss_layers cannot select layers there -- so an explicit layer
        # list would be accepted and ignored. Refused rather than silently narrowed.
        if args.hidden_loss_layers.strip():
            raise ConfigError(
                f"loss_arm={args.loss_arm!r} cannot honour --hidden-loss-layers "
                f"{args.hidden_loss_layers!r}: the fused liger path captures only the last "
                "decoder layer, via an lm_head pre-hook (custom_gold_trainer.py:3358). "
                "Either drop the layer list (last layer only) or use a non-liger arm."
            )

    render_common.validate_topology(args.gpus_per_node, args.nodes)

    # Server nodes come OUT of the allocation, so the trainer needs at least one left.
    if args.lmbda > 0.0 and args.vllm_mode == "server":
        if args.vllm_num_servers >= args.nodes:
            raise ConfigError(
                f"vllm_num_servers={args.vllm_num_servers} with nodes={args.nodes} leaves "
                "no trainer node: every server node is taken away from the trainer. This "
                "is the same guard the reference launcher enforces at run time; failing "
                "here means it costs no allocation."
            )

    if not check_paths:
        return

    # Resume enforcement. Path-dependent, so it lives below the check_paths gate -- but
    # FIRST within it, because it is the check most likely to abort a run the operator did
    # not mean to start, and it costs one directory listing.
    render_common.validate_resume(args.resume, args.output_dir, entrypoint="gold.py")

    missing = []
    for label, value in (
        ("student_model_path", args.student_model_path),
        ("teacher_model_path", args.teacher_model_path),
        ("teacher_tokenizer_path", args.teacher_tokenizer_path),
    ):
        if not value:
            missing.append(f"{label} is empty")
        elif not Path(value).is_dir():
            missing.append(f"{label} is not a directory: {value}")
    if not args.corpus_path:
        missing.append("corpus_path is empty")
    elif not Path(args.corpus_path).exists():
        missing.append(f"corpus_path does not exist: {args.corpus_path}")
    if not Path(args.deepspeed_config).is_file():
        missing.append(f"deepspeed_config is not a file: {args.deepspeed_config}")
    if missing:
        raise ConfigError("input validation failed:\n  - " + "\n  - ".join(missing))

    # The retagged student must carry the chat template (the retag is what installs it, and
    # the collator's prompt/completion boundary depends on it), and there must be ONE
    # tokenizer per run for both sides. Both are shared with distill-sft-baseline: see
    # render_common.validate_student_against_corpus for why each is an assertion in the step
    # rather than a note in a README.
    render_common.validate_student_against_corpus(args.student_model_path, args.corpus_path)


def render(args: argparse.Namespace) -> dict[str, Any]:
    """Build the GOLD config mapping. Assumes validate() has already passed."""
    cfg: dict[str, Any] = {
        "model_name_or_path": args.student_model_path,
        "teacher_model_name_or_path": args.teacher_model_path,
        # Separate from the teacher directory ON PURPOSE -- the overlay is the tokenizer_class-trap
        # fix, and loading the teacher tokenizer from teacher_model_name_or_path instead
        # would silently reintroduce it.
        "teacher_tokenizer_name_or_path": args.teacher_tokenizer_path,
        "dataset_name": args.corpus_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "temperature": args.temperature,
        "vllm_temperature": args.vllm_temperature,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # Quoted in the step template and kept a string here so "1e-6" survives: YAML
        # parses 1e-6 as a float only if it is written 1.0e-6, and bare 1e-6 is a STRING
        # in YAML 1.1. Emitting it as a float removes that trap entirely.
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": args.num_train_epochs,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "last_message_only": args.last_message_only,
    }

    if args.lmbda > 0.0:
        cfg["use_vllm"] = True
        cfg["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "server":
            cfg["vllm_num_servers"] = args.vllm_num_servers
    else:
        # Explicit False rather than omitted: the off-policy arm's defining property is
        # that no generation happens, and relying on a default to express it means a
        # change to that default silently changes the arm.
        cfg["use_vllm"] = False

    cfg.update(LOSS_ARMS[args.loss_arm]["flags"])

    # Hidden-state matching, emitted only when actually asked for. use_hidden_loss is
    # derived from gamma rather than carried as its own flag, so the two cannot contradict
    # each other; gamma 0.0 emits nothing at all and leaves the dataclass default.
    if args.hidden_loss_gamma > 0.0:
        cfg["use_hidden_loss"] = True
        cfg["hidden_loss_gamma"] = args.hidden_loss_gamma
        if args.hidden_loss_layers.strip():
            # int() not str: hidden_loss_layers is Optional[list[int]] and the trainer
            # indexes output_hidden_states with these directly, so a list of strings would
            # fail deep inside compute_hidden_loss rather than here.
            cfg["hidden_loss_layers"] = [
                int(_x) for _x in args.hidden_loss_layers.split(",") if _x.strip()
            ]

    # ---- Experiment tracking. Emitted ONLY when non-empty, because step-template.yaml renders
    # every unset string key as "" and a key present-but-empty is a different statement from a
    # key absent. tracking.py treats "" as unset too, so this is belt and braces -- but it keeps
    # the rendered yaml honest, which is the file a human reads when a run logged nowhere.
    #
    # Note what is NOT here: a run name is not defaulted. Under granite.build this config is
    # rendered to the fixed name `gold-config.rendered.yaml`, so tracking.py cannot use the file
    # stem and composes the name from student/teacher/corpus/arm instead. Defaulting it here would
    # mean two places inventing run names.
    for _key, _val in (
        ("clearml_project", args.clearml_project),
        ("clearml_run_name", args.clearml_run_name),
        ("wandb_entity", args.wandb_entity),
        ("wandb_project", args.wandb_project),
        ("wandb_run_name", args.wandb_run_name),
    ):
        if (_val or "").strip():
            cfg[_key] = _val.strip()

    extra_raw = (args.extra_config_yaml or "").strip()
    if extra_raw:
        extra = yaml.safe_load(extra_raw)
        if not isinstance(extra, dict):
            raise ConfigError(
                f"extra_config_yaml must parse to a mapping; got {type(extra).__name__}."
            )
        # Merged UNDER the contract keys: a collision is an error, not a silent winner.
        # An escape hatch that can override an explicit contract key is not an escape
        # hatch, it is a second config system with undefined precedence.
        collisions = sorted(set(extra) & set(cfg))
        if collisions:
            raise ConfigError(
                "extra_config_yaml collides with keys this step sets explicitly: "
                f"{collisions}. Change the step's own config value instead -- these keys "
                "have one owner on purpose."
            )
        cfg.update(extra)

    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-model-path", required=True)
    p.add_argument("--teacher-model-path", required=True)
    p.add_argument("--teacher-tokenizer-path", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lmbda", type=float, required=True)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-temperature", type=float, default=1.0)
    p.add_argument(
        "--loss-arm",
        default="jsd",
        # No `choices=`: the explicit membership test in validate() names the offending value
        # AND lists the alternatives with what each one is, which argparse's bare
        # "invalid choice" cannot. The test there is the authority.
        help="distillation objective; one of: " + ", ".join(sorted(LOSS_ARMS)),
    )
    # Hidden-state matching: an AUXILIARY term added on top of any arm, not an arm itself.
    # gamma 0.0 disables it, which is why there is no separate boolean -- a flag plus a
    # weight lets the two disagree (--use-hidden-loss with gamma 0.0 trains nothing extra
    # and says it is on).
    p.add_argument("--hidden-loss-gamma", type=float, default=0.0)
    p.add_argument(
        "--hidden-loss-layers",
        default="",
        help="comma-separated 0-indexed decoder layers to match; empty = last layer only",
    )
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--vllm-mode", default="server")
    p.add_argument("--vllm-num-servers", type=int, default=1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", default="1e-6")
    p.add_argument("--num-train-epochs", type=float, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--last-message-only", action="store_true")
    p.add_argument("--deepspeed-config", required=True)
    p.add_argument("--extra-config-yaml", default="")
    p.add_argument("--resume", default="auto", choices=RESUME_MODES)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--nodes", type=int, default=2)
    # Tracking. Default "" rather than None so an unset step config key round-trips as "unset"
    # instead of the literal string "None".
    p.add_argument("--clearml-project", default="")
    p.add_argument("--clearml-run-name", default="")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--out", required=True, help="Path to write the rendered GOLD YAML.")
    p.add_argument("--no-check-paths", action="store_true",
                   help="Skip filesystem checks (for rendering outside the runtime).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate(args, check_paths=not args.no_check_paths)
        cfg = render(args)
    except ConfigError as exc:
        print(f"FATAL [distill-gold-train config]: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=True))
    print(f"rendered GOLD config -> {out}")
    for k in sorted(cfg):
        print(f"  {k}: {cfg[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
