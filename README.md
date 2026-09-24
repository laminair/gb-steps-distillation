# gb-steps-distillation

Source for the data-processing, corpus-prep, and config-rendering code behind six
knowledge-distillation steps in [granite.build](https://github.com/ibm-granite/granite.build)
(`ibm-granite/granite.build`): `distill-tokenizer-align`, `distill-corpus-prep`,
`distill-logit-precompute`, `distill-gold-train`, `distill-hf-export`, `distill-eval`.

**This repo is not runnable on its own.** It has no CLI, no orchestrator, and no recipe
runner — it is the code granite.build's steps pull in at run time. To actually run a
distillation job, you need granite.build itself.

## How this fits into granite.build

Each of the six steps above ships a `step-template.yaml` under `steps/<step-name>/` in
granite.build. That template's `code_config` block names a `code_dir` (a checkout of this
repo) plus an `expect_ref` (the exact commit this step is pinned to) and refuses to run if
the checkout isn't at that commit. In other words: granite.build owns *when* and *how* a
step runs, and this repo owns *what code* runs.

- **Steps**, in granite.build: `steps/distill-tokenizer-align/`, `steps/distill-corpus-prep/`,
  `steps/distill-logit-precompute/`, `steps/distill-gold/`, `steps/distill-hf-export/`,
  `steps/distill-eval/`. Each one's `step-template.yaml` is what wires it to this repo.
- **Recipes** — ready-to-run build definitions that chain these steps into a full
  distillation pipeline — live under `recipes/` in granite.build (for example
  `recipes/granite4-gold-distillation/`). Start there if you want to run an existing
  pipeline rather than build one from scratch.
- granite.build's own docs (`docs/steps/`, `docs/builds/build-yaml-reference.md`) explain
  `build.yaml` and the step/target model this repo's code plugs into.

If you've cloned this repo directly and are wondering how to run any of it: you don't, from
here. Go to granite.build, find the matching step or recipe, and follow its own
instructions — this repo will already be wired in as that step's `code_dir`.

## What's here

Each step's entrypoint lives under `steps/<step-name>/src/`. All of them import from the
shared package at `src/gb_steps_post_training/distillation/`.

```
steps/<step-name>/
  src/            # the step's entrypoint(s)
  pyproject.toml  # dependency declarations for `uv sync`, with the pin rationale inline
  uv.lock         # where present, a locked resolution against the pins above

src/gb_steps_post_training/distillation/
  ...             # shared modules every entrypoint above imports
```

## What this is not

This repo does **not** contain the GOLD trainer itself (`gold.py`, `custom_gold_trainer.py`, and
related training-loop code) — that lives in a separate, not-yet-public checkout.
`distill-gold-train`'s entrypoint here (`render_gold_config.py`, `run-gold.sh`) covers the
step's config-rendering and launch-decision logic, which is unit-tested and delivered by the
same `code_dir` mechanism, but the trainer it launches is external.

## Provenance

Some code comments reference companion checks, launchers, and scripts that aren't included in
this repo — they document engineering rationale and validation history from the environment
this code was developed and run in, not runtime dependencies. Every file here runs standalone.
