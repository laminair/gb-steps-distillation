# gb-steps-distillation

Vendored source for the data-processing and config-rendering code that
[granite.build](https://github.com/ibm-granite/granite.build)'s `distill-*` steps deliver at
run time.

## What this is

Six of granite.build's steps — `distill-tokenizer-align`, `distill-corpus-prep`,
`distill-logit-precompute`, `distill-gold-train`, `distill-hf-export`, `distill-eval` — don't
carry their own trainer/data-processing code in the granite.build repo or container image.
Instead, each step's `step-template.yaml` pins a `code_dir` + `expect_ref` (an exact commit) and
refuses to run unless that checkout matches. This repo is that checkout's source of truth.

Each step's entrypoint lives under `steps/<step-name>/src/`. All of them import from the shared
package at `src/gb_steps_post_training/distillation/`.

## What this is not

This repo does **not** contain the GOLD trainer itself (`gold.py`, `custom_gold_trainer.py`, and
related training-loop code) — that lives in a separate, not-yet-public checkout.
`distill-gold-train`'s entrypoint here (`render_gold_config.py`, `run-gold.sh`) covers the
step's config-rendering and launch-decision logic, which is unit-tested and delivered by the
same `code_dir` mechanism, but the trainer it launches is external.

## Layout

```
steps/<step-name>/
  src/            # the step's entrypoint(s)
  pyproject.toml  # dependency declarations for `uv sync`, with the pin rationale inline
  uv.lock         # where present, a locked resolution against the pins above

src/gb_steps_post_training/distillation/
  ...             # shared modules every entrypoint above imports
```

## Provenance

Some code comments reference companion checks, launchers, and scripts that aren't included in
this repo — they document engineering rationale and validation history from the environment
this code was developed and run in, not runtime dependencies. Every file here runs standalone.
