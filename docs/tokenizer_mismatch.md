# Tokenizer Mismatch Between Precompute and Training

> **Ported from an earlier exploratory checkout on 2026-08-25, with one correction.**
> Eight code sites in `src/gb_steps_post_training/distillation/` cite this file by
> path (`utils.py`, `gold.py`, `run_vllm_serve.py`); before the port every one of
> those error messages dead-ended, because the doc existed only inside that earlier
> checkout. Paths in the text below have been rewritten from `gold/<x>.py` to
> `src/gb_steps_post_training/distillation/<x>.py`.
>
> **The correction concerns the trigger, not the mechanism.** The *Root cause*
> section below attributes the pre_tokenizer override to the presence of the legacy
> `vocab.json` + `merges.txt` sidecars. Measured under this tree's transformers
> 5.8.0, that is a correlation, not the cause:
>
> ```
>   files present                            class              pre_tokenizer used    ids
>   tokenizer.json only                      TokenizersBackend  from tokenizer.json   match
>   + vocab.json + merges.txt                TokenizersBackend  from tokenizer.json   match
>   + tokenizer_config.json                  GPT2Tokenizer      GPT-2 ByteLevel       DIVERGE
>   + tokenizer_config.json minus that key   TokenizersBackend  from tokenizer.json   match
> ```
>
> The sidecars are **inert** at this version; the load-bearing input is
> `tokenizer_config.json`'s `tokenizer_class: "GPT2Tokenizer"`. The *mechanism* this
> doc describes is exactly right -- `GPT2Tokenizer.__init__` does rebuild the backend
> and overwrite the pre_tokenizer -- and this doc's own later section, *Correct way to
> load this teacher's tokenizer* (2026-07-03), already names `tokenizer_class` as the
> operative key. The stale attribution survived only in the earlier section, which is
> the one every code comment went on to quote. Both sections are kept below, with the
> earlier one annotated rather than rewritten, so the empirical record stays intact.
>
> Two consequences worth carrying forward:
> - **Why this is silent on the teacher and lethal on the student.** granite-4.2's
>   `tokenizer.json` pre_tokenizer *is* a plain `ByteLevel`, so the override is a
>   no-op there. granite-4.1-3b-base's is `Sequence[Split(GPT-4-style regex),
>   ByteLevel]`, so the override discards the `Split`. Divergence is confined to
>   punctuation/markup-dense text (4 of 6 probes) -- ordinary prose still matches --
>   which concentrates the damage exactly at ChatML turn boundaries.
> - **`is_fast` is not the predicate, and neither is the class name.** Every variant
>   above reports `is_fast=True`, and the class is named `GPT2Tokenizer` even in the
>   teacher case where segmentation is correct. The load-bearing check is comparing
>   the *resolved* pre_tokenizer (`tok.backend_tokenizer.to_str()`) against the one in
>   `tokenizer.json` -- which is what `utils.py:verify_fast_tokenizer` check 3 and
>   `build_overlay.verify()` both do.


Failure mode for the precomputed-logits KD path (`docs/precompute_logits.md`)
when the training-time tokenizer and the precompute-time tokenizer encode
the same chat-templated string into different token streams.

## Symptom

`gold/sft.py:681` raises during dataset preparation (a path in the earlier checkout; `sft.py` is the off-policy KD entry point and is deliberately NOT ported into this repo -- on-policy GOLD only):

```
RuntimeError: KD: assistant token count from chat template (1895) does not
match num_assistant_tokens (1893) from index.jsonl for source_idx=116040.
This usually means the chat template / tokenizer changed between precompute
and training.
```

This fires before the first forward pass but only after the dataset
`map(tokenize_fn, ...)` reaches a divergent example, which can be far
into the dataset.

## Why it triggers

`src/gb_steps_post_training/distillation/sft.py` calls
`apply_chat_template(..., return_assistant_tokens_mask=True)` to recover
the per-token assistant mask. The granite chat template does not contain a
`{% generation %}` block, so HF falls back to scanning the rendered
`input_ids` for the `response_template_ids` prefix and marking the run up
to EOS as the assistant span.

The scan operates on token ids, not on the chat-rendered text. If the
precompute-time tokenizer and the training-time tokenizer produce
different ids for the same string, the scan finds an assistant span of a
different length than the one that was used to write `num_assistant_tokens`
into `index.jsonl`. The KD slice into the precomputed top-K memmap then
no longer aligns with the student's logit positions, so we fail fast on
the count mismatch.

## Root cause

> **Superseded on the trigger (2026-08-25).** Read "which sidecar files are
> present" as "whether `tokenizer_config.json` declares
> `tokenizer_class: GPT2Tokenizer`". Under transformers 5.8.0 the sidecars are
> inert; see the correction at the top of this file. Everything else in this
> section -- the `GPT2Tokenizer.__init__` mechanism, the live pre_tokenizer dumps,
> the empirical divergence table -- reproduced exactly as written.

Two granite tokenizer directories that ship the same `tokenizer.json`,
the same `chat_template.jinja`, and the same BPE merges can still
tokenize differently, depending on **which sidecar files** are present.

`transformers.GPT2Tokenizer.__init__`
(`models/gpt2/tokenization_gpt2.py:94-129`) unconditionally rebuilds the
backend BPE and overwrites the pre_tokenizer with the GPT-2 default
whenever the legacy `vocab.json` + `merges.txt` sidecars are loaded:

```python
self._tokenizer = Tokenizer(BPE(vocab=..., merges=..., ...))
self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=add_prefix_space
)
```

A directory that ships **only** `tokenizer.json` (no `vocab.json`,
no `merges.txt`) bypasses that override and keeps the custom granite
pre_tokenizer from `tokenizer.json`.

In the failing run:
- precompute dir (teacher, `granite-4.1-30b/r260401a`) had the legacy
  sidecars → live pre_tokenizer is the GPT-2 default;
- training dir (student, `granite-4.5-3b-pipecleaner-r260528a-ct`) had
  only `tokenizer.json` → live pre_tokenizer is granite's custom
  `Sequence([Split(...), ByteLevel(...)])`.

Live pre_tokenizer dumps:

```
teacher:
  ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)

student:
  Sequence([
    Split(pattern=Regex(r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
          behavior="Isolated"),
    ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
  ])
```

Empirical divergence on the failing example (`source_idx=116040`):

| input             | teacher ids                  | student ids                  |
|-------------------|------------------------------|------------------------------|
| `'a\n\nb'`        | `[64, 198, 198, 65]`         | `[64, 271, 65]`              |
| `'public.roads2'` | `[898, 13, 43791, 17]`       | `[898, 31942, 7819, 17]`     |
| full example      | 2081 ids                     | 1934 ids                     |
| assistant span    | 1893                         | 1895                         |

The student's granite pre_tokenizer splits digits on `\p{N}{1,3}` and
merges consecutive newlines into the `ĊĊ` token (id 271); the teacher's
GPT-2 default does neither.

### Secondary divergence (observed in production, 2026-06-26)

The "currently latent" trap below the BPE/pre_tokenizer override has
now been hit on a real run:
`spyre-3b-swa-test_from-4.1-30b_en-sft-4.1-0.2-16K_logits_node4`. Same
symptom string (`1895 vs 1893`), different root cause — the legacy
sidecars on the student dir were already aligned with the teacher's,
but the `tokenizer.json` `added_tokens` block disagrees on the named
special-token id table:

| id      | teacher (precompute, `granite-4.1-30b/r260401a`) | student (`granite-4.5-3b-pipecleaner-r260528a-ct`) |
|---------|--------------------------------------------------|----------------------------------------------------|
| 100266  | `<\|unused_1\|>`                                 | `<\|tool_call\|>`                                  |
| 100270  | `<tool_call>`                                    | `<\|unused_1\|>`                                   |
| 100271  | `</tool_call>`                                   | `<\|unused_2\|>`                                   |
| 100272  | `<tool_response>`                                | `<\|unused_3\|>`                                   |
| 100273  | `</tool_response>`                               | `<\|unused_4\|>`                                   |
| 100274  | `<think>`                                        | `<\|unused_5\|>`                                   |
| 100275  | `</think>`                                       | `<\|unused_6\|>`                                   |
| 100276  | `<think_on>`                                     | `<\|unused_7\|>`                                   |
| 100277  | `<think_off>`                                    | `<\|unused_8\|>`                                   |
| 100278  | `<schema>`                                       | `<\|unused_9\|>`                                   |
| 100279  | `</schema>`                                      | `<\|unused_10\|>`                                  |
| 100280  | `<tools>`                                        | `<\|unused_11\|>`                                  |
| 100281  | `</tools>`                                       | `<\|unused_12\|>`                                  |
| 100282  | `<documents>`                                    | `<\|unused_13\|>`                                  |
| 100283  | `</documents>`                                   | `<\|unused_14\|>`                                  |

`chat_template.jinja` is identical on both sides and emits literal
strings like `<tool_call>...</tool_call>`. The teacher tokenizer
encodes each literal as **one** special-token id; the student
tokenizer, whose `added_tokens` block holds `<|unused_N|>` at those
ids, BPE-splits the literal into multiple regular pieces. A single
`<think>...</think>` pair yields ~2 extra tokens on the student side
— exactly the 1893→1895 delta in the traceback.

**Why this fails far into the run, not at startup.** The 15 affected
ids are only exercised when the rendered chat template contains one of
the affected literals. A pure plain-chat sample never produces them,
so its assistant-token count agrees between the two tokenizers and the
probe passes. With a dataset that is mostly plain English SFT and a
few-percent tail of tool-call / RAG / thinking samples, after shuffle
the first divergent example can land arbitrarily late — in the failing
run it was past the 70% mark.

**Why the existing fail-fast probe missed it.** The four probes in
`gold/sft.py:842-867` (a path in the earlier checkout, not ported) (`"a\n\nb"`, `"public.roads2"`, `"1200"`,
`"schema\n\nfollow"`) only exercise BPE / pre_tokenizer differences.
They produce no `<tool_call>` / `<think>` / `<schema>` etc. literals,
so they tokenize identically on both sides and the probe passes even
though tool/think/schema-bearing samples will later trip the slow
post-`map` check. Probe set should be extended to include the named
special-token literals — see the updated *Recommended fix (code)*
below.

**Why `LegacyTokenizerSidecarCallback` does not catch this.** The
callback added in `src/gb_steps_post_training/distillation/utils.py` only preserves the
`vocab.json` / `merges.txt` / `special_tokens_map.json` shape of the
source dir on save, to keep the BPE/pre_tokenizer override decision
stable across reloads. It does not touch `tokenizer.json` or
`tokenizer_config.json`. When the source dir's `tokenizer.json`
already has the wrong special-token id mapping, every save inherits
that mapping unchanged, and reload reproduces the same divergence.
The callback is orthogonal to this failure mode.

**A known-good sibling student dir** for the same precompute teacher
(`granite-4.1-30b/r260401a`) is
`<a known-good sibling student directory>/`. Its
`tokenizer.json` `added_tokens` ids 100266 and 100270–100283 carry the
same content strings as the teacher; the only residual diff is a
cosmetic `"special": true → false` flag on ~12 entries (in both
`tokenizer.json` and `tokenizer_config.json`'s
`added_tokens_decoder`), which governs `skip_special_tokens=True`
decoding but does not affect encoding. Empirical check on every
affected literal:

```
'<tool_call>{"x":1}</tool_call>'  →  [100270, 5018, 87, 794, 16, 92, 100271]
'<think>plan</think>'             →  [100274, 10609, 100275]
'<schema>{}</schema>'             →  [100278, 6390, 100279]
'<tools>x</tools>'                →  [100280, 87, 100281]
'<documents>y</documents>'        →  [100282, 88, 100283]
```

Identical on both sides.

## How to detect

```python
from transformers import AutoTokenizer

t_pre = AutoTokenizer.from_pretrained(meta["tokenizer_name_or_path"],
                                      trust_remote_code=True)
t_run = AutoTokenizer.from_pretrained(model_name_or_path,
                                      trust_remote_code=True)

print(repr(t_pre.backend_tokenizer.pre_tokenizer))
print(repr(t_run.backend_tokenizer.pre_tokenizer))

# Mode-1 probes (pre_tokenizer / BPE override)
for probe in ["a\n\nb", "public.roads2", "1200", "schema\n\nfollow"]:
    a = t_pre.encode(probe, add_special_tokens=False)
    b = t_run.encode(probe, add_special_tokens=False)
    assert a == b, (probe, a, b)

# Mode-2 probes (named special-token id table; see "Secondary
# divergence" above). Each literal must encode to the *same single*
# special-token id on both sides. If either side returns more than 1
# id, the chat template will mis-tokenize on every sample containing
# that literal.
for probe in ["<tool_call>", "</tool_call>",
              "<tool_response>", "</tool_response>",
              "<think>", "</think>",
              "<think_on>", "<think_off>",
              "<schema>", "</schema>",
              "<tools>", "</tools>",
              "<documents>", "</documents>"]:
    a = t_pre.encode(probe, add_special_tokens=False)
    b = t_run.encode(probe, add_special_tokens=False)
    assert a == b and len(a) == 1, (probe, a, b)
```

Direct id-table diff (catches mode 2 without having to enumerate
literals):

```python
import json
def added_token_map(path):
    with open(f"{path}/tokenizer.json") as f:
        return {t["id"]: t["content"] for t in json.load(f).get("added_tokens", [])}

ma = added_token_map(meta["tokenizer_name_or_path"])
mb = added_token_map(model_name_or_path)
for tid in sorted(set(ma) | set(mb)):
    if ma.get(tid) != mb.get(tid):
        print(f"  id={tid}  pre={ma.get(tid)!r}  run={mb.get(tid)!r}")
```

Also useful:

```bash
for f in tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
  md5sum "$PRECOMP_DIR/$f" "$TRAIN_DIR/$f"
done
```

A matching md5 on `tokenizer.json` is **not** sufficient — the live
pre_tokenizer depends on which other files are in the directory. And a
matching md5 on `vocab.json` / `merges.txt` is also not sufficient —
those bytes are identical even when `tokenizer.json`'s `added_tokens`
block disagrees (mode 2). Hash `tokenizer.json` separately.

**Rate-of-failure shortcut.** To predict how many samples will trip
mode 2 before launching, count occurrences in the source jsonl:

```bash
grep -cE '<tool_call>|</tool_call>|<tool_response>|</tool_response>|<think>|</think>|<think_on>|<think_off>|<schema>|</schema>|<tools>|</tools>|<documents>|</documents>|<\|tool_call\|>' \
  "$SOURCE_JSONL"
```

Divide by total rows for the upper bound on the fraction of samples
that will fail the alignment probe (lower in practice because some
matches occur in tool-response text that the chat template wraps
rather than the literal-emission positions).

## Workarounds

### A. Strip legacy sidecars from the precompute dir; re-precompute

> **2026-08-25:** stripping the sidecars alone is a no-op under transformers
> 5.8.0. The equivalent effective action is to remove `tokenizer_class` from
> `tokenizer_config.json` (sufficient for a tokenizer-only dir) or to set it to
> `PreTrainedTokenizerFast` (see the fixture note under *Canonical
> fast-tokenizer fixture*, which is the form that also survives a `config.json`
> being present).

Copy the teacher into a clean dir without `vocab.json` and `merges.txt`,
re-run precompute against that dir. Both ends then load via
`tokenizer.json` and the granite pre_tokenizer survives on both sides.

Cost: full re-precompute.

### B. Mirror the legacy sidecars into the training dir

> **2026-08-25: this workaround no longer does anything** under transformers
> 5.8.0 -- copying the sidecars in does not change the resolved pre_tokenizer,
> so it cannot make the two sides agree. It was already marked not-recommended
> by the 2026-07-03 warning below, on the stronger grounds that it forces both
> sides onto a tokenization the teacher was never trained with. Kept for the
> record only; do not reach for it.

Copy `vocab.json` + `merges.txt` from the precompute dir into the
student dir. Both ends now hit the GPT-2 override and produce the same
(GPT-2-default) tokenization, so the existing precompute is usable.

Cost: zero re-compute. Downside: every downstream artifact saved from
the student dir inherits the GPT-2 pre_tokenizer, which is **not** what
granite's runtime inference uses. Not recommended for any checkpoint
that will be served.

> **Warning (2026-07-03):** empirical evidence (see the *Which
> tokenizer was the teacher trained with?* section below) shows that
> `granite-4.1-30b/r260401a` was actually trained with the granite
> fast pre_tokenizer, **not** the GPT-2 default. Workaround B forces
> both sides to use the GPT-2 default, which is NOT what the teacher
> was trained with — the resulting precompute logits are OOD for the
> teacher itself. Prefer Workaround A, C, E, or F.

### C. Override the training tokenizer path in `src/gb_steps_post_training/distillation/sft.py`

When `precomputed_logits_dir` is set, load the training tokenizer from
`meta["tokenizer_name_or_path"]` instead of `model_args.model_name_or_path`.
The student model still loads from its own dir; only the tokenizer is
pinned to whatever produced the precompute.

Cost: ties training-time tokenization to the precompute dir. Safer than
B because the student checkpoint dir is not mutated.

### D. Ship `input_ids` + `assistant_masks` in the precompute output

Store the precomputed `input_ids` and assistant mask alongside the
top-K logits and skip re-tokenization at training time entirely. This
removes the failure mode structurally.

Cost: precompute format change, larger on-disk footprint, schema bump.
Tracked as future work in `docs/precompute_logits.md`.

### E. Fix the upstream tokenizer artifact (structural)

Ship the same `tokenizer.json` bytes from teacher and student model
builds, with no `vocab.json` / `merges.txt` sidecars on either side.
This is the only fix that makes the user's expected invariant —
"same tokenizer family ⇒ same tokenization" — actually true.

### F. Replace student `tokenizer.json` + `tokenizer_config.json` with the teacher's (mode-2 only)

Applies when the legacy-sidecar override (mode 1) is already aligned
but the named special-token id table differs (mode 2, as in the
2026-06-26 occurrence). The chat template emits literals like
`<tool_call>` expecting them to encode to single special-token ids;
forcing the student dir to use the teacher's `tokenizer.json`
mapping makes that hold:

```bash
# WARNING: the source dir may have these as symlinks pointing back at
# the upstream model dir. Use `rm` on the symlinks before `cp` if you
# want the change scoped to the *-ct/ overlay only.
cp "$PRECOMP_TEACHER_DIR/tokenizer.json"        "$STUDENT_DIR/tokenizer.json"
cp "$PRECOMP_TEACHER_DIR/tokenizer_config.json" "$STUDENT_DIR/tokenizer_config.json"
```

Cost: zero re-compute. The student's model weights at the swapped ids
(originally `<|unused_N|>` rows) become the rows the teacher was
training as `<tool_call>` etc. This is fine for training (the rows
get gradient signal under their new names), but the resulting
checkpoint is only servable against a tokenizer that uses the
teacher's mapping. Document the swap in the run's notes.

Alternative: pick a student dir whose `tokenizer.json` already matches
the teacher's. For the `granite-4.1-30b/r260401a` precompute set,
`<a known-good sibling student directory>/` is a verified
match (see *Secondary divergence* above).

## Recommended fix (code)

Even after a per-run workaround, future runs will hit the same trap if
upstream model dirs keep shipping inconsistent sidecars. `src/gb_steps_post_training/distillation/sft.py`
now runs a fail-fast probe at trainer init when `precomputed_logits_dir`
is set:

1. Load `precompute_tok = AutoTokenizer.from_pretrained(
   meta["tokenizer_name_or_path"], trust_remote_code=True)`.
2. Assert
   `repr(precompute_tok.backend_tokenizer.pre_tokenizer)
    == repr(tokenizer.backend_tokenizer.pre_tokenizer)`.
3. Assert
   `precompute_tok.encode(p, add_special_tokens=False)
    == tokenizer.encode(p, add_special_tokens=False)`
   for `p` in `["a\n\nb", "public.roads2", "1200", "schema\n\nfollow"]`.
4. On failure, raise with both directory paths and a pointer to this
   file.

This catches the drift before the dataset `map` and before any training
step.

## Detection coverage

The `verify_tokenizer_consistency` helper in `src/gb_steps_post_training/distillation/utils.py` is now
called at every train↔reference tokenizer pairing site (off-policy KD
in `src/gb_steps_post_training/distillation/sft.py`; on-policy GOLD in `src/gb_steps_post_training/distillation/gold.py`). It runs these
checks in order, first failure raises:

1. **Backend `pre_tokenizer` repr.** Original Mode-1 check.
2. **Backend `post_processor` repr.** Catches e.g. `GPT2Tokenizer`
   slow-path injecting a `TemplateProcessing` post_processor even
   when `tokenizer.json` says `post_processor: null`.
3. **Backend `normalizer` repr.**
4. **Backend `decoder` repr.**
5. **`added_tokens_decoder` mapping equality.** Entry-by-entry
   comparison of `content` strings. Primary Mode-2 fast-fail:
   catches "id 100270 = `<tool_call>` on teacher, `<|unused_1|>` on
   student" without waiting for the chat template to emit the
   literal.
6. **Extended encoding probe battery**, in three groups:
   - Mode-1 probes: `"a\n\nb"`, `"public.roads2"`, `"1200"`,
     `"schema\n\nfollow"`.
   - Mode-2 special-token literals: `<tool_call>`, `</tool_call>`,
     `<tool_response>`, `</tool_response>`, `<think>`, `</think>`,
     `<think_on>`, `<think_off>`, `<|start_of_role|>`,
     `<|end_of_role|>`, `<schema>`, `</schema>`, `<tools>`,
     `</tools>`, `<documents>`, `</documents>`, `<|end_of_text|>`,
     `<|pad|>`, `<|unk|>`.
   - One compound chat-template-shaped probe:
     `"<|start_of_role|>system<|end_of_role|>you are helpful<|end_of_text|>"`.

Mode 2 is now **fail-fast at trainer bootstrap**, not slow-fail deep
into `dataset.map(tokenize_fn, ...)`. The 2026-06-26 occurrence would
have raised at check 5 (id-table content mismatch on ids 100270–100283)
before the KD dataset was materialized.

Class name equality is intentionally not checked. In this tree's
transformers 5.8 venv, a teacher dir shipping legacy `vocab.json` +
`merges.txt` legitimately loads as `GPT2Tokenizer` while a student dir
shipping only `tokenizer.json` loads as `TokenizersBackend`. If the two
sides actually agree on tokenization, the class-name asymmetry is
harmless; if they don't, the backend-repr and encoding-probe checks
above will fire with actionable diagnostics.

The on-policy GOLD trainer in `src/gb_steps_post_training/distillation/gold.py` now runs the same helper
at bootstrap between the student tokenizer and the live teacher
tokenizer (`training_args.teacher_model_name_or_path`). Previously
only `vocab_size` was compared with a print-warning, so a silent
tokenizer override on either side would have gone undetected.

The per-example `num_assistant_tokens` alignment check in `src/gb_steps_post_training/distillation/sft.py`
(lines 834-840 in the trainer's own checkout) is preserved as a second line of defense — it
catches residuals that specific probes might miss and catches genuine
chat-template regressions.

## Out-of-scope but worth tracking

- The `LegacyTokenizerSidecarCallback` (`src/gb_steps_post_training/distillation/utils.py`, wired into
  `src/gb_steps_post_training/distillation/gold.py` and `src/gb_steps_post_training/distillation/sft.py`) preserves the legacy-sidecar shape
  on save but does *not* sync `tokenizer.json` / `tokenizer_config.json`
  from the source dir. If a future failure mode is found where saved
  checkpoints drift in those files specifically, extend the helper to
  copy them too. As of 2026-06-26 there is no observed drift on those
  two files between source dir and saved checkpoint — HF Trainer's
  `_save` writes them faithfully.
- The cosmetic `"special": true / false` flag on `<|fim_prefix|>` and
  friends in `tokenizer_config.json`'s `added_tokens_decoder` (and in
  `tokenizer.json`'s `added_tokens`) does not affect encoding but does
  affect `skip_special_tokens=True` decoding. If KD eval or completion
  logging stops stripping `<tool_call>` / `<think>` etc. after a tokenizer
  swap, this is why.

## Which tokenizer was the teacher trained with? (empirical, 2026-07-03)

The earlier sections describe *how* two granite dirs can end up with
different live pre_tokenizers. This section pins down which of the two
matches the actual training tokenization for the canonical precompute
teacher `granite-4.1-30b/r260401a` — the direction of divergence
matters for choosing a workaround.

**Result:** the teacher was trained with the granite fast
pre_tokenizer `Sequence([Split(...), ByteLevel(use_regex=False)])`,
**not** the GPT-2 default that `AutoTokenizer.from_pretrained(<teacher dir>)`
currently returns from its own directory. The GPT-2 default is what
the legacy sidecar override forces at load time; it is a live-load
artifact, not the training-time behavior.

### Method

Compare total NLL = -log P(text) that the teacher assigns to the same
chat-template-rendered text under two tokenizations of that text:

- `slow` = `AutoTokenizer.from_pretrained(teacher_dir, use_fast=False)`
  → live `GPT2Tokenizer` (GPT-2 regex) because of the legacy sidecar
  override.
- `fast` = `PreTrainedTokenizerFast(tokenizer_file=teacher_dir/tokenizer.json)`
  → the granite `Sequence[Split, ByteLevel(use_regex=False)]`
  pre_tokenizer that `tokenizer.json` actually declares.

The tokenizer that yields lower total NLL (higher likelihood under the
model) is the one the model was trained on. Comparing total NLL rather
than per-token PPL is important because the two tokenizations produce
different token counts for the same string.

Script: `tmp/compare_tokenizer_ppl.py`. Dataset: first 256 examples of
`data/en_sft_4.1/subsampled_0.01_shuffled.jsonl`, rendered through the
model's chat template (`tokenize=False`) and then re-encoded by each
tokenizer.

### Numbers (256 examples, bf16 across 2 GPUs)

| Tokenizer                                   | total NLL | avg NLL/ex | total tokens | PPL / token |
|---------------------------------------------|----------:|-----------:|-------------:|------------:|
| slow (`GPT2Tokenizer`)                      | 1,211,582 |    4,732.7 |      371,281 |    **26.1** |
| fast (`Split + ByteLevel(use_regex=False)`) |   376,603 |    1,471.1 |      315,887 |    **3.29** |
| Δ (fast − slow)                             |  −834,979 |   −3,261.6 |      −55,394 |           — |

The fast tokenization is ~3.2× more likely per example under the
teacher. PPL/token of 3.29 is in-range for a well-trained 30B model on
its training distribution; PPL/token of 26 is an order of magnitude
too high and can only be explained by a tokenizer mismatch. The fast
tokenizer also produces ~15% fewer tokens for the same text, matching
what the granite pre_tokenizer regex (`\p{N}{1,3}`, `.ĊĊ`-style merges)
predicts.

Raw per-example results: `tmp/tokenizer_ppl_results.json`. Job log:
`tmp/logs/tokenizer_ppl.<jobid>.out`.

### Consequences for the workarounds above

- **Workaround A (strip legacy sidecars from precompute dir; re-precompute)**
  is the correct fix: after stripping `vocab.json` + `merges.txt`, the
  teacher dir loads via `tokenizer.json` and the live pre_tokenizer
  matches training. All downstream artifacts remain consistent.
- **Workaround B (mirror legacy sidecars into training dir)** aligns
  the two sides *at the wrong pre_tokenizer*: both ends then produce
  GPT-2-default tokenizations, which the teacher never saw during
  training. Precompute logits and any student trained against them
  will therefore be conditioned on OOD tokenizations. Do not use B
  for the canonical precompute teacher.
- **Workaround C (override training tokenizer from `meta["tokenizer_name_or_path"]`)**
  is only safe when the precompute dir itself is loaded correctly.
  If the precompute run loaded the teacher via `AutoTokenizer` and
  therefore recorded a GPT-2-default tokenization, C propagates the
  same wrong tokenization to training.
- **Workarounds E and F** are unaffected — they either replace the
  upstream artifacts structurally (E) or copy the teacher's already-
  correct `tokenizer.json` bytes to the student side (F).

### Fast-tokenizer assertion (2026-07-03)

`src/gb_steps_post_training/distillation/utils.py` now exposes `verify_fast_tokenizer(tok, model_dir, *,
source_label)`. It is called at every student/teacher tokenizer load
site: `src/gb_steps_post_training/distillation/gold.py` (student + teacher), `src/gb_steps_post_training/distillation/sft.py` (student +
precompute teacher from `meta["tokenizer_name_or_path"]`), and
`src/gb_steps_post_training/distillation/run_vllm_serve.py` (pre-flight on `--model <path>` before
vLLM binds a port). `src/gb_steps_post_training/distillation/gold.py` also runs a trainer→vLLM
tokenization probe after trainer init when `vllm_mode == "server"`,
comparing local `tokenizer.encode(probe)` against
`vllm_client.generate(prompts=[probe], max_tokens=1)["prompt_ids"][0]`
on the Mode-1 probes; a mismatch aborts before the first training
step.

Failure surfaces as `RuntimeError` with a diff of the live
`pre_tokenizer` / `post_processor` / `normalizer` / `decoder` reprs
against the ones declared in `tokenizer.json`, plus a pointer back
to the *Which tokenizer was the teacher trained with?* section
below. The check requires only `tokenizer.json` in the model
directory and does not depend on the presence of legacy
`vocab.json` / `merges.txt` sidecars.

**Migration note for existing checkpoints.** Prior to this change,
`LegacyTokenizerSidecarCallback` in `src/gb_steps_post_training/distillation/utils.py` actively copied
`vocab.json` + `merges.txt` from the source dir into every
`checkpoint-N/` directory written by HF Trainer. That callback and
the associated `copy_legacy_tokenizer_sidecars` helper have been
removed, and the trailing post-`save_model` sidecar copies at the
end of `src/gb_steps_post_training/distillation/gold.py` and `src/gb_steps_post_training/distillation/sft.py` have been dropped. Old
checkpoints saved by the previous code will still reload with the
legacy sidecars present and will now trip
`verify_fast_tokenizer` at resume time. The correct action is to
strip `vocab.json` + `merges.txt` from those checkpoint dirs (or
adjust `tokenizer_config.json` so `AutoTokenizer` returns the fast
class), not to reintroduce the callback — sidecars are unnecessary
for a correctly-loaded fast tokenizer because `tokenizer.json`
already contains the BPE vocab and merges inline.

### Correct way to load this teacher's tokenizer

Because `tokenizer_config.json` for this dir sets
`"tokenizer_class": "GPT2Tokenizer"` and has no `fast_tokenizer_class`,
`AutoTokenizer.from_pretrained(teacher_dir)` returns the slow
`GPT2Tokenizer`, which silently rebuilds the backend with the GPT-2
default pre_tokenizer (see *Root cause* above). Two correct ways to
load the same tokenizer that the model was trained on:

```python
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast(tokenizer_file=f"{teacher_dir}/tokenizer.json")
# copy chat_template / special tokens across from the AutoTokenizer
# view if you need chat rendering (see tmp/compare_tokenizer_ppl.py).
```

or, cleanest, use a teacher dir that ships **only** `tokenizer.json`
(no `vocab.json` / `merges.txt`), so the sidecar override never fires.
The verified-good student sibling of this teacher —
`<a known-good sibling student directory>/` — is an example of
the latter shape.

### Canonical fast-tokenizer fixture + in-place replacer

For student checkpoints that must be loaded via `AutoTokenizer.from_pretrained`
(i.e. every training / vLLM entry point in this repo), a canonical fixture
of the granite-4.1-30b/r260401a fast tokenizer lives at:

```
<a canonical tokenizer mirror directory>/
```

It contains `tokenizer.json`, `special_tokens_map.json`,
`chat_template.jinja` verbatim from the teacher, plus a
`tokenizer_config.json` copied from the teacher and amended so
`tokenizer_class = "PreTrainedTokenizerFast"`. Note this is **not**
just stripping `tokenizer_class`: any dir with `config.json` setting
`model_type: granite` triggers `TOKENIZER_MAPPING_NAMES["granite"] →
"GPT2Tokenizer"` and reintroduces the slow path even when
`tokenizer_class` is absent from `tokenizer_config.json`. Setting
`tokenizer_class = "PreTrainedTokenizerFast"` explicitly bypasses that
model_type lookup and forces the fast loader to honor
`tokenizer.json`'s declared pre_tokenizer.

To repair a broken student ckpt in place (removes legacy
`vocab.json` + `merges.txt` sidecars, then copies the canonical files
on top; unlinks any pre-existing symlink at the destination path so
upstream shared trees are never mutated):

```bash
bash checkpoints/replace_canonical_tokenizer.sh <student_ckpt_dir>
```

The script's exit code propagates from a post-hook
`checkpoints/check_fast_tokenizer.sh` run against the mutated dir.

### How to redo this check for future teachers

The comparison is model-agnostic. Point the script at any (teacher, jsonl)
pair:

```bash
.venv/bin/python tmp/compare_tokenizer_ppl.py \
    --model <teacher_dir> \
    --data  <path/to/subsample.jsonl> \
    --n 256 --max-len 4096
```

The `likely_training_tokenizer` field in
`tmp/tokenizer_ppl_results.json` and the per-token PPL columns are the
diagnostic: a PPL/token below ~5 on in-distribution text indicates the
correct tokenizer; PPL/token above ~15 flags a pre_tokenizer /
sidecar mismatch.

> **That script no longer exists** — `tmp/` was scratch and did not survive. The
> measurement was re-authored as a permanent companion tool (not included in this
> repo) that takes `--tokenizer` repeatably (two or more candidates), holds the
> document text byte-identical across them, and **ranks on total NLL rather than
> PPL/token** — the candidates emit different token counts, so a per-token mean
> answers a different question for each of them. It also has a
> `--segmentation-only` mode that reports how much the candidates actually
> disagree, in token boundaries, on CPU in seconds.

## Pinning `PreTrainedTokenizerFast` is a measurement, not a rule (2026-09-22)

Everything above describes a checkpoint whose `tokenizer.json` holds the
**trained** segmentation while its declared `tokenizer_class` overrides it.
Pinning the class then restores the trained one, and that is why
`build_overlay.CONFIG_KEYS_FORCED` forces it. **The override can also run the
other way**, and when it does, pinning is the defect rather than the repair.

`granite-5.0-20b-sft` is that case, and it cost an 8-GPU allocation and zero
trained steps (`karve-b9`, killed on all 8 ranks at `utils.py:283`):

| | declared `tokenizer_class` | `tokenizer.json` `pre_tokenizer` | what the model was **trained** with |
|---|---|---|---|
| granite-4.1 / 4.2 family | a concrete class that overrides | the trained one | `tokenizer.json`'s → **pin the class** |
| granite-5.0-20b-sft | `GPT2Tokenizer` | `Sequence[Split(cl100k-ish), ByteLevel]`, **vestigial** | plain `ByteLevel(use_regex=True)` → **do not pin alone** |

Three things this case establishes that the sections above do not:

1. **The mechanism is class identity, not fast-vs-slow.** Loading the
   published directory untouched: `is_fast` is **True** and it *still*
   substitutes plain `ByteLevel`, because the GPT-2 class rebuilds its backend
   from vocab+merges instead of adopting `tokenizer.json`'s `pre_tokenizer`.
   Any reasoning of the form "it loads slow, therefore it discards
   tokenizer.json" is wrong about this checkpoint.

2. **The `<~5` / `>~15` PPL/token bands do not transfer to a pre-split-only
   difference.** They were calibrated on a slow class rebuilding the **BPE
   merges** — 3.29 against 26.1, an **8×** spread, where every token can move.
   Two candidates that differ only in a pre-split regex agree on most text and
   diverge mainly on digit runs (`\p{N}{1,3}` caps groups at three where the
   GPT-2 regex takes the whole run) and cased contractions. They cannot produce
   an 8× spread, so the **absolute** PPL says nothing and applying the band
   sends you to "inconclusive" on a decided question. It cost us one round.
   The readable statistic is the **margin together with its concentration**: for
   granite-5.0-20b-sft, plain `ByteLevel` won by 17.0% (raw render) and 19.8%
   (chat render) of total NLL over 512 documents while only **3.33%** of token
   boundaries differed at all — so ~272,666 extra nats landed on ~22,400
   tokens, order 12 nats each against a ~2.5 nat corpus average. A margin many
   times the divergent share is what a real segmentation difference looks like,
   measured directly across three separate runs.

3. **A pin can be necessary *and* insufficient.** Dropping `tokenizer_class`
   is not a safe middle course: a direct measurement showed `model_type` reviving the
   override through `TOKENIZER_MAPPING_NAMES` with the key absent entirely. So
   the 5.0 teacher needs **both** — the pin, *and* the trained `pre_tokenizer`
   written into `tokenizer.json`. `build-model-mirror.py --pre-tokenizer-from
   DIR` does the second, adopting the blob byte-for-byte from a directory that
   already has it rather than synthesising one (a hand-written blob is an
   artifact nobody measured; the donor's is the exact object the NLL comparison
   ranked and the exact object `verify_tokenizer_consistency` compares). It
   asserts that donor and source share `model.vocab` **and** `model.merges`,
   because a `pre_tokenizer` decides where pieces begin while the merges decide
   which pieces can exist — transplant across a different merge table and the
   backend degrades quietly into byte fallback rather than failing.

**This is not Workaround B.** B is condemned above for aligning both sides at
the *wrong* `pre_tokenizer`. Here the measurement is what identifies which one
is right, and the sides are aligned at that one. The distinction is the
measurement; without it the two actions are indistinguishable from the outside.

**The rule to carry forward:** never decide `tokenizer_class` from what
`tokenizer.json` contains. `tokenizer.json` is not authoritative about what a
checkpoint was trained with — it is one of two candidates. Ask the model, with
the NLL-comparison tool above, per checkpoint. A companion config check runs
`verify_tokenizer_consistency` on the student/teacher pair whenever a config sets
`use_uld_loss: false`, mirroring `gold.py`'s own guard, so this class of defect
costs CPU seconds instead of a multi-node allocation — but the guard can
only tell you the two sides **disagree**, never which side is **right**. That
part is still a measurement, and there is no check that can replace it.
