# Precomputed Top-K Teacher Logits

> **Ported from an earlier exploratory checkout on 2026-09-21.** Paths rewritten from
> `gold/precompute_logits.py` and `gold/sft.py` to
> `src/gb_steps_post_training/distillation/{precompute_logits.py,sft.py}`.
>
> **This is the one of the four ported docs that describes code we already ship, have
> run, and have scored.** It is therefore a *design* doc reconciled against an
> *implementation*, and the two have drifted. Where they disagree the implementation is
> right and this document says so inline; see *Status in this tree* at the foot for the
> full delta, including three things the design did not anticipate and one flag whose
> name would now be wrong if copied from upstream.
>
> Cited (previously as a dead link) from [`tokenizer_mismatch.md`](tokenizer_mismatch.md)
> at `:47` and `:371`.

## Motivation

In standard KD the teacher forward runs every step, alongside the student's forward and
backward. For a 30B teacher against a 3B student that teacher forward is roughly 67% of
model FLOPs, repeated identically for every epoch and every hyperparameter arm that
shares a corpus.

Precomputing pays it once. Run the teacher over the corpus ahead of training, keep only
the top-K logits per assistant token, and the training loop reads them off disk. The
teacher never loads during training, so the arm that would have needed a 30B model in
memory needs only the student — which is what makes a 1-GPU or single-node KD arm
possible at all, and what makes an N-arm sweep cost one teacher pass instead of N.

The cost is disk and fidelity: top-K truncates the distribution, and the artifact is
valid only for the exact corpus, tokenizer, and rendering it was built under. That
binding is the source of every failure mode in this document.

## Storage layout

```
<output_dir>/
  meta.json                      provenance for the whole artifact
  index.jsonl                    one row per SOURCE example, in source order
  shards/
    indices_000000.bin           int32   top-K token ids
    logits_000000.bin            float16 the matching raw logits
    indices_000001.bin
    logits_000001.bin
    ...
```

**`meta.json`** records what the artifact was built from: teacher weights,
tokenizer, `top_k`, dtypes, vocab size, source JSONL and its md5, `max_length`, and the
world size at precompute time. It is the thing a consumer checks before trusting a byte
of the shards.

**`index.jsonl`** carries one row per source example — including skipped ones — with
`source_idx`, the `messages`/`tools`/`documents` as rendered, `skipped` and
`skip_reason`, `num_assistant_tokens`, and the `shard_id` / `shard_offset` pair locating
the row's logits. Rows stay in source order after the merge, so `index.jsonl` line *i*
is source line *i*.

**Shards** are flat memmaps. Row *r* of shard *s* at offset *o* with `n` assistant
tokens occupies `[o, o+n)` in both files, `n × K` int32 in `indices_*.bin` and `n × K`
float16 in `logits_*.bin`. Nothing in a shard identifies which row it belongs to — the
index is the sole authority, which is why the writer appends to the shards first and
names the row in the index second, and why a killed job leaves shards holding rows the
index does not mention (reconciled on resume, see *Resumption*).

`shard_id = rank * 1000 + local_idx`, so every rank writes its own files and no lock is
needed. `--shard-target-tokens` (default 4,000,000) rolls to a new shard once a shard's
token count passes the target, keeping individual files at a few GB.

### Size

Per assistant token: `K × (4 + 2)` bytes = 1,536 B at K=256. Measured in this tree:
20,000 rows → **13 GB**; 192,000 rows → **122 GB**. Extrapolating the full 802,027-row
corpus gives ~540 GB, which is why every run here has used an explicit subset rather
than the whole corpus (see *Status in this tree*).

## CLI

```bash
python -m gb_steps_post_training.distillation.precompute_logits \
    --input-jsonl  data/.../train.jsonl \
    --output-dir   data/.../precompute/blend20k-top256 \
    --teacher-model <teacher path> \
    --teacher-tokenizer <the tokenizer the STUDENT will train under> \
    --top-k 256 --max-length 16384 --batch-size 4 --dtype bfloat16 \
    --shard-target-tokens 4000000
```

> **Flags are hyphenated here, not underscored.** Upstream's design doc writes
> `--input_jsonl`, `--output_dir`, `--teacher_model`, `--top_k`. Every entrypoint in this
> collection takes hyphens, and the step launcher passes them through unchanged instead
> of translating — one fewer place for a name to drift. Copying an upstream command line
> verbatim will fail at argparse.

| Flag | Default | Notes |
|---|---|---|
| `--input-jsonl` | — | the corpus, one JSON object per line |
| `--output-dir` | — | created; resumable (see below) |
| `--teacher-model` | — | weights |
| `--teacher-tokenizer` | `--teacher-model` | **read the warning below before defaulting it** |
| `--top-k` | 256 | recorded in `meta.json`; the trainer cross-checks it |
| `--max-length` | 8192 | rows over this are **skipped, not truncated** |
| `--batch-size` | 4 | teacher forward batch |
| `--dtype` | `bfloat16` | teacher load dtype; stored logits are always float16 |
| `--shard-target-tokens` | 4,000,000 | shard roll threshold |
| `--ignore-documents` | off | drop the `documents` field before rendering |
| `--response-template` | `<\|im_start\|>assistant\n` | fallback masking only — see *Masking* |
| `--max-skip-fraction` | 0.05 | refuse if more of the corpus was skipped than this |
| `--allow-tokenizer-mismatch` | off | escape hatch for the tokenizer guard |
| `--expectation-file` / `--emit-expectation` | — | pin/record the input fingerprint |
| `--shard-count` / `--shard-index` | — | residue-class passes; see *Parallelism* |
| `--verify-only` | off | certify an existing directory, write nothing |

## Behaviour

**Sharding across ranks.** `PartialState()` gives each rank its slice round-robin by row.
The module builds a 2-D mesh — `init_device_mesh("cuda", (num_nodes, local_world_size),
("dp", "tp"))` — teacher tensor-parallel within a node, data-parallel across nodes.

**Rendering and masking.** `apply_chat_template(..., return_assistant_tokens_mask=True)`
supplies the assistant mask when the template carries `{% generation %}` markers. For
templates without them the step falls back to **scanning for `--response-template`**.
These two mechanisms are not interchangeable — see the warning under *Masking* — and the
chosen one is recorded so the trainer can cross-check it.

**Logit position.** The logits that predict assistant token at position `p` are the
model's outputs at `p-1`. The step stores the shifted position, so the trainer does not
shift again.

**Attention.** FlashAttention-2 for inference; `attn_implementation` goes into
`meta.json`.

**Skips, not truncation.** A row longer than `--max-length`, or with no assistant tokens
at all, is written to the index with `skipped=true` and `skip_reason` of `too_long` or
`no_assistant`, and contributes nothing to the shards. It is *not* truncated.

> Upstream's design assumed the trainer truncates overlong rows and slices the
> precomputed logits to match. It does not, and skipping is the safer of the two: a
> truncated row's stored logits and the trainer's re-render would have to agree on where
> the cut fell, and nothing enforces that. The consequence is that `--max-length` is a
> **corpus filter**, which is exactly why `--max-skip-fraction` exists.

**Resumption.** Each rank appends to `index_part_<rank>.jsonl` as it goes. On restart a
rank reads its own part file, skips rows already present, and **reconciles the shards**:
rows appended to `*.bin` but not yet named in the index are truncated away, because the
index is the authority on shard contents and a killed job leaves the two out of step.
Rank 0 merges the part files into `index.jsonl` in source order at the end.

**Why HF + accelerate rather than vLLM.** vLLM caps `prompt_logprobs`, so it cannot
return a full top-256 per prompt token. A plain HF forward can.

## Consuming the artifact in training

`sft.py` grows a KD surface (`:305-314`):

| Field | Default | Meaning |
|---|---|---|
| `precomputed_logits_dir` | `""` | set it to switch `compute_loss` onto the KD path |
| `kd_top_k` | 256 | sanity-checked against `meta.json` |
| `kd_weight` | 1.0 | weight on the KD term |
| `ce_weight` | 0.0 | weight on the cross-entropy term |
| `kd_temperature` | 1.0 | softmax temperature `T` |

A collator wraps the base one and attaches the per-row `[n, K]` index/logit blocks
(`:397-407`), positionally aligned to the labeled tokens. The loss
(`_compute_loss_kd`, `:694`) masks first so it never materializes `[B*T, V]`:

```python
T = self.kd_temperature
t_log_probs = F.log_softmax(t_lg.float() / T, dim=-1)              # [N, K]
t_probs = t_log_probs.exp()
s_log_probs_topk = torch.gather(
    F.log_softmax(flat_logits / T, dim=-1),
    dim=-1, index=t_idx.long(),
)                                                                   # [N, K]
kl = (t_probs * (t_log_probs - s_log_probs_topk)).sum(-1).mean()
loss = self.kd_weight * (T * T) * kl
if self.ce_weight > 0.0:
    loss = loss + self.ce_weight * F.cross_entropy(flat_logits, flat_labels)
```

Forward KL on the **renormalized top-K** teacher distribution — the `log_softmax` is over
K entries, not V, so the stored top-256 is treated as the whole distribution rather than
as a truncation of it. `T²` scaling keeps the gradient magnitude comparable across
temperatures.

Two guards matter more than they look:

- **Token-count equality** (`:724-730`). If the student's labeled-token count disagrees
  with the teacher block's row count, the step raises rather than training. A one-token
  shift pairs every stored logit with the wrong target, and the error text names the
  three causes (chat template changed, tokenizer changed, mask mechanism changed).
- **Mutual exclusion with `use_liger_kernel`** (`:495-497`). Liger's fused CE owns the
  loss, so the two cannot both be active. There *is* a Liger-fused path inside
  `_compute_loss_kd` (`:702-714`) that captures hidden states and projects only the
  masked rows — that one is compatible; the top-level flag is not.

`compute_loss` dispatches to the KD path first when `precomputed_logits_dir` is set
(`:527-529`).

## Verification

1. Precompute over a handful of rows at a small `--top-k`.
2. Check `meta.json` against the inputs you think you used.
3. Check `index.jsonl` line count equals the source line count.
4. Check shard file sizes against `Σ num_assistant_tokens × K × {4,2}`.
5. Train two steps and confirm a finite non-zero loss and non-zero `grad_norm`.
6. Confirm the token-count guard fires if you deliberately change the tokenizer.

---

## Status in this tree

**Shipped, executed, and scored — which is what distinguishes this doc from the other
three ported in this pass.** Measured 2026-09-21:

| Surface | Where |
|---|---|
| module | `src/gb_steps_post_training/distillation/precompute_logits.py` |
| trainer side | `sft.py:305-314`, `:397-407`, `:477-491`, `:495-497`, `:512-513`, `:527-529`, `:694` |
| step | `steps/distill-logit-precompute/` (Dockerfile, Makefile, step.yaml, step-template.yaml, `src/run-precompute.sh`, `test/test_precompute_logits.py`) |
| submitters | a companion submit script (real), a smoke-test variant (1 GPU, 64 rows) |
| shard merge | a companion index-merge tool |
| OpenShift build | `build/openshift/buildconfig-distill-logit-precompute.yaml` |

Two real artifacts exist on disk — `blend20k-top256` (20,000 rows, 13 GB, 1 node × 8
GPUs in 1,407 s) and `blend192k-top256` (192,000 rows, 122 GB) — both at top-K 256,
`max_length` 16384, teacher `granite-4.2-30b`, **0 rows skipped**. Both were consumed by
training arms (`blend-submit.sh`: `ce` / `kd` / `blend`, plus a `kd_weight` curve) and
those checkpoints were scored through BFCL. So upstream's "out of scope: GOLD-trainer
integration" is still true here — the KD path lives in the **SFT** trainer, not
`CustomGOLDTrainer` — but the feature as a whole is not speculative.

### What we added beyond the design

**1. Eight arguments the design doc does not have, and one of them is load-bearing.**
`--teacher-tokenizer`, `--response-template`, `--max-skip-fraction`,
`--allow-tokenizer-mismatch`, `--expectation-file`, `--emit-expectation`,
`--shard-count`/`--shard-index`, `--verify-only`. The design doc derives the tokenizer
from `--teacher-model` and has no notion of an input fingerprint or a completeness check.

**2. `--teacher-tokenizer` must usually point at the *student's* retagged tokenizer, and
getting this wrong cost a precompute plus three arm launches.** That failure is where this
requirement comes from. The reasoning, preserved in `precompute-submit.sh:23-45`: the 30B
teacher's tokenizer directory and the retagged student's agree on
`tokenizer_identity` (both `sha256:883975314d587437`), which was taken as licence to
point at the teacher. **That hash does not cover the chat template**, and the templates
differ in exactly the way that matters — the retagged student carries `{%- generation %}`
markers, the 30B teacher carries none. So pointing at the teacher silently selected the
**response-template scan** instead of the generation-marker mask, and the two disagree by
exactly one token per assistant turn (the markers cover the newline after `<|im_end|>`;
the scan stops at it):

```
RuntimeError: KD: assistant token count from chat template (378) does not match
num_assistant_tokens (377) from index.jsonl for source_idx=0
```

The stored logits are positionally aligned to assistant tokens, so a one-token shift
pairs every one of them with the wrong target. `sft.py`'s per-row cross-check caught it
rather than letting it through — which is the strongest argument in this document for why
that guard should never be relaxed. The rule: **rendering must match what the student
trainer renders, because that is what the logits are aligned to.** The failed artifact is
preserved on disk as
`data/distillation/work/precompute/blend20k-top256.WRONG-MASK-MECHANISM/` rather
than deleted.

**3. Richer provenance than `meta.json` was designed to carry.** Beyond the design's
fields we record `teacher_tokenizer_sha` (content hash, because the tokenizer is a
directory in this tree and an in-place edit of `tokenizer.json` would otherwise be
invisible), `response_template` *and* its resolved ids (`[100256, 78191, 198]`),
`attn_implementation`, full skip accounting (`n_source` / `n_kept` / `n_skipped` /
`skips` / `max_skip_fraction`), the mesh shape (`tp_world_size`, `num_nodes`,
`dp_world_size`), `code_provenance` (source-tree digest, repo sha, branch, **dirty
flag**), and an embedded `expectation` block. The expectation is compared before the
first append and a difference is **refused**: resuming into a directory built from other
inputs would append to one index over two corpora, and every consumer of that index
would be quietly wrong.

`--max-skip-fraction` defaults to 0.05 because silently training on 60% of a corpus is
the failure mode that number exists to make loud. Both real artifacts skipped zero rows,
so it has never fired in anger.

### Where the design over-promises

**Multi-node precompute does not work from here, and the module is not the reason.**
`run-precompute.sh:308` refuses `NODES>1` unconditionally. Confirmed by submitting it —
`NODES=4` was **refused in 8 seconds** — and the refusal is correct rather than a gap to
route around: the row split is `(i % num_nodes) == node_id`, so a job started with
`num_nodes=4` on a world size that never materialized would precompute **one residue
class** of the corpus and look successful. The mesh is genuinely 2-D and each node
already writes its own `index_part_%04d.jsonl` for merge; what is missing is the
**launcher** — nothing hands the step `LSB_HOSTS`, a master IP, or a per-host
`machine_rank`. That plumbing exists and works in
the reference off-/on-policy launchers, so the fix is to port that fan-out here.
It is real work, not a flag. Tracked as D3.

So `world_size_at_precompute` in `meta.json` has only ever read 8, and the only shape
submittable today is **1 node × 8 GPUs**.

`--shard-count` / `--shard-index` are the workaround: independent single-node
residue-class passes, merged afterwards with a companion index-merge tool and
then certified with `--verify-only`. The split is separate from the launcher's refusal
because a shard job *cannot* check completeness — completeness is a property of the
whole — so certification has to be a distinct pass over the merged directory.
`blend20k-top256-shard4` is a completed four-way instance of this.

Projected wall clock at the measured rate, all at the same ~125 GPU-h (parallelism buys
wall clock, not budget):

| rows | steps @192 | wall @ 1×8 | GPU-h | status |
|---|---|---|---|---|
| 20,000 | 100 | 0.4 h | 3.1 | **done** — `blend20k-top256` |
| 192,000 | 1,000 | 3.8 h | 30 | **done** — `blend192k-top256` |
| 802,027 | 4,177 | 15.7 h | 125 | full corpus, 1 epoch — unreached |

One caveat the design does not raise: the split is round-robin **by row**, so nodes get
equal row counts but not equal *token* counts, and the slowest node sets the wall time.
Over 802,027 rows the imbalance should be small, but a corpus whose length structure
correlates with position would skew it — worth reading the per-node progress lines on the
first large run rather than assuming.

### One measured result that changes how to read `kd_weight`

The design doc presents `kd_weight` as *the* knob. Measured over 1,000 steps on this
artifact, in this configuration it is close to a no-op on ranking while it does move
gradient magnitude:

```
blendce  (kd 0.0)  grad_norm  min 7.191  median  8.513  max 245.486
blendkd  (kd 1.0)  grad_norm  min 9.701  median 11.785  max 494.327
```

`ce_weight` is pinned at 1.0 across that curve, which is what makes the existing
`blendce` and `blendkd` checkpoints points *on* the curve rather than a separate
experiment. The honest reading is that raising `kd_weight` raises total loss magnitude as
well as the KD share, so the two effects are not separated by this design — a caveat for
whoever writes up the results, not a defect in the step.

### Not attempted

Cross-tokenizer precompute (ULD alignment would have to happen at precompute time, and
the stored top-K would no longer be in the student's vocabulary), adaptive K, HF Hub
push, and `CustomGOLDTrainer` integration. All four were out of scope upstream and remain
so. No config under `configs/` sets `precomputed_logits_dir` — the blend arms pass it on
the command line from `blend-submit.sh`, so there is no YAML for the config-key gate to
check.
