# Code architecture record

This page is the maintainer record of the code; the reader page is [architecture.md](architecture.md).
It is the shape that maintenance keeps. It lists, for Kefir 1 only: the entries and what
each normalizes to, the one authority for each fact, every state vocabulary and its single
classification, the surface of each component, the memory kept on purpose, what ran versus what
is new, the equivalence evidence, and the open items. Change it in the same change as the code.

Evidence words: **observed** (reproduced here: build, command, positive control named),
**argued** (reasoned, not executed), **unknown** (not checked).

## Contract and non-goals

The repository trains, fine-tunes, evaluates and serves the Kefir 1 mixture-of-experts language
model (144,074,648 parameters, 2048 context, SuperBPE 8000 tokenizer) with the code that
produced its weights. Observers: the trained weights, the checkpoint files and receipts, the
`events.jsonl` training log, evaluation outputs, and the terminal.

Non-goals: provider launchers or account handling of any kind, notebooks, data download for SFT
conversations (the SFT data is not shipped here), automatic retries, automatic winner selection
between recipes, and any model or tokenization other than Kefir 1.

## Layout

```
src/kefir/
  model/ layers/ trainer/ data/   MoE model, GDN-2 + attention layers, pretraining trainer, shard I/O (as ran)
  layers/gdn2_pairwise.py         pairwise FP32 GDN-2 backend used by every SFT run, eval and chat (as ran)
  training.py train.py            pretraining API (RunSpec + ExecutionSpec) and `kefir pretrain`
  artifact.py export.py           the one inference weights form; trusted trainer-state export
  generate.py evaluate.py         raw-text continuation; held-out text NLL + thematic panels
  tokenizer.py assets/            pinned tokenizer (068e20ef...) and 12 thematic validation panels
  sft/                            renderer (Renderer C), data, recipe, train, evaluation, chat, cli
  bench/                          frozen kefir-bench/1 chat suite: build, run, score
  staging.py                      new outputs appear whole or not at all
  cli.py                          the single `kefir` command
data/pretraining/                 run 003 corpus recipe (filters, packing), separate environment
data/sft/                         SFT data selection, package sft_data (questions, filters, overlap, cohorts)
data/tokenizer/                   tokenizer training code (being added)
examples/                         pretraining caller and JSON configs; SFT smoke example and the Kefir 1 SFT record
tests/                            CPU-only suite, including the equivalence goldens
```

## Entries and their normalized values

| Entry | Accepted forms | Normalized to | Failure class at entry |
|---|---|---|---|
| `kefir pretrain --run --execution [--resume]` | RunSpec / ExecutionSpec JSON | `kefir.training.train` → `kefir.model.__main__` | user (unknown field, bad value); plan (identity mismatch on resume) |
| `kefir sft render` | JSONL `{conversation_id, messages}` | token rows `{conversation_id, input_ids, assistant_mask}` via Renderer C | user (empty/duplicate id, non-alternating roles, >2048 tokens: `ContextOverflow`, never truncated) |
| `kefir sft train --recipe --execution [--resume DIR]` | `Recipe` + `Execution` JSON | one `Stream` (rows + batch order) and one `TrainerConfig` on the target clock | user (unknown field, unpinned input), permanent (hash mismatch: the bytes cannot change on retry) |
| SFT `parent` | `.safetensors` weights, or a trainer state pinned by SHA-256 (the pin declares trust) | `MoESmallLLM` + pairwise backend | permanent (hash), user (form) |
| `--resume DIR` | `exposure-NNNNN/{trainer_state.pt, receipt.json}` | start mode `resume` or `fork` (below) | permanent (receipt/sha/prefix mismatch, non-schedule fork, LR discontinuity) |
| weights for `generate`, `chat`, `evaluate` | one `.safetensors` file with `model_config` metadata (the released form); a trainer state only with `--trusted-state` | `kefir.artifact.load_weights` | user (pickle without trust is refused before it is opened) |
| `kefir export --trusted-state --output` | trainer state `.pkl`/`.pt` | `.safetensors` with `model_config`, `tokenizer_sha256`, source sha/step; read back tensor-for-tensor | permanent |
| `kefir bench build --cache --output [--download]` | pinned upstream files (URLs + SHA-256 in `bench/data/source-manifest.json`) | `dev/test/reserve.jsonl` equal to the frozen manifest, or an error | transient (network), permanent (hash) |
| `kefir evaluate chat` | weights + built suite (+ held-out token rows) | `metrics.json`, `responses.jsonl`, `scored.jsonl`, `raw-retention.json` | permanent (split hash differs from the frozen suite) |
| corpus builders | pinned sources and assets | historical builder outputs | permanent; `students.pkl` missing → explanatory `FileNotFoundError` |

## Authorities (one owner per fact)

| Fact | Authority | Derived views |
|---|---|---|
| Chat token layout, reserved IDs 7993/7994/7992, assistant-only mask, 2048 whole-conversation limit | `src/kefir/sft/renderer.py` (byte-identical Renderer C, SHA-256 `d67c871f...`) | token rows, evaluation prompts, chat prompts |
| Tokenizer bytes and ordinary-text/control split | `kefir.tokenizer` + `assets/superbpe_8000_v2.json` (`068e20ef...`) | renderer, corpus encoder, evaluation |
| Row validity | `kefir.sft.data.read_rows` (checks of the run-time worker) | — |
| Consumption order and batch boundaries | `kefir.sft.data.Stream` (each segment batched on its own; frozen order or `random.Random(seed)`) | `prefix_sha256`, `targets_at_chats` |
| Batch geometry (EOD padding to 2048, shifted labels, `-100` elsewhere) | `kefir.sft.data.batch_tensors` | — |
| LR schedule, optimizer, loss (global mean over the batch's Assistant targets), clock | the core: `TrainerConfig`, `TokenLRScheduler`, `MoETrainerEngine` | the recipe's chat-denominated schedule, converted once by `Recipe.trainer_config` |
| Start mode | `kefir.sft.train.classify_start` + the core `load_state_dict` / `load_fork_state_dict` | `receipt.start.mode` |
| Inference weights form | `kefir.artifact` | `generate`, `chat`, `evaluate`, `export`, SFT parent |
| GDN-2 backend at SFT and inference | `kefir.artifact.use_pairwise_backend` (chunk 32) | — |
| Frozen chat suite identity | `src/kefir/bench/data/manifest.json` | `build_suite`, `load_split` |
| Scoring | `kefir.bench.scoring` (snapshot bytes) | `metrics.json` |
| Corpus asset identities | `data/pretraining/assets/manifest.json` | `verify_assets.py`, builders |
| SFT judge questions; selection thresholds | `data/sft/sft_data/questions.json` (hash-checked at import); `sft_data.selection.STRICT`/`INTERMEDIATE` | `label`, `select`, `check` |
| A new result output is refused if it exists and appears only whole | `kefir.staging.staged_output` | `sft render`, `evaluate chat`, `bench build` |
| Reduced precision needs CUDA at inference | the generation core (`trainer/eval_generation.py`, fp16/bf16) and `kefir.sft.evaluation.check_precision` (fp16) | `generate` checks the core's rule before loading (a mirror kept because the core is pinned); `evaluate chat` calls `check_precision` before loading |

## State vocabularies (each with one classification)

- **SFT start mode** `fresh | resume | fork`: no checkpoint → `fresh`; checkpoint whose trainer
  configuration differs from the recipe only in execution fields (`microbatch_size`, cadence) →
  `resume` (exact `load_state_dict`); any other difference → `fork`, handed to the core
  `load_fork_state_dict`, which accepts only learning-rate-recipe changes with a continuous LR
  at the parent clock and rejects everything else. One function, `classify_start`; tested.
- **Schedule kind** `constant | wsd` (recipe). `wsd` has `warmup_chats ≤ decay_start_chats <
  decay_end_chats`; spans become tokens as the Assistant targets in the first N conversations of
  the stream order. The core also knows `wsqd`; the SFT recipe does not expose it (never used).
- **Training events** `setup, resume_verified | fork_verified, resume_validation_verified, train,
  validation, checkpoint, complete, failed`. A run ends in `complete` or `failed`; there is no
  partial-success state, and no wall-clock cap (resume from the last observed checkpoint instead).
- **Pretraining segment result** `complete | incomplete`: `incomplete` when the segment stopped at
  `max_wall_seconds` before `target_tokens`; both exit 0 and the caller resumes with `--resume latest`.
- **Weights input** `safetensors | trusted trainer state | refused`; a pickle is never opened
  without trust.
- **Benchmark split** `dev | test | reserve`; `test`/`reserve` need `--confirm-held-out-split`.
- **Generation termination** `native_end | output_cap` (evaluation); chat warns on the cap.
- **Corpus student asset** `present (hash checked) | missing (explanatory error)`.

## Surfaces

- `kefir` CLI: `pretrain`, `sft {render,train}`, `evaluate {text,chat}`, `generate`, `chat`,
  `export`, `bench build`. One entry point (`kefir = kefir.cli:main`).
- Python: `kefir.train/RunSpec/ExecutionSpec` (pretraining), `kefir.load_weights/save_weights`,
  `kefir.sft.recipe.Recipe/Execution`, `kefir.sft.train.train`, `kefir.sft.evaluation.evaluate_model`,
  `kefir.sft.chat.Chat`.
- SFT recipe fields: `run_id, parent, segments[{path, sha256, order?}], validation,
  global_batch_size, seed, optimizer{TrainerConfig optimizer fields}, schedule, precision,
  initial_loss_scale, observe_at_chats`. Execution: `output, device, microbatch_size, threads`.

Figures: `results/make_figures.py` draws the measurement plots, `results/make_architecture.py` the detailed and the overview architecture diagrams, and `results/make_benchmarks.py` the zero-shot comparison figures and `results/benchmarks.md` from `results/data/benchmarks.json`.

## Theme images (maintenance)

Plots used on release pages have transparent light and dark PNGs. The architecture generator owns the
diagram and shares the same export routine. The dark exports use `rsvg-convert` with the dark palette
applied to the generated SVG. Detailed experimental figures retain their original light export.

The Castor footer PNGs (`results/figures/castor-footer{,-dark}.png`, used by the README footer) are static
transparent exports of `docs/castor-footer.svg`. To reproduce the PNGs locally from the repository root, resolve the two existing
palettes in the master and rasterize at twice the display width:

```python
from pathlib import Path
import re
import subprocess

svg = Path("docs/castor-footer.svg").read_text()
palettes = re.findall(r":root\{([^}]+)\}", svg)
for suffix, declarations in zip(("", "-dark"), palettes, strict=True):
    resolved = svg
    for declaration in declarations.split(";"):
        name, value = declaration.split(":", 1)
        resolved = resolved.replace(f"var({name})", value)
    subprocess.run(
        ["rsvg-convert", "--width", "700", "--output",
         f"results/figures/castor-footer{suffix}.png"],
        input=resolved.encode(),
        check=True,
    )
```

## Memory kept on purpose

- `exposure-NNNNN/receipt.json` next to every SFT `trainer_state.pt`: `checkpoint_sha256`,
  `step`, `assistant_tokens_seen`, `stream_prefix_sha256` (conversation ids batch by batch),
  `parent_sha256`, `recipe_sha256`, start mode, trainer configuration, held-out NLL. Obligation:
  a resume or fork must prove it continues a prefix of the same stream from the same parent, and
  re-measure the held-out NLL within 5e-5 of the receipt. Retirement: none while the checkpoint
  may be resumed; the receipt goes with its state file.
- Pretraining checkpoint identity (`scientific_run` in `checkpoint.json`, local manifests) as ran.
- `tests/data/ran-runtime-manifest.json`: SHA-256 of the 90 runtime files that ran, plus the
  pairwise backend and Renderer C, and the declared list of files changed here. Retirement: when
  a new run is trained from this tree, its own source identity replaces this one.

## What ran versus what is new

"What ran" refers to the original, unpublished run tooling. Its source files are identified below by SHA-256;
[`tests/data/ran-runtime-manifest.json`](../tests/data/ran-runtime-manifest.json) lists the full set.

| Part | What ran | Here |
|---|---|---|
| Model, layers, pretraining trainer, shard I/O | the `src/kefir` runtime of the original run tooling (the 90 files pinned by the SFT plan, byte for byte) | kept byte-identical except 7 entry/inference files (`__init__`, `artifact`, `evaluate`, `export`, `generate`, `train`, `model/evaluation`), declared in the manifest and tested |
| A later variant of that runtime (5 files) | the pinned runtime ran | the pinned runtime kept: `extend_budget`, `thematic_microbatch_size`, the distributed-validation failure vote, `generate --interactive`; the later variant lacks them |
| GDN-2 pairwise backend | the run tooling's pairwise backend (`c3da1400...`) | `layers/gdn2_pairwise.py`, byte-identical |
| Renderer C | the run tooling's frozen chat renderer (`d67c871f...`) | `sft/renderer.py`, byte-identical |
| SFT data checks, order, batching, loop | the run tooling's training, warmup, continuation and decay workers (2×T4, one per experiment) | `sft/data.py` (functions copied), `sft/train.py` (one loop: fresh/resume/fork), `sft/recipe.py` (new: chat-denominated schedule normalized to the token clock) |
| Milestone evaluation | in-loop: frozen suite + held-out NLL + the 80-case Open80 panel | in-loop: held-out NLL only; the suite runs as `kefir evaluate chat` on the saved weights; Open80 prompts are not included |
| Evaluator | the run tooling's evaluator (`eadfd80b...`) with its own renderer and tokenizer copies, and its raw-retention check | `sft/evaluation.py`: same backend logic and functions, prompts from the one Renderer C and the core tokenizer |
| Benchmarks | snapshot of `kefir_bench` (manifest `a13c4d2d...`) | `bench/`: `authored.py`, `io.py`, `scoring.py` byte-identical to the snapshot; `public_data.py` differs only in its data path; `runtime.evaluate_cases` identical; HF/plugin backends, overlap/compare/review tools dropped |
| Chat | the run tooling's chat (HF download, runtime-manifest check, pairwise backend) | `sft/chat.py`: same context fitting, sampling and streaming; no download |
| Inference weights | released SFT weights: one `model.safetensors` with `model_config` metadata; core artifact was a directory (`model.json` + weights) | one form: the released single file; the directory form is gone |
| An earlier draft of this public tree | never ran | its shape (renderer module, row validation, one CLI, chat) used; its behavior (max-length padding, no order, single process) not used: padding width changes router load balancing, so the run-time 2048 padding is kept |

## Equivalence evidence

Proved on CPU, FP32, one thread, tiny 2048-context model with the real 8000 vocabulary, by
running the run-time code (the run tooling's workers and evaluator, the benchmark
snapshot) and the public code on the same synthetic inputs, then pinning the run-time outputs as
goldens in `tests/test_sft.py` and `tests/test_provenance_and_inference.py` (**observed**):

- Renderer and tokenizer: the original evaluator's renderer/tokenizer produce exactly the public
  Renderer C prompts for all 766 generated cases of the frozen suite and identical encodings for
  2,039 case texts; 500 random decodes equal. Token-row files equal byte for byte (golden hashes).
- Rows, order and batches: identical batch lists (frozen order and seeded shuffle) and identical
  `(inputs, labels)` tensors for every update (golden hash), across a two-segment stream.
- Schedule spans: warmup tokens, stable end and total from the warmup-worker formulas equal the
  recipe conversion.
- Flat run: final weights bitwise equal to the run-time sequence of core calls; per-update LR equal;
  in-loop validation does not perturb training (bitwise; positive control: other schedules differ).
- Warmup arms: run-time "shared prefix, then `load_fork_state_dict`" equals the public straight
  arm and the public fork from the prefix checkpoint, bitwise, for warmup-decay and
  warmup-stable-decay; LR sequences pinned.
- Late decay fork from the flat segment endpoint: bitwise equal to the run-time decay-worker
  transformation; LR sequence pinned; exact resume equals uninterrupted.
- Loss: the step loss equals the summed cross-entropy over Assistant targets divided by the
  batch's target count.
- Two ranks: `torchrun --nproc_per_node=2` (gloo, CPU) gives weights bitwise equal to one process,
  including uneven tail batches.
- Evaluation: held-out NLL equal; suite responses, scores and raw-retention rows equal on a mixed
  case sample.
- Frozen suite: authored cases rebuild offline to the pinned hashes; with the snapshot's public
  selections, reassembly reproduces `dev/test/reserve` hashes of the frozen manifest.

Not proved (**unknown** unless stated):

- CUDA FP16 with dynamic loss scaling, T4 kernels, NCCL: not runnable here. The same core code
  ran there; equality of trajectories on GPU is **argued** from code identity, not observed.
- The real 144M model, real SFT data and the released weights file: not available here; loading
  the released metadata layout is tested on a synthetic file of the same form.
- Base-model generation now uses the pairwise backend instead of the chunkwise default: equal up
  to FP32 round-off (**observed** at 2e-5 on a tiny model), not bitwise.
- Benchmark scores computed out of loop equal the in-loop ones by construction (the in-loop
  evaluator restored RNG and buffers): **argued**.
- `read_rows` rejects supervised reserved controls and an unsupervised first token but cannot
  detect a mask on ordinary header text; Renderer C is the authority for that (as at run time).
- Pretraining run 003 itself: "as ran" above means the core that ran the supervised fine-tuning runs. Run 003 and its cooldown were trained from an earlier source tree (SHA-256 identities in [provenance](provenance.md)); equivalence of that tree with this core is **unknown**, and nothing is re-trained here.

## Removed

Material unrelated to Kefir 1, research handoff notes, the
provider launch example, provider launchers and multi-account logic, HF upload/publication tools of the
SFT, Open80 prompts, the HF/plugin benchmark backends, the duplicate inference loader
in `model/evaluation.py`, the CUDA-only test, `data/pretraining/assets/students.pkl`.

## Open items

1. `data/pretraining/assets/students.pkl` (SHA-256 `b8b84722...`) is not bundled in the MIT code tree.
   The CC BY-NC 4.0 classifier artifact is listed in [NOTICE](../NOTICE.md); the corpus builder
   requires that file and checks its hash.
2. Redactions change two pinned corpus assets (`contract.json`, `FILTER-PROVENANCE.json`); their
   original hashes stay in `data/pretraining/assets/manifest.json`. A rebuilt raw-region contract hash now
   differs from the historical one (the builder embeds `contract.json`).
3. Benchmark licensing: selected public cases are rebuilt from upstream, not shipped (BLiMP CC BY
   4.0, ARC CC BY-SA 4.0, PopQA MIT). The authored cases and code now fall under the repository
   MIT licence.
4. SFT checkpoints produced before this tree have no `receipt.json` of this schema; resuming or
   forking them through `kefir sft train` needs a receipt written from their recorded step,
   token count and order. The published conversations alone do not supply those receipts.
5. The historical SFT recipe `examples/configs/sft-recipe-k1-b32-s17.json` carries the run-time
   hashes where known and `"unknown"` for the three not recorded in the public context (the
   frozen order, the continuation rows and their order); the parser refuses it, so it is a record,
   not a runnable file. `examples/sft-smoke/` is the runnable SFT example.
6. Comments in byte-identical files still name the cloud providers once used; changing them would break byte identity.
7. [`provenance/source-manifest.json`](provenance/source-manifest.json) records the original
   pretraining source tree. Corpus asset identities are recorded separately in `data/pretraining/assets/`.

The bundled tokenizer and its Hugging Face card use the MIT licence, as recorded in [NOTICE](../NOTICE.md).
