# Method

This page describes what Kefir 1 is and how each stage was done. Measured numbers are taken from
[`results/`](../results/README.md); invariants of the code are recorded in the
[code architecture record](code-record.md), which is the authority when this page and the code disagree.

## Model

Kefir 1 is an eight-layer decoder. The layer sequence is `GDN-2, GDN-2, GDN-2, attention`, repeated twice:

- **Gated DeltaNet 2 (GDN-2)** blocks are linear-recurrent sequence mixers with a chunked computation (chunk
  32). Pretraining on GPU uses the `fla` kernels; SFT, evaluation and chat use a pure-PyTorch pairwise FP32
  backend (`src/kefir/layers/gdn2_pairwise.py`), the one every SFT run used.
- **Attention** blocks are full causal multi-head attention with 4 heads of 64, QK normalization, rotary
  position embeddings and a sigmoid output gate.
- **Every layer's feed-forward block is a mixture of experts**: 64 SwiGLU experts with hidden width 352, top-2
  routing. Router scores are square-root softplus; dispatch is dropless (padded batched GEMM); the two selected
  outputs are combined with normalized weights. Quantile load balancing adjusts expert selection without adding
  a balancing term to the loss.

Model width is 256 and the context is 2,048 tokens. The model has 144,074,648 parameters (counted on the loaded
model), of which 9,987,992 are active per token (total − all expert weights + 2 experts per layer, an expert being
3 × 256 × 352 weights). That active count includes the tied 8,192 × 256 embedding/output matrix (2,097,152
weights); without it, 7,890,840 active non-embedding parameters remain, the count used for compute estimates and
the external comparison. A simplified diagram is in the [README](../README.md#architecture); the detailed diagram
and the parameter breakdown are in [architecture.md](architecture.md).

The objective is next-token cross-entropy over the 8,000 semantic IDs; embedding rows
8,000–8,191 exist only for alignment and are never inputs or output classes.

Code: `src/kefir/model/` (configuration, routing, forward pass, optimizer step) and `src/kefir/layers/`.

## Tokenizer

SuperBPE with 8,000 IDs, pinned by SHA-256 `068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb`
(`src/kefir/assets/superbpe_8000_v2.json`). IDs 0–7991 are ordinary text; 7992 is the document end (EOD); 7993–7999
are reserved controls. `kefir.tokenizer` encodes ordinary text with the control IDs removed from the vocabulary, so
user text can never produce a control token. The tokenizer was trained earlier on a separately filtered English
seed, so its filtering differs from the corpus recipe below. Compare models with different tokenizers in bits per
byte, not per token.

## Pretraining corpus

The source is [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) at revision
`5eaa64b9c0c85b7f56af01d7dffdb0795816b12b` (licensed CC BY-NC 4.0). The recipe, in
[`data/pretraining/`](../data/pretraining/README.md):

1. **Decode and filter.** Source documents are decoded from their GPT-2 token IDs. Programming cluster 11 is
   excluded; the other 19 cluster labels are kept. A structural filter (`filters_v4.classify_v4`) screens
   code-like, mathematical, non-English-script and boilerplate text. fastText requires a word-weighted mean English
   probability ≥ 0.65 with at most 30 % of chunks below 0.5. A student filter rejects documents with predicted
   `poor_english ≥ 0.80`, `not_prose ≥ 0.80` or `symbolic ≥ 0.70`. The student models are five logistic regressions
   on hashed character and word n-gram TF-IDF features, distilled from the labels that a decision model gave to
   15,922 random source documents. The judge's labels cover five properties of each document: the English is
   poor, i.e. spun, machine-translated, garbled or truncated (`poor_english`); the text is not connected English
   prose, e.g. lists, menus, tables, keyword pages or other languages (`not_prose`); the text is symbolic
   (`symbolic`); the text is largely verbal argument or explanation (`reasoning_dense`); the text states, lists,
   narrates or advertises without reasoning (`reasoning_none`). Each student predicts the probability that the judge would say
   so. The rejection thresholds were chosen precision-first on a held-out 15 % of the labelled documents (a false
   reject costs little because source text is abundant; a false accept is the failure): `poor_english` at 0.80 had
   precision 0.81 against the judge and flagged 2.1 % of the stream, `not_prose` at 0.80 had precision 0.85 and
   flagged 1.1 %, `symbolic` at 0.70 had precision 0.73 and flagged 0.5 %. Held-out ROC AUC against the judge's labels
   is recorded in [`data/pretraining/assets/FILTER-PROVENANCE.json`](../data/pretraining/assets/FILTER-PROVENANCE.json).
   `reasoning_dense` and `reasoning_none` are kept as scores, not rejection rules. Their held-out agreement measures
   agreement with the judge, not human quality, and automated filters both reject good text and keep bad text.
2. **Overlap with validation.** Documents that match the frozen validation seed exactly or by paragraph are
   removed.
3. **Tokenize and pack.** Accepted documents are encoded with the SuperBPE tokenizer, with EOD after each
   document. The packer fills 2,049-token records with documents from one cluster, allocates each 64-record
   block by filtered cluster proportions, rotates through quartiles of `reasoning_dense` score and of mean
   document length within clusters, and finally permutes blocks with a fixed seed. A record gives 2,048 input
   positions and 2,048 shifted targets; EOD padding is an ordinary target, as in run 003.

According to the run 003 dataset manifest, filtering accepted 145,656,118 documents from 2,100 build regions, and
packing produced 763,004 training blocks and 813 validation blocks. Run 003 changed filtering, tokenization and
ordering together with respect to earlier runs, so its loss does not isolate any one of them. The student
classifier file is not bundled with the code; its separate distribution and licence are listed in [NOTICE](../NOTICE.md).

## Pretraining recipe

| | Run 003 (steps 0–280,000) | Cooldown C30 (steps 280,000–310,000) |
|---|---|---|
| Global batch | 64 × 2,048 = 131,072 targets per update | same |
| Optimizer | hybrid Muon / AdamW: weight decay 0.1, Muon momentum 0.95, gradient clipping 1.0 | same, continued state |
| Precision | BF16 | BF16 |
| Learning rate | linear warm-up to 3e-4 (reached at step 38,147), then constant | cosine from 3e-4 to 3e-5 |
| Hardware (recorded update time) | 1 GPU, H100 per the run report (28.76 h) | 2× T4 (60.60 h) |

The cooldown is a fork: a new run whose parent is the step-280,000 checkpoint, keeping model, data order,
optimizer state and batch, and changing only the learning-rate curve, which must be continuous at the fork. It
consumed 3,932,160,000 targets, 9.68 % of the 40,632,320,000 total. Validation loss on 2,097,152 held-out targets
went from 2.638766 to 2.530763 nats per target. There is one pretraining seed and no constant-LR control for the
cooldown, so its gain mixes extra tokens with the lower learning rate.

Twelve thematic panels (`src/kefir/assets/thematic_panels_v1/`, at most 32 windows each, assigned by keywords)
report teacher-forced loss per theme during training. They are diagnostics, not training losses and not a task
benchmark; their overlap with the source corpus was not ruled out.

## Supervised fine-tuning

### Data

The SFT conversations are English chats drawn from a quality-filtered pool: 100,000 candidate conversations from
public instruction and chat datasets were labelled by an open-weights decision model, 28,058 passed the strict filter
and 27,854 remained after removing overlaps with benchmark prompts and shared groups. The selection code (judge
questions, filters, thresholds, overlap removal, stratified cohorts) is in [`data/sft/`](../data/sft/README.md).

The judge read each conversation as plain text after a fixed preamble and answered ten fixed questions, each one
separately. Seven are yes/no questions, answered with the probability of yes: `english_only` (all natural-language
text is English in Latin script), `excluded_formalism` (any turn contains code, shell commands, structured data or
markup, LaTeX, or symbolic mathematics or logic), `instruction_fidelity` (the assistant follows the material
instructions and supplied context of each turn), `content_error` (an assistant reply makes a substantive factual,
numerical or reasoning error), `reasoning_required`, `context_required` and `factual_uncertainty`.
`english_quality` rates the English of the weakest assistant reply on a 0–4 rubric and returns the expected level;
`task` and `subject` return one category each. The eight numeric answers become scores in 0–1 (the probability of
yes, or the English level divided by 4). The strict filter keeps a conversation only when all five hold:
`english_only ≥ 0.99`, `english_quality ≥ 0.90`, `instruction_fidelity ≥ 0.95`, `content_error ≤ 0.05` and
`excluded_formalism ≤ 0.01`; the other five answers describe the conversation and are used to stratify the
selection. The judge protocol is generic ([judge interface](../data/sft/README.md#plugging-in-a-judge)): any model that
answers the same questions in the same answer format can be used, but its probabilities, and so the share each threshold keeps, will differ. From that pool, 10,000 conversations form the first
training segment (7,036,586 assistant targets) and 5,000 further unseen conversations the continuation (3,456,531
assistant targets); 512 disjoint conversations (364,453 assistant targets) are the validation set. The selection
keeps the pool's proportions of source, task, subject, length, number of turns and System messages.

For the long continuation a second 100,000 candidates were labelled. Of the 200,000, 56,661 pass the strict
filter (39,907,627 assistant tokens): 40,968 from `HuggingFaceTB/smol-smoltalk`, 7,744 from
`HuggingFaceH4/ultrachat_200k`, 3,825 from `allenai/tulu-3-sft-personas-instruction-following`, 2,056 from
`HuggingFaceTB/smoltalk2`, 756 from `OpenAssistant/oasst2`, 575 from `databricks/databricks-dolly-15k`, 391 from
`HuggingFaceH4/no_robots` and 346 from `OpenAssistant/oasst1`. The long continuation used 40,490 of them that the
first 15,000 and the validation set had not used. Licences of the sources: [NOTICE.md](../NOTICE.md). The
conversation data is not distributed with this repository.
The [labelled SFT pool](https://huggingface.co/datasets/abcastor/kefir-1-sft-labelled-pool) records the five thresholds used for the strict selection and the original split membership. The [selected conversation view](https://huggingface.co/datasets/abcastor/kefir-1-sft-conversations) exposes the train and validation splits directly.

### Renderer

One renderer, Renderer C (`src/kefir/sft/renderer.py`, SHA-256 `d67c871f…`), turns a conversation into token IDs
for training, evaluation and chat:

```
[7993] "system\n"    <system text>    [7994]      optional, first message only
[7993] "user\n"      <user text>      [7994]
[7993] "assistant\n" <assistant text> [7992]      supervised: text and EOD
...
```

Roles must alternate User/Assistant after an optional System message, and a training conversation ends with an
Assistant reply. Only the Assistant text and its EOD are targets; headers, System and User text are context.
The whole conversation must fit 2,048 positions: longer ones are rejected (`ContextOverflow`), never truncated.
At generation time the prompt ends with `[7993] "assistant\n"` and the reply ends at EOD.

### Training

- **Rows, order and batches.** Each conversation is one row; there is no packing. Each training segment is
  ordered on its own (a frozen order file, or `random.Random(seed)`), cut into global batches, and the segments are
  concatenated: at batch 32 that is 313 + 157 updates for 10,000 + 5,000 chats. Every row is padded with EOD to the
  full 2,048 positions (padding width changes router load balancing, so the run-time geometry is kept); padding
  positions are not targets.
- **Loss.** Mean NLL over all assistant targets of the global batch (text and EOD), summed across micro-batches
  and ranks and divided by the batch's target count.
- **Optimizer.** The pretraining hybrid Muon/AdamW with a fresh state at SFT start, gradient clipping 1.0,
  peak LR 3e-5, weight decay 0.1, β = (0.9, 0.95), Muon momentum 0.95. FP16 with dynamic loss scaling (initial
  scale 256, at most 3 overflow retries); no overflow retry occurred in the runs.
- **Schedule clock.** A recipe names its schedule in conversations (`warmup_chats`, `decay_start_chats`,
  `decay_end_chats`); the recipe converts these once into token spans on the committed assistant-target clock,
  which is what checkpoints store and resumes compare.
- **Checkpoints.** At each `observe_at_chats` count and at the end, the loop measures the held-out NLL and writes
  `exposure-NNNNN/trainer_state.pt` with a `receipt.json` that pins the state hash, step, targets seen, the
  consumed conversation prefix, parent and recipe. A resume or fork must prove it continues a prefix of the same
  stream from the same parent and re-measure the held-out NLL within 5e-5 of the receipt.

### The batch screen

All arms start from the step-310,000 parent, on 2× T4 with micro-batch 2 per GPU. They compared batch size 32 and
64 at constant LR, and four schedules: flat; late decay (a fork of the flat run at 10,000 chats, cosine to 3e-6
over the last 5,000); warm-up + decay (512-chat warm-up, then decay to 3e-6 at 15,000); warm-up + stable + decay
(flat until the target count of the first 5,000 chats, then decay). Seeds 17 and 18 change chat identities and
order, not the parent. Batch 16 and flat batch 32 seed 18 were planned but never ran. Results are in the
[results](../results/README.md#supervised-fine-tuning).

### The long continuation

The flat batch 32 seed 17 run was then continued from its full 15,000-chat state (weights, optimizer, loss scaler
and random generators) with the same recipe, constant LR 3e-5, over 40,490 further conversations in one frozen
order, to 55,490 chats and update 1,736. Two arms saw the same conversations in the same order. In the second arm
a System prompt was added in front of 40 % of the conversations (16,193; where a conversation already had a System
message, the added text was placed before it); assistant targets are identical in both arms. The finished checkpoint of the added-System arm is Kefir 1. Every 2,000 chats the
loop saved the weights and measured the held-out NLL with and without one added System message. The benchmark
suite was run afterwards on every saved checkpoint, on CPU in FP32. The code that added the System prompts is not
part of this repository.

## Evaluation design

Three kinds of measurement, described in [evaluation.md](evaluation.md):

- **Text likelihood** of base checkpoints: validation loss on held-out packed blocks, the 12 thematic panels, and
  bits per byte on 384 held-out shared-text documents with a paired document bootstrap.
- **Held-out assistant NLL** of SFT checkpoints on the 512 validation conversations.
- **The frozen `kefir-bench/1` chat suite** (1,022 cases in development, confirmation and reserve splits), scored
  by exact rules with no aggregate score, plus a raw-retention probe of answer likelihood without chat formatting.

Only the development split was used to compare recipes. Separately, Kefir 1 Base and the chat model were compared
zero-shot with six small open models in lm-evaluation-harness 0.4.13: [results/benchmarks.md](../results/benchmarks.md).
