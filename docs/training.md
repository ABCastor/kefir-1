# Training

Kefir 1 has two training entry points, both on the `kefir` command and both able to run on one process or on two
GPUs under `torchrun`:

- `kefir pretrain --run RUN.json --execution EXECUTION.json [--resume latest|step-NNNNNNNN]`
- `kefir sft train --recipe RECIPE.json --execution EXECUTION.json [--resume exposure-NNNNN]`

Each takes two JSON files. The **scientific** file (run / recipe) holds everything that decides the weights; the
**execution** file holds hardware, output and cadence, which may change between segments of the same run. Unknown
fields are rejected at entry. More than two ranks are rejected.

## Pretraining

### Input data

Pretraining reads a prepared schema-v2 corpus: `dataset_dir` contains `manifest.json` and the train shards it
names (validation shards too when `validation_blocks > 0`). The manifest must record the pinned tokenizer SHA-256
under `tokenizer.tokenizer_sha256`; a mismatch is rejected. The corpus is built with the recipe in
[`data/pretraining/`](../data/pretraining/README.md); see [provenance](provenance.md) for the run 003 dataset identity.

### Run locally

The two files in [`examples/configs/`](../examples/configs/) make **one CUDA BF16 update** on the real 64-sequence
geometry against `data/prepared-schema-v2`. They are a smoke test, not the run 003 recipe.

```bash
kefir pretrain --run examples/configs/local-run.json --execution examples/configs/local-execution.json
python examples/train.py examples/configs/local-run.json examples/configs/local-execution.json   # same path, from Python
```

`RunSpec` (run file) fields: `run_id`, `dataset_dir`, `target_tokens` and `max_steps` (absolute, cumulative
limits), `training` (optimizer, learning rate, schedule and their parameters; `optimizer`, `learning_rate` and
`schedule` are required), `global_batch_size` (64), and optionally `data_manifest`, `parent_checkpoint`,
`run_source_commit`, `extend_budget`. The model is always the accepted Kefir 1 configuration and the tokenizer
always the pinned one. The `training` block also accepts the historical run-003 schedule fields listed in
`_SCIENTIFIC_FIELDS` in `src/kefir/training.py`.

`ExecutionSpec` (execution file) fields include `run_root`, `microbatch_size`, `device`, `precision` (`bf16`
default; `fp32` for CPU diagnostics), `checkpoint_every_steps`, `evaluation_every_steps`, `validation_blocks`,
`keep_last_checkpoints`, `max_wall_seconds`, `thematic_panels` (the panel directory, to log per-theme validation
loss) and `thematic_microbatch_size` (default 32; lower it if panel evaluation does not fit on the GPU; it changes
no gradient). Weights & Biases logging and Hugging Face transport of checkpoints or shards are optional and off by
default; credentials are read from the environment variables named in the execution file (see [`examples/configs/env.example`](../examples/configs/env.example)).

Checkpoints are written to `run_root/<run_id>/checkpoints/step-NNNNNNNN/` (`trainer_state.pkl` and
`checkpoint.json`). Each checkpoint records the run identity, the trainer configuration, tokenizer and dataset
identity under `scientific_run`, and a resume checks this contract.

### Two GPUs

The global batch is split exactly across ranks and micro-batches; each update still uses 64 sequences, and a
combination that does not divide it is rejected.

```bash
torchrun --standalone --nproc-per-node=2 -m kefir.cli pretrain --run run.json --execution execution.json
```

### Resume

```bash
kefir pretrain --run run.json --execution execution.json --resume latest
kefir pretrain --run run.json --execution execution.json --resume step-00280000
```

Each call ends by printing one `training_result` line. `status` is `complete` when `target_tokens` is reached and
`incomplete` when the segment stopped earlier at `max_wall_seconds`; both exit 0, since a stopped segment is a normal
step of a long run (a non-zero exit would make torchrun report a failure). A caller that loops reads `status` and
runs the same command with `--resume latest` until it is `complete`; errors exit non-zero.

A run that already has checkpoints refuses to start without `--resume`. `latest` fails on a corrupt newest
checkpoint instead of silently rolling back; a deliberate rollback names a checkpoint ID. An ordinary resume
rejects any change to the scientific configuration, including the learning-rate schedule. A checkpoint pins its
precision, so changing precision on resume is rejected.

To continue a run that reached its limits, set `extend_budget: true` with larger `max_steps` and `target_tokens`
and resume normally. Both previous bounds must have been reached and every other scientific field must match.
This keeps optimizer, scheduler, routing and data state.

### Fork: change the learning-rate curve (cooldown)

To change the learning-rate curve from an existing checkpoint, create a **new** run with a new `run_id` and set
`parent_checkpoint` to the absolute path of a complete parent `step-…` directory. Model, dataset identity, global
batch, optimizer and precision stay unchanged; the new curve must meet the parent's learning rate at the fork
point. The first call creates a child checkpoint at the parent step that records a digest of the parent state,
then continues from the saved data cursor; later calls use `--resume latest` on the child. The parent files are
never rewritten.

The Kefir 1 cooldown (C30) is such a fork of step 280,000: a WSD cosine curve with no warm-up, the stable phase
ending at the parent's committed token count, and a decay from 3e-4 to 3e-5 (`minimum_lr_ratio` 0.1) over the
next 30,000 updates. With 131,072 targets per update the token counts follow from the step counts. A run file of
this shape (the exact C30 file is not part of this repository):

```json
{
  "run_id": "my-cooldown",
  "dataset_dir": "data/prepared-schema-v2",
  "parent_checkpoint": "/absolute/path/to/runs/run-003/checkpoints/step-00280000",
  "target_tokens": 40632320000,
  "max_steps": 310000,
  "global_batch_size": 64,
  "training": {
    "optimizer": "hybrid_muon_adamw",
    "learning_rate": 0.0003,
    "schedule": "wsd",
    "warmup_tokens": 0,
    "stable_tokens": 36700160000,
    "decay_tokens": 3932160000,
    "minimum_lr_ratio": 0.1
  }
}
```

With `warmup_tokens` 0, `stable_tokens` is the end of the stable phase on the absolute token clock (280,000 ×
131,072); `target_tokens` and `max_steps` are absolute and larger than the parent's counters. The `training` block
must otherwise repeat the parent's optimizer fields.

### Streaming shards

For a corpus staged incrementally, `examples/stage_incremental.py` prepares a checkpoint-aligned bootstrap
manifest; the run file points `data_manifest` at it, and the execution file names the shard source
(`dataset_shard_bucket`, `dataset_shard_run_id`, prefetch and wait timeout) or `local_incremental_frontier`. The
trainer then reads future verified shards without changing the frozen checkpoint identity.

### Precision

Pretraining ran in BF16 with the `fla-core` Triton GDN-2 kernels (`uv sync --extra fla`): run 003 on one H100
(about 0.37 s per update over its recorded update time) and the cooldown C30 on 2× T4 (about 7.3 s per update; the
T4 has no native BF16, so it is much slower but works). The first update of a new process compiles the kernels:
about 2–3 minutes on an H200 or 2× A10G, while on Kaggle's 2× T4 a three-update run completed only after about
30 minutes, mostly compilation. Plan short T4 runs accordingly. FP16 single-update probes matched the
loss but did not reproduce the BF16 parameter update within a strict tolerance, so FP16 is not qualified for
continuing a BF16 checkpoint; keep the checkpoint's precision.

## Supervised fine-tuning

### Try it: the smoke example

[`examples/sft-smoke/`](../examples/sft-smoke/) fine-tunes the released base weights on 12 short conversations
(batch 8, two updates) and measures held-out NLL on 4 more. From the repository root:

```bash
hf download abcastor/kefir-1-base model.safetensors --local-dir weights/kefir-1-base
echo "11ca59f45f8ebcfd019ed5a027198d6adb549502e3d118a031f4bec8931888ac  weights/kefir-1-base/model.safetensors" | sha256sum -c
kefir sft render --conversations examples/sft-smoke/train.jsonl --output runs/sft-smoke/train-tokens.jsonl
kefir sft render --conversations examples/sft-smoke/validation.jsonl --output runs/sft-smoke/validation-tokens.jsonl
kefir sft train --recipe examples/sft-smoke/recipe.json --execution examples/sft-smoke/execution.json
kefir export --trusted-state runs/sft-smoke/train/exposure-00012/trainer_state.pt --output runs/sft-smoke/model.safetensors
kefir chat --weights runs/sft-smoke/model.safetensors --prompt "What season comes after winter?"
```

The recipe pins the base weights and the rendered rows by SHA-256, so the paths above are part of it. It needs a CUDA
GPU (FP16, the Kefir 1 precision; a T4 is enough); fine-tuning on CPU is possible with `"precision": "fp32"` and
`"device": "cpu"` but slow and memory-hungry (about 10 GB). Two updates on 12 chats do not make a chat model; the example only proves the whole path runs.

### 1. Render conversations

Input: JSONL, one object per conversation, `{"conversation_id": "...", "messages": [{"role": "system|user|assistant",
"content": "..."}]}`. Roles alternate User/Assistant after an optional first System message, and the last message
is from the Assistant.

```bash
kefir sft render --conversations train.jsonl --output train-tokens.jsonl
kefir sft render --conversations validation.jsonl --output validation-tokens.jsonl
```

The output rows are `{"conversation_id", "input_ids", "assistant_mask"}`. A conversation longer than 2,048 tokens
stops rendering with `ContextOverflow`; filter such conversations beforehand. The output file must not exist and
appears only when every conversation rendered. Training and validation must not share a conversation ID.

### 2. The parent weights

The recipe's `parent` is either a `.safetensors` weights file (the released base model) or a pretraining trainer
state you trust, pinned by SHA-256 (the pin declares trust; a pickle is executable input).

### 3. Recipe and execution

Runnable files: the smoke example above. [`sft-recipe-k1-b32-s17.json`](../examples/configs/sft-recipe-k1-b32-s17.json)
is a record of the Kefir 1 flat batch 32 seed 17 arm (15,000 chats), not a runnable file: the SHA-256 values of the
frozen order and the continuation rows and order are missing from that record, so they read
`"unknown"` and the parser refuses the file. [`sft-execution.json`](../examples/configs/sft-execution.json) is the
execution it ran with (the defaults: CUDA, micro-batch 2, 2 threads). For your own data, pin every input with
`sha256sum`; a malformed or all-zero hash is rejected when the recipe is read.

Recipe fields:

| Field | Meaning |
|---|---|
| `run_id` | name recorded in events and receipts |
| `parent` | `{path, sha256}` of the starting weights or trainer state |
| `segments` | list of `{path, sha256, order?}` token-row files consumed in sequence; `order` is an optional pinned JSON list of row indexes, otherwise the order is `random.Random(seed)` |
| `validation` | `{path, sha256}` of held-out token rows |
| `global_batch_size` | conversations per update (Kefir 1: 32 or 64) |
| `seed` | data order and initial RNG |
| `optimizer` | `TrainerConfig` optimizer fields: `optimizer` and `learning_rate` (required), `weight_decay`, `beta1`, `beta2`, `adam_epsilon`, `muon_*`, `max_grad_norm`, `max_overflow_retries` |
| `schedule` | `{"kind": "constant"}` or `{"kind": "wsd", "warmup_chats" (default 0), "decay_start_chats", "decay_end_chats", "minimum_lr_ratio" (default 0.1)}` with integers `0 <= warmup_chats <= decay_start_chats < decay_end_chats` |
| `precision`, `initial_loss_scale` | `fp16` (default, CUDA only) with initial loss scale 256 (a positive number), or `bf16` / `fp32` |
| `observe_at_chats` | increasing conversation counts at which to checkpoint and measure held-out NLL (the end of the stream is always observed) |

Every field value is checked when the recipe is read, before any input is opened; each input's bytes are then
checked against its SHA-256 (a mismatch names both hashes).

An observation at `N` chats happens at the first batch boundary at or after `N`, and its folder is named by the
chats actually seen there: at batch 32, `6000` gives `exposure-06016` (after update 188). Held-out NLL is measured
in FP16 for an `fp16` recipe and in FP32 otherwise, `bf16` included; the `setup` event records it as
`validation_precision`.

Relative paths resolve against the recipe's directory. Execution fields: `output` (a new directory, relative to
the working directory), `device` (default `cuda`), `microbatch_size` (rows per forward pass per rank; default 2, as
Kefir 1), `threads` (default 2).

### 4. Train

```bash
kefir sft train --recipe recipe.json --execution execution.json                     # one process
torchrun --standalone --nproc-per-node=2 -m kefir.cli sft train \
    --recipe recipe.json --execution execution.json                                 # two GPUs
```

Each rank receives the same global batch and the core step splits it; two ranks give the same weights as one
process (proved bitwise on CPU; on 2× T4 and 2× A10G the smoke example's held-out NLL after two updates
agreed with one GPU to 0.03%, the size of GPU reduction-order noise). Under `torchrun` each rank uses `cuda:LOCAL_RANK`, so the execution `device` must
be `cuda` (an explicit index is rejected); global rank 0 writes the outputs and prints the result. The output
directory must not exist (checked on every rank before the ranks connect). It gets `events.jsonl` (setup, train,
validation, checkpoint, complete or failed) and one `exposure-NNNNN/` folder per observation, with
`trainer_state.pt` and `receipt.json`.
There is no wall-clock cap and no automatic retry: if a job stops, resume from its last checkpoint.

### 5. Resume or fork

```bash
kefir sft train --recipe recipe.json --execution new-execution.json --resume runs/k1-b32-s17-15k/exposure-10000
```

The execution file must name a **new** output directory. The start mode is decided by one comparison of the
checkpoint with the recipe:

- **resume**: only execution fields differ (micro-batch size, cadence); the full trainer state is loaded and the
  stream continues where it stopped;
- **fork**: the learning-rate recipe differs; any other change, or a discontinuous learning rate at the fork, is
  rejected. The Kefir 1 late-decay arm was such a fork of the flat run at `exposure-10000`: the same recipe with
  `run_id` `k1-b32-s17-late-decay-10k-15k`, `schedule` `{"kind": "wsd", "warmup_chats": 0, "decay_start_chats":
  10000, "decay_end_chats": 15000, "minimum_lr_ratio": 0.1}` and `observe_at_chats` `[11000, 12000, 13000, 14000]`.

Either way the checkpoint must match its receipt, the consumed conversations must be a prefix of this recipe's
stream, the parent must be the same, and the held-out NLL is re-measured and must agree with the receipt within
5e-5. SFT checkpoints written by the original run-time workers have no receipt of this schema, so they cannot be
resumed this way without one.

### 6. Export the result

The SFT loop writes trainer states. Convert the final one into the released weights form:

```bash
kefir export --trusted-state runs/k1-b32-s17-15k/exposure-15000/trainer_state.pt --output kefir-1-sft-15k.safetensors
```

The `.safetensors` file carries `model_config`, the tokenizer hash and the source hash and step in its metadata,
and is read back tensor for tensor after writing. `kefir chat` and `kefir evaluate chat` load only this form unless
`--trusted-state` is passed; `kefir generate` and `kefir evaluate text` load only this form.

### Resources measured in the Kefir 1 runs

On 2× T4 in FP16 with micro-batch 2 per GPU: median 11.7 s per update at batch 32 and 22.3–23.0 s at batch 64,
59.7–61.4 minutes of optimizer time per 10,000 chats, peak 13.2–13.3 GiB allocated per GPU. See
[`results/README.md`](../results/README.md).
