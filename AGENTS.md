# Working on Kefir 1

Instructions for coding agents and contributors. Read [README.md](README.md) for scope and
[docs/code-record.md](docs/code-record.md) before changing code: it is the authority for the entries, the one
owner of each fact, the state vocabularies, what ran versus what is new, and the open items. Change it in the same
change as the code it describes.

## Layout

| Path | Role |
|---|---|
| `src/kefir/model/`, `layers/`, `trainer/`, `data/` | MoE model, GDN-2 and attention layers, pretraining trainer, shard I/O (as ran) |
| `src/kefir/training.py`, `train.py` | pretraining API (`RunSpec`, `ExecutionSpec`, `train`) and `kefir pretrain` |
| `src/kefir/sft/` | Renderer C, token rows and stream order, recipe, training loop, evaluation, chat |
| `src/kefir/bench/` | frozen `kefir-bench/1` suite: build, run, score |
| `src/kefir/artifact.py`, `export.py`, `generate.py`, `evaluate.py` | `.safetensors` weights form, export, generation, text evaluation |
| `src/kefir/tokenizer.py`, `assets/` | pinned tokenizer and thematic panels |
| `src/kefir/staging.py` | the one rule for new outputs: refused if they exist, written whole or not at all |
| `src/kefir/cli.py` | the single `kefir` command |
| `data/pretraining/` | pretraining corpus recipe; separate Python 3.12 environment (`data/pretraining/requirements.txt`) |
| `data/sft/` | SFT data selection, package `sft_data` (`PYTHONPATH=data/sft python -m sft_data`) |
| `data/tokenizer/` | tokenizer training code (being added) |
| `examples/` | pretraining caller, JSON configs and `configs/env.example`; a runnable SFT smoke example; the Kefir 1 SFT recipe as a record |
| `results/` | measured data, `make_figures.py`, figures |
| `tests/` | CPU-only suite with equivalence goldens |
| `docs/` | method, training, evaluation, provenance, architecture record |

## One command line

Every user-facing action is a subcommand of `kefir` (`src/kefir/cli.py`): `pretrain`, `sft {render,train}`,
`evaluate {text,chat}`, `generate`, `chat`, `export`, `bench build`. Add behaviour to an existing path rather than a
new entry point or script; each input is normalized once at entry (see the entries table in the architecture
record).

## Tests

```bash
uv run pytest -q
# or, without uv:
PYTHONPATH=src python -m pytest -q
```

The suite runs on CPU, FP32, and needs no network, GPU or private data. The corpus recipe has its own check,
`PYTHONPATH=src data/pretraining/.venv/bin/python data/pretraining/test_recipe.py`; the SFT data selection has
`python -m pytest data/sft -q`.

## Files that must not change

`tests/data/ran-runtime-manifest.json` pins the SHA-256 of the runtime files that produced the Kefir 1 weights
(90 files under `src/kefir`, the pairwise GDN-2 backend `src/kefir/layers/gdn2_pairwise.py` and Renderer C
`src/kefir/sft/renderer.py`), and declares the few files changed for this tree. Tests fail if a pinned file
changes. Do not edit, reformat or lint those files, even comments: their byte identity is the evidence that this is
the code that ran. A change of behaviour there means a new run with its own source identity, recorded in the
architecture record. The same applies to `filters_v4.py` and the hashed corpus sources listed in
`data/pretraining/assets/manifest.json`, and to the frozen benchmark scoring files.

## Invariants (summary; the architecture record is authoritative)

- The model is the accepted 64-expert, top-2 configuration; 8,000 semantic IDs, embedding rows 8,000–8,191 are
  padding; EOD is 7992; chat controls are 7993 (message start) and 7994 (message end).
- One tokenizer, pinned by SHA-256 `068e20ef…`, and one renderer (Renderer C) for training, evaluation and chat.
  SFT rejects conversations over 2,048 tokens. Interactive chat fits the prompt and reply budget by dropping
  older whole exchanges when needed.
- One inference weights form: a single `.safetensors` file with `model_config` metadata. A pickle (trainer state)
  is opened only with explicit trust (`--trusted-state` or a pinned SHA-256).
- Scientific configuration (run file, SFT recipe) and execution (hardware, output, cadence) are separate files;
  resume accepts only execution changes, fork only learning-rate changes with a continuous LR.
- SFT pads every row with EOD to 2,048 positions; the loss is the global mean over the batch's assistant targets.
- One or two ranks only; `test`/`reserve` benchmark splits are not for recipe tuning.

## Style

- Python 3.12+, standard library first; dependencies are pinned in `pyproject.toml` and `uv.lock`.
- Small explicit functions; errors raised at entry with a clear message; no silent fallbacks, retries or
  truncation.
- No provider launchers, account handling, credentials or private data in this tree. Credentials come only from
  environment variables named in the execution file.
- Documentation in clear technical English; numbers in docs come from `results/` or the architecture record.
- Keep changes focused, with a test that would fail without them.
