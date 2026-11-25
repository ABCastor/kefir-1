# Using Kefir 1

Install, download the weights, chat and generate. Training: [training.md](training.md). Evaluation:
[evaluation.md](evaluation.md). Everything goes through one command, `kefir` (`python -m kefir.cli` is equivalent).

## Install

Python 3.12 or newer (3.13 was used). With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra hf      # runtime, tests and the hf download command
uv sync --locked --extra hf --extra fla  # adds the CUDA GDN-2 kernels used by pretraining on GPU
```

or with pip: `pip install -e ".[hf]"` (the `hf` extra provides the `hf` command used below to download weights;
`".[hf,fla]"` adds the CUDA kernels). Chat, generation and SFT evaluation run on CPU with the pure-PyTorch GDN-2
backend and do not need `fla`. `kefir bench build` reads two Parquet files and needs the `bench` extra (`pyarrow`):
`uv sync --locked --extra bench` or `pip install -e ".[bench]"`.

Hardware: pretraining runs in BF16 on CUDA GPUs with the `fla-core` Triton kernels (an H100 for run 003, 2× T4 for
the cooldown, much slower without native BF16); SFT runs in FP16 on a T4 or newer; chat, generation and evaluation
also run on a CPU.

## Chat with Kefir 1

Download the weights once, then run chat locally on CPU. Chat makes no network requests and saves no transcript.
Activate the installed environment first (`source .venv/bin/activate` with uv):

```bash
hf download abcastor/kefir-1 model.safetensors --local-dir weights/kefir-1
echo "5d5bcb0d13b1df69c2625f336e724a4387126c2cef39e6d16a6aded2912758fe  weights/kefir-1/model.safetensors" | sha256sum -c
kefir chat --weights weights/kefir-1/model.safetensors
kefir chat --weights weights/kefir-1/model.safetensors --prompt "What is the capital of Japan?"
```

A reply to the last command looks like `Assistant> The capital of Japan is Tokyo.` Most harder questions get a
fluent wrong answer.

`/reset` clears the conversation, `/system` shows the System prompt, `/exit` quits. Decoding is greedy by default
(`--temperature 0`); `--temperature 0.7 --top-p 0.95 --seed 42` samples. `--sha256` checks the file before loading.

Other options: `--device cpu|cuda` (default `cpu`; on CUDA the model runs under FP16 autocast, elsewhere in FP32),
`--threads N` (CPU threads, default 1), `--max-new-tokens N` (reply cap, default 128), `--no-stream` (print the
reply only when it is complete), `--system-prompt TEXT` (an empty string removes it) and `--trusted-state` (load a
trainer state you trust instead of `.safetensors`).

## Continue raw text with Kefir 1 Base

```bash
hf download abcastor/kefir-1-base model.safetensors --local-dir weights/kefir-1-base
echo "11ca59f45f8ebcfd019ed5a027198d6adb549502e3d118a031f4bec8931888ac  weights/kefir-1-base/model.safetensors" | sha256sum -c
kefir generate --weights weights/kefir-1-base/model.safetensors --prompt "Once upon a time"
kefir generate --weights weights/kefir-1-base/model.safetensors --interactive
```

`--prompt` can be repeated (one continuation per prompt). Options: `--max-new-tokens N` (default 128), `--device`
(default `cpu`), `--precision fp32|fp16|bf16` (default `fp32`; FP16 and BF16 need CUDA), `--temperature` (default 0,
greedy), `--top-k` (default 0, off), `--top-p` (default 1.0, off) and `--seed`. `kefir generate` reads only
`.safetensors` weights.

## Tests and figures

```bash
uv run pytest -q                  # or: PYTHONPATH=src python -m pytest -q
python results/make_figures.py    # needs matplotlib and rsvg-convert
python results/make_architecture.py
python results/make_benchmarks.py
```

The figure script uses no network and is deterministic for a given matplotlib version (3.11.0 was used).
