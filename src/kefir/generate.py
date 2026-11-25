"""Generate text from a saved Kefir v1 model with its pinned tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kefir.trainer.eval_generation import GenerationRequest, sample_token_ids_batched

from .artifact import load_weights
from .tokenizer import EOD_TOKEN_ID, KefirTokenizer, load_tokenizer


class _TextVocabulary(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        # 7993..7999 are unused controls; EOD remains a valid stopping token.
        return logits[..., :EOD_TOKEN_ID + 1]


def _generate_loaded(
    model: torch.nn.Module,
    max_seq_len: int,
    tokenizer: KefirTokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    precision: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    batch_size: int,
) -> list[str]:
    requests = [GenerationRequest(tuple(tokenizer.encode(prompt)), max_new_tokens, seed + index)
                for index, prompt in enumerate(prompts)]
    generated = sample_token_ids_batched(
        _TextVocabulary(model), requests, max_seq_len=max_seq_len, eos_token_id=EOD_TOKEN_ID,
        temperature=temperature, top_p=top_p, top_k=top_k,
        precision=precision, batch_size=batch_size,
    )
    return [tokenizer.decode(ids[:-1] if ids and ids[-1] == EOD_TOKEN_ID else ids)
            for ids in generated]


def generate_text(
    checkpoint: str | Path,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    device: str = "cpu",
    precision: str = "fp32",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 17,
    batch_size: int = 8,
) -> list[str]:
    model, config, _ = load_weights(Path(checkpoint), device=torch.device(device))
    return _generate_loaded(
        model, config.max_seq_len, load_tokenizer(), prompts,
        max_new_tokens=max_new_tokens, precision=precision, temperature=temperature,
        top_p=top_p, top_k=top_k, seed=seed, batch_size=batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kefir generate", description="Continue raw text with Kefir 1 weights")
    parser.add_argument("--weights", required=True, help="Kefir .safetensors weights file")
    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--prompt", action="append", help="generate once for each prompt")
    input_mode.add_argument("--interactive", action="store_true", help="continue text entered in the terminal")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.precision != "fp32" and not args.device.startswith("cuda"):
        # Same rule as the generation core (trainer/eval_generation.py), checked before the weights load.
        parser.error(f"--precision {args.precision} needs a CUDA device; use --precision fp32 on {args.device}")
    if args.interactive:
        model, config, _ = load_weights(Path(args.weights), device=torch.device(args.device))
        tokenizer = load_tokenizer()
        while True:
            try:
                prompt = input("prompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip() in {"/quit", "/exit"}:
                break
            if not prompt.strip():
                continue
            result = _generate_loaded(
                model, config.max_seq_len, tokenizer, [prompt],
                max_new_tokens=args.max_new_tokens, precision=args.precision,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                seed=args.seed, batch_size=1,
            )[0]
            print(f"continuation> {result}" if result else "continuation> [EOD]", flush=True)
        return 0
    result = generate_text(
        args.weights, args.prompt, max_new_tokens=args.max_new_tokens,
        device=args.device, precision=args.precision, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, seed=args.seed,
    )
    print(json.dumps({"continuations": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
