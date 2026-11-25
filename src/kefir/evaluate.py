"""Evaluate a saved Kefir model on held-out data and frozen thematic panels."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from kefir.trainer.evaluation import evaluate_batches
from kefir.trainer.shards import SchemaV2ShardReader
from kefir.trainer.shard_config import load_manifest
from kefir.trainer.thematic_validation import evaluate_thematic, load_thematic_panels

from .artifact import load_weights
from .tokenizer import TOKENIZER_SHA256


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    *,
    device: str = "cpu",
    precision: str = "fp32",
    validation_blocks: int = 1,
    microbatch_size: int = 1,
    thematic_panels: str | Path | None = None,
) -> dict[str, object]:
    if validation_blocks <= 0:
        raise ValueError("validation_blocks must be positive")
    root = Path(checkpoint)
    target = torch.device(device)
    model, config, _ = load_weights(root, device=target)
    manifest = load_manifest(Path(dataset_dir) / "manifest.json")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("held-out dataset uses a different or unknown tokenizer")
    reader = SchemaV2ShardReader(
        dataset_dir, split="validation", semantic_vocab_size=config.semantic_vocab_size,
        context_length=config.max_seq_len,
    )
    engine = SimpleNamespace(model=model, raw_model=model, device=target,
                             config=SimpleNamespace(precision=precision),
                             best_validation_loss=None)
    result: dict[str, object] = {
        "held_out": evaluate_batches(
            engine, reader.iter_from_start(validation_blocks),
            maximum_batches=validation_blocks, microbatch_size=microbatch_size,
        )
    }
    if thematic_panels is not None:
        panels = load_thematic_panels(thematic_panels)
        if panels.tokenizer_sha256 != TOKENIZER_SHA256:
            raise ValueError("thematic panels use a different tokenizer")
        result["thematic"] = evaluate_thematic(
            model, device=target, microbatch=microbatch_size, panels=panels,
            precision=precision, semantic_vocab_size=config.semantic_vocab_size,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kefir evaluate text", description="Held-out text NLL and thematic panels for Kefir 1 weights")
    parser.add_argument("--weights", required=True, help="Kefir .safetensors weights file")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--validation-blocks", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--thematic-panels", nargs="?", const=str(files("kefir.assets").joinpath("thematic_panels_v1")))
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.weights, args.dataset_dir, device=args.device, precision=args.precision,
        validation_blocks=args.validation_blocks, microbatch_size=args.microbatch_size,
        thematic_panels=args.thematic_panels,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
