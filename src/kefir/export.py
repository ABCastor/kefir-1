"""Explicit trusted conversion from a trainer state to the one inference weights form."""

from __future__ import annotations

from pathlib import Path

import torch

from .artifact import load_weights, read_weights, save_weights


def export_checkpoint(state_file: str | Path, destination: str | Path, *,
                      trusted: bool = False) -> Path:
    """Convert your own trainer state. Pickle loading can execute code: trust is mandatory.

    Accepts a pretraining ``trainer_state.pkl`` or an SFT ``trainer_state.pt``; writes a
    ``.safetensors`` file with ``model_config`` metadata, then reads it back and requires
    tensor-for-tensor equality.
    """
    if not trusted:
        raise ValueError("export requires trusted=True: trainer states can execute arbitrary code")
    config, tensors, info = read_weights(state_file, trusted=True)
    from kefir.model.evaluation import normalize_moe_model_config
    from kefir.model.model import MoESmallLLM
    model = MoESmallLLM(normalize_moe_model_config(config))
    model.load_state_dict(tensors, strict=True)
    path = save_weights(model, destination, metadata={
        "source_trainer_state_sha256": info["sha256"], "source_global_step": info["global_step"]})
    loaded, _, _ = load_weights(path)
    expected = model.state_dict()
    actual = loaded.state_dict()
    if set(actual) != set(expected) or any(not torch.equal(actual[k], expected[k]) for k in expected):
        path.unlink()
        raise RuntimeError("exported weights read back differently")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="kefir export",
                                     description="Export a trusted trainer state as safe inference weights")
    parser.add_argument("--trusted-state", required=True,
                        help="trainer_state.pkl/.pt you trust to execute code during conversion")
    parser.add_argument("--output", required=True, help="new .safetensors file")
    args = parser.parse_args(argv)
    print(export_checkpoint(args.trusted_state, args.output, trusted=True))
    return 0
