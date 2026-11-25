"""Normalize a serialized MoE model configuration (checkpoint or weights metadata)."""

from __future__ import annotations

from typing import Mapping

from kefir.layers.config import ModelConfig

from .config import MoEModelConfig


def normalize_moe_model_config(raw: Mapping[str, object]) -> MoEModelConfig:
    values = dict(raw)
    dense_raw = values.pop("dense", None)
    if not isinstance(dense_raw, Mapping):
        raise RuntimeError("MoE model config must contain a dense ModelConfig mapping")
    dense_values = dict(dense_raw)
    pattern = dense_values.get("layer_pattern")
    if isinstance(pattern, list):
        dense_values["layer_pattern"] = tuple(pattern)
    layer_indices = values.get("moe_layer_indices")
    if isinstance(layer_indices, list):
        values["moe_layer_indices"] = tuple(int(index) for index in layer_indices)
    dense = ModelConfig(**dense_values)  # type: ignore[arg-type]
    return MoEModelConfig(dense=dense, **values)  # type: ignore[arg-type]


__all__ = ["normalize_moe_model_config"]
