"""MoE evaluation-loader identity tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from kefir.model.config import MoEModelConfig
from kefir.model.evaluation import normalize_moe_model_config
from kefir.model.model import MoESmallLLM


def _tiny_moe_config() -> MoEModelConfig:
    return MoEModelConfig.accepted(
        max_seq_len=8, d_model=32, n_layers=4, d_ff=64,
        n_heads=2, head_dim=16, gdn_num_key_heads=2, gdn_num_value_heads=2,
        gdn_key_dim=16, gdn_value_dim=16, gdn_chunk_size=4,
        num_experts=4, expert_d_ff=32,
    )


class TestMoEEvaluation(unittest.TestCase):
    def test_checkpoint_model_config_round_trips_exactly(self) -> None:
        original = _tiny_moe_config()
        payload = original.as_dict()
        # JSON checkpoint metadata may materialize tuples as lists.
        payload["moe_layer_indices"] = list(payload["moe_layer_indices"])
        dense = dict(payload["dense"])
        dense["layer_pattern"] = list(dense["layer_pattern"])
        payload["dense"] = dense

        restored = normalize_moe_model_config(payload)
        self.assertEqual(restored.as_dict(), original.as_dict())

    def test_eval_loader_rejects_dense_only_config(self) -> None:
        with self.assertRaises(RuntimeError):
            normalize_moe_model_config({"d_model": 512, "n_layers": 20})

    def test_eval_loader_fails_closed_on_topk_mismatch(self) -> None:
        payload = _tiny_moe_config().as_dict()
        payload["top_k"] = 1
        with self.assertRaises(ValueError):
            normalize_moe_model_config(payload)


if __name__ == "__main__":
    unittest.main()
