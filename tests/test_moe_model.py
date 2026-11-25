"""CPU-unit contracts for the Kefir Top-2 MoE architecture."""

from __future__ import annotations

import unittest

import torch

from kefir.layers.config import ModelConfig
from kefir.model.accounting import count_moe_parameters
from kefir.model.config import MoEModelConfig
from kefir.model.model import DroplessTopKMoE, MoESmallLLM
from kefir.model.optimizer import classify_moe_parameters
from kefir.model.router import SwitchTopKRouter


def _tiny_moe_config() -> MoEModelConfig:
    dense = ModelConfig(
        semantic_vocab_size=32,
        padded_vocab_size=32,
        max_seq_len=8,
        d_model=64,
        n_layers=4,
        d_ff=96,
        n_heads=4,
        head_dim=16,
        gdn_num_key_heads=4,
        gdn_num_value_heads=4,
        gdn_key_dim=16,
        gdn_value_dim=16,
        gdn_conv_kernel_size=4,
        gdn_chunk_size=4,
    )
    return MoEModelConfig(
        dense=dense,
        num_experts=8, top_k=2, expert_d_ff=96,
        router_scoring="sqrt_softplus", router_z_loss_coefficient=0.0,
        load_balancing="quantile",
        moe_layer_indices=tuple(range(4)),
    )


class TestMoEModel(unittest.TestCase):
    def test_router_is_random_bias_free_top2_and_differentiable(self) -> None:
        torch.manual_seed(17)
        router = SwitchTopKRouter(16, 8, init_std=0.01, top_k=2, scoring="sqrt_softplus")
        self.assertIsNone(router.projection.bias)
        self.assertGreater(float(router.projection.weight.std()), 0.0)

        x = torch.randn(64, 16, requires_grad=True)
        first = router(x)
        second = router(x)
        self.assertTrue(torch.equal(first.expert_indices, second.expert_indices))
        self.assertTrue(
            torch.equal(first.selected_probabilities, second.selected_probabilities)
        )
        self.assertEqual(first.expert_indices.shape, (64, 2))
        self.assertEqual(first.selected_probabilities.shape, (64, 2))
        self.assertEqual(int(first.expert_counts.sum()), 128)

        loss = first.selected_probabilities[:, 0].mean()
        loss.backward()
        self.assertIsNotNone(router.projection.weight.grad)
        self.assertTrue(bool(torch.isfinite(router.projection.weight.grad).all()))

    def test_dropless_moe_executes_exactly_two_experts_per_token(self) -> None:
        torch.manual_seed(3)
        config = _tiny_moe_config()
        moe = DroplessTopKMoE(config)
        x = torch.randn(2, 7, config.d_model, requires_grad=True)
        y, z_loss, telemetry = moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(int(telemetry.expert_counts.sum()), 28)
        self.assertEqual(telemetry.token_count, 14)
        self.assertTrue(bool(torch.isfinite(z_loss)))

    def test_only_selected_expert_gets_token_path_gradient(self) -> None:
        torch.manual_seed(5)
        config = _tiny_moe_config()
        moe = DroplessTopKMoE(config)
        with torch.no_grad():
            moe.router.projection.weight.zero_()
            moe.router.projection.weight[0].fill_(0.1)
            moe.router.projection.weight[1].fill_(0.05)
        x = torch.ones(2, 5, config.d_model, requires_grad=True)
        y, z_loss, telemetry = moe(x)
        self.assertEqual(int(telemetry.expert_counts[0]), 10)
        self.assertEqual(int(telemetry.expert_counts[1]), 10)
        self.assertEqual(int(telemetry.expert_counts[2:].sum()), 0)

        (y.square().mean() + 1e-4 * z_loss).backward()
        self.assertIsNotNone(moe.router.projection.weight.grad)
        for weight in (moe.gate_weight, moe.up_weight, moe.down_weight):
            self.assertGreater(float(weight.grad[:2].abs().sum()), 0)
            self.assertEqual(float(weight.grad[2:].abs().sum()), 0)

    def test_config_fails_closed_on_old_identity_or_drops(self) -> None:
        config = _tiny_moe_config()
        dense = config.dense
        common = {
            "dense": dense,
            "expert_d_ff": dense.d_ff,
            "moe_layer_indices": tuple(range(dense.n_layers)),
        }
        with self.assertRaises(ValueError):
            MoEModelConfig(**common, version=2)
        with self.assertRaises(ValueError):
            MoEModelConfig(**common, load_balancing="none", capacity_factor=1.5)
        with self.assertRaises(ValueError):
            MoEModelConfig(**common, dropped_tokens_allowed=True)

    def test_moe_optimizer_routes_experts_to_muon_and_router_to_adamw(self) -> None:
        config = _tiny_moe_config()

        wrapper = MoESmallLLM(config)
        routing = classify_moe_parameters(wrapper).routing
        self.assertTrue(
            "blocks.0.ffn.gate_weight" in routing.muon
        )
        self.assertIn("blocks.0.ffn.router.projection.weight", routing.adamw_no_decay)
        self.assertEqual(
            set(routing.all_names),
            {name for name, parameter in wrapper.named_parameters() if parameter.requires_grad},
        )

    def test_parameter_accounting_separates_stored_and_active_experts(self) -> None:
        config = _tiny_moe_config()

        wrapper = MoESmallLLM(config)
        counts = count_moe_parameters(wrapper, num_experts=config.num_experts, top_k=config.top_k)
        self.assertGreater(counts.experts_stored, counts.experts_active_per_token)
        self.assertEqual(
            counts.experts_stored,
            counts.experts_active_per_token * config.num_experts // config.top_k,
        )
        self.assertEqual(
            counts.active_per_token,
            counts.backbone_non_expert + counts.router + counts.experts_active_per_token,
        )


if __name__ == "__main__":
    unittest.main()
