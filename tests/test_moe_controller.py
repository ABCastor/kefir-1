"""Controller, sparse AMP combine and committed-state contracts."""
from dataclasses import replace
import unittest
import copy

import torch

from kefir.model.config import MoEModelConfig
from kefir.model.engine import MoETrainerEngine
from kefir.model.model import DroplessTopKMoE, MoESmallLLM
from kefir.model.router import SwitchTopKRouter
from tests.test_moe_model import _tiny_moe_config
from kefir.trainer import TrainerConfig
from kefir.trainer.engine import TrainerEngine
from tests.trainer_fixtures import TinyLM, batch


class TestMoEController(unittest.TestCase):
    def test_real_scaler_failed_update_preserves_model_optimizer_and_controller(self):
        def same_tree(first, second):
            if isinstance(first, torch.Tensor):
                self.assertTrue(torch.equal(first, second))
            elif isinstance(first, dict):
                self.assertEqual(first.keys(), second.keys())
                for key in first:
                    same_tree(first[key], second[key])
            elif isinstance(first, (list, tuple)):
                self.assertEqual(len(first), len(second))
                for a, b in zip(first, second, strict=True):
                    same_tree(a, b)
            else:
                self.assertEqual(first, second)

        config = TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32", microbatch_size=1, max_overflow_retries=1,
                               schedule="wsd", warmup_tokens=60, stable_tokens=60, decay_tokens=60)
        for kind in ("dense", "moe"):
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                model = (TinyLM() if kind == "dense" else MoESmallLLM(_tiny_moe_config()))
                engine_cls = TrainerEngine if kind == "dense" else MoETrainerEngine
                engine = engine_cls(model, replace(config, optimizer=("adamw" if kind == "dense" else "hybrid_muon_adamw")), device="cpu")
                engine.train_batch(batch(0))
                engine.scaler = torch.amp.GradScaler("cpu", init_scale=1., enabled=True)
                before = engine.state_dict()
                for parameter in model.parameters():
                    parameter.register_hook(lambda gradient: torch.full_like(gradient, 1e20))
                with self.assertRaises(FloatingPointError):
                    engine.train_batch(batch(1))
                after = engine.state_dict()
                for key in ("model", "optimizer", "scheduler", "global_step", "consumed_tokens"):
                    same_tree(before[key], after[key])
                self.assertEqual(engine.overflow_events, 2)
                self.assertEqual(engine.scaler.get_scale(), .25)

    def test_retry_then_success_commits_controller_and_targets_once(self):
        config = _tiny_moe_config()
        engine = MoETrainerEngine(MoESmallLLM(config), TrainerConfig(
            optimizer="hybrid_muon_adamw", precision="fp32", microbatch_size=2,
            schedule="wsd", warmup_tokens=60, stable_tokens=60, decay_tokens=60), device="cpu")
        reference = MoETrainerEngine(copy.deepcopy(engine.model), engine.config, device="cpu")
        reference.train_batch(batch(0))
        engine.scaler = torch.amp.GradScaler("cpu", init_scale=1., enabled=True)
        calls = 0
        def overflow_once(gradient):
            nonlocal calls
            calls += 1
            return torch.full_like(gradient, float("inf")) if calls == 1 else gradient
        handle = next(engine.model.parameters()).register_hook(overflow_once)
        try:
            result = engine.train_batch(batch(0))
        finally:
            handle.remove()
        self.assertEqual(calls, 2)
        self.assertEqual(engine.overflow_events, 1)
        self.assertEqual(engine.global_step, 1)
        self.assertEqual(engine.consumed_tokens, 6)
        self.assertEqual(result.step, 1)
        for block in engine.model.blocks:
            router = block.ffn.router
            self.assertEqual(int(router.token_exposure.sum()), 6 * config.top_k)
            self.assertTrue(torch.equal(router.expert_updates, router.token_exposure.gt(0).long()))
        for expected, actual in zip(reference.model.blocks, engine.model.blocks, strict=True):
            torch.testing.assert_close(
                actual.ffn.router.selection_bias, expected.ffn.router.selection_bias, rtol=0, atol=0,
            )

    def test_forward_and_eval_do_not_advance_controller(self):
        router = SwitchTopKRouter(4, 8, init_std=.01, top_k=2,
                                  scoring="sqrt_softplus", balancing="quantile")
        before = {k: v.clone() for k, v in router.named_buffers()}
        for training in (True, False):
            router.train(training)
            router(torch.randn(3, 4))
        for name, value in router.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)
        router.train()
        router.begin_step(4)
        counts = router(torch.randn(4, 4)).expert_counts
        router.commit_load(counts)
        self.assertTrue(torch.isfinite(router.selection_bias).all())
        self.assertTrue(torch.equal(router.token_exposure, counts))
        self.assertTrue(torch.equal(router.expert_updates, counts.gt(0).long()))

    def test_controller_and_exposure_are_checkpoint_state(self):
        config = _tiny_moe_config()
        model = MoESmallLLM(config)
        for block in model.blocks:
            router = block.ffn.router
            router.begin_step(8)
            router.commit_load(router(torch.randn(8, config.d_model)).expert_counts)
        restored = MoESmallLLM(config)
        restored.load_state_dict(model.state_dict())
        for first, second in zip(model.buffers(), restored.buffers()):
            self.assertTrue(torch.equal(first, second))

    def test_config_rejects_misreported_identity(self):
        config = _tiny_moe_config()
        for key, value in [
            ("version", 999), ("router_scoring", "sigmoid"),
            ("router_combine", "uniform"), ("dispatch", "dense_all_experts"),
            ("expert_type", "other"), ("load_balancing", "unknown"),
            ("version", 2), ("load_balancing", "loss_free_sign"),
            ("balancing_step_size", .001), ("balancing_step_size", float("nan")),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(config, **{key: value})

    def test_mixed_precision_expert_output_combines_into_float_residual(self):
        config = _tiny_moe_config()
        moe = DroplessTopKMoE(config)
        inputs = torch.randn(2, 3, config.d_model, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, z_loss, _ = moe(inputs)
        self.assertEqual(output.dtype, torch.float32)
        (output.square().mean() + z_loss).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())


if __name__ == "__main__":
    unittest.main()
