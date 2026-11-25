"""Two CPU ranks prove the public MoE DDP update matches one logical block."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kefir.model.distributed import prepare_distributed_engine
from kefir.model.engine import MoETrainerEngine
from kefir.model.model import MoESmallLLM
from kefir.trainer.config import TrainerConfig
from kefir.trainer.distributed import DistributedContext
from kefir.trainer.state import _cuda_rng_states, _restore_cuda_rng, load_trainer_state_file
from kefir.trainer.types import TokenBatch


from tests.moe_fixtures import tiny_training_config as _model_config


def _engine(*, microbatch_size: int = 1, precision: str = "fp32") -> MoETrainerEngine:
    torch.manual_seed(271)
    model = MoESmallLLM(_model_config())
    return MoETrainerEngine(
        model,
        TrainerConfig(
            optimizer="hybrid_muon_adamw", precision=precision, microbatch_size=microbatch_size,
            weight_decay=0.0, muon_weight_decay=0.0, max_grad_norm=100.0,
        ),
        device="cpu",
    )


def _batch(block_id: int = 0, rows: int = 4) -> TokenBatch:
    ids = torch.tensor([
        [1, 4, 5, 9], [2, 3, 8, 1], [7, 6, 2, 4], [5, 2, 3, 9],
    ], dtype=torch.long)
    labels = torch.tensor([
        [4, 5, 9, 1], [3, 8, 1, 2], [6, 2, 4, 7], [2, 3, 9, 5],
    ], dtype=torch.long)
    labels[-1, -1] = -100
    ids, labels = ids[:rows], labels[:rows]
    return TokenBatch(block_id, "train", ids, labels, rows, int(labels.ne(-100).sum()))


def _record_scaled_objectives(engine) -> list[tuple[float, float]]:
    engine.scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    scale = engine.scaler.scale
    recorded = []

    def record(objective):
        scaled = scale(objective)
        recorded.append((float(objective.detach()), float(scaled.detach())))
        return scaled

    engine.scaler.scale = record
    return recorded


def _worker(rank: int, root: str, mode: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/group", rank=rank, world_size=2
    )
    try:
        context = DistributedContext(rank, rank, 2)
        engine = _engine(microbatch_size=2 if mode.startswith("tail") else 1,
                         precision="bf16" if mode == "autocast" else "fp32")
        engine.distributed = context
        initial = Path(root) / "initial.pt"
        if initial.exists():
            engine.load_state_dict(torch.load(initial, weights_only=False))
        objectives = _record_scaled_objectives(engine) if mode == "backward_scale" else []
        prepare_distributed_engine(engine, context)
        if mode == "overflow" and rank == 1:
            forward = engine.model.module.forward

            def bad_forward(inputs, labels):
                ce, z, layers = forward(inputs, labels)
                return ce * float("nan"), z, layers

            engine.model.module.forward = bad_forward
        if mode in {"overflow", "gradient_overflow"}:
            gradient_failure = (
                patch("kefir.model.distributed.torch.nn.utils.clip_grad_norm_",
                      return_value=torch.tensor(float("inf")))
                if mode == "gradient_overflow" and rank == 1 else nullcontext()
            )
            with gradient_failure:
                try:
                    engine.train_batch(_batch())
                except FloatingPointError:
                    failed = True
                else:
                    failed = False
            payload = {
                "failed": failed,
                "step": engine.global_step,
                "tokens": engine.consumed_tokens,
                "drain": context.vote(rank == 1, engine.device),
            }
        else:
            backward = torch.Tensor.backward

            def outside_autocast(tensor, *args, **kwargs):
                assert not torch.is_autocast_enabled("cpu"), "backward must run outside autocast"
                return backward(tensor, *args, **kwargs)

            rows = int(mode[-1]) if mode.startswith("tail") else 4
            with patch.object(torch.Tensor, "backward", outside_autocast):
                metrics = engine.train_batch(_batch(engine.global_step, rows=rows))
            payload = {
                "step": engine.global_step,
                "tokens": engine.consumed_tokens,
                "loss": metrics.loss,
                "gradients": {
                    name: parameter.grad.detach().clone()
                    for name, parameter in engine.raw_model.named_parameters()
                    if parameter.grad is not None
                },
                "model": engine.raw_model.state_dict(),
                "optimizer": engine.optimizer.state_dict(),
                "router": metrics.moe,
                "scaled_objectives": objectives,
            }
            if rank == 0:
                payload["checkpoint"] = engine.state_dict()
                engine.save_checkpoint_state(Path(root) / "streamed.pt")
        torch.save(payload, Path(root) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _finalization_worker(rank: int, root: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/final-group", rank=rank, world_size=2
    )
    try:
        from kefir.trainer.cli import main

        context = DistributedContext(rank, rank, 2)
        engine = SimpleNamespace(
            distributed=context, device=torch.device("cpu"), global_step=0,
            consumed_tokens=0,
        )

        class Session:
            def step(self):
                raise StopIteration

            def save_checkpoint(self, *args, **kwargs):
                raise RuntimeError("synthetic final save failed")

        setup = lambda args: (
            object(), SimpleNamespace(evaluation_every_steps=0, checkpoint_every_steps=0),
            engine, Session(), object(),
        )
        argv = ["--dataset-dir", root, "--checkpoint-dir", root, "--steps", "1"]
        with patch("kefir.trainer.cli.configure_remote_publication", return_value=None), \
             patch("kefir.trainer.cli.configure_wandb", return_value=None):
            try:
                main(argv, setup_fn=setup)
            except RuntimeError as error:
                result = str(error)
            else:
                result = "no error"
        (Path(root) / f"final-rank{rank}.txt").write_text(result)
    finally:
        dist.destroy_process_group()


def _periodic_failure_worker(rank: int, root: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/periodic-group", rank=rank, world_size=2
    )
    try:
        from kefir.trainer.cli import main

        engine = SimpleNamespace(
            distributed=DistributedContext(rank, rank, 2), device=torch.device("cpu"),
            global_step=0, consumed_tokens=0,
        )

        class Session:
            def step(self):
                engine.global_step += 1
                engine.consumed_tokens += 4
                return SimpleNamespace(as_dict=lambda: {"step": engine.global_step})

            def save_checkpoint(self, *args, **kwargs):
                raise RuntimeError("synthetic periodic save failed")

        setup = lambda args: (
            object(), SimpleNamespace(evaluation_every_steps=0, checkpoint_every_steps=1),
            engine, Session(), object(),
        )
        argv = ["--dataset-dir", root, "--checkpoint-dir", root, "--steps", "1"]
        with patch("kefir.trainer.cli.configure_remote_publication", return_value=None), \
             patch("kefir.trainer.cli.configure_wandb", return_value=None):
            try:
                main(argv, setup_fn=setup)
            except RuntimeError as error:
                result = str(error)
            else:
                result = "no error"
        (Path(root) / f"periodic-rank{rank}.txt").write_text(result)
    finally:
        dist.destroy_process_group()


class TestMoEDistributedCore(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.get("SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY")
        os.environ["SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY"] = "1"
        torch.set_num_threads(2)

    def tearDown(self) -> None:
        environ = os.environ
        if self._env is None:
            environ.pop("SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY", None)
        else:
            environ["SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY"] = self._env

    def _run(self, root: Path, mode: str) -> tuple[dict, dict]:
        mp.spawn(_worker, args=(str(root), mode), nprocs=2, join=True)
        return tuple(
            torch.load(root / f"rank{rank}.pt", weights_only=False) for rank in (0, 1)
        )

    def test_one_and_two_ranks_commit_the_same_batch_and_resume(self) -> None:
        single = _engine()
        first = single.train_batch(_batch())
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in single.raw_model.named_parameters()
            if parameter.grad is not None
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank0, rank1 = self._run(root, "normal")
            self.assertEqual((rank0["step"], rank0["tokens"]), (1, 15))
            self.assertEqual((rank1["step"], rank1["tokens"]), (1, 15))
            self.assertAlmostEqual(rank0["loss"], first.loss, places=5)
            self.assertAlmostEqual(rank1["loss"], first.loss, places=5)
            for name, expected in gradients.items():
                torch.testing.assert_close(rank0["gradients"][name], expected, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(rank1["gradients"][name], expected, rtol=2e-4, atol=2e-5)
            for name, expected in single.raw_model.state_dict().items():
                torch.testing.assert_close(rank0["model"][name], expected, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(rank1["model"][name], expected, rtol=2e-4, atol=2e-5)
            self.assertEqual(rank0["router"]["routing"]["global_expert_counts"],
                             first.moe["routing"]["global_expert_counts"])
            self.assertEqual(rank1["router"], {})
            self.assertTrue(zipfile.is_zipfile(root / "streamed.pt"))
            streamed = load_trainer_state_file(root / "streamed.pt")
            self.assertEqual((streamed["global_step"], streamed["consumed_tokens"]), (1, 15))
            self.assertEqual(set(streamed["model"]), set(single.raw_model.state_dict()))
            torch.save(rank0["checkpoint"], root / "initial.pt")
            second = single.train_batch(_batch(1))
            rank0, rank1 = self._run(root, "normal")
            self.assertEqual((rank0["step"], rank0["tokens"]), (2, 30))
            self.assertAlmostEqual(rank0["loss"], second.loss, places=5)
            resumed = _engine()
            resumed.load_state_dict(rank0["checkpoint"])
            self.assertEqual((resumed.global_step, resumed.consumed_tokens), (2, 30))
            for name, expected in single.raw_model.state_dict().items():
                torch.testing.assert_close(resumed.raw_model.state_dict()[name], expected,
                                           rtol=2e-4, atol=2e-5)

    def test_grad_scaler_receives_the_same_microbatch_objective_on_one_or_two_ranks(self) -> None:
        single = _engine()
        objectives = _record_scaled_objectives(single)
        single.train_batch(_batch())
        with tempfile.TemporaryDirectory() as directory:
            ranks = self._run(Path(directory), "backward_scale")
        self.assertEqual(len(objectives), 4)
        for rank, result in enumerate(ranks):
            expected = torch.tensor(objectives[2 * rank:2 * rank + 2])
            observed = torch.tensor(result["scaled_objectives"])
            # DDP averaging must be compensated after unscale, so FP16 backward
            # sees the same magnitudes as the corresponding single-rank rows.
            torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(observed[:, 1], observed[:, 0] * 128.0)
            for name, parameter in single.raw_model.named_parameters():
                torch.testing.assert_close(result["gradients"][name], parameter.grad,
                                           rtol=2e-4, atol=2e-5)

    def test_one_rank_forward_failure_aborts_both_and_drain_reaches_both(self) -> None:
        for mode in ("overflow", "gradient_overflow"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                rank0, rank1 = self._run(Path(directory), mode)
            for result in (rank0, rank1):
                self.assertTrue(result["failed"])
                self.assertTrue(result["drain"])
                self.assertEqual((result["step"], result["tokens"]), (0, 0))

    def test_partial_blocks_match_single_rank_without_losing_rows(self) -> None:
        for rows in (1, 2, 3):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                single = _engine(microbatch_size=2)
                expected = single.train_batch(_batch(rows=rows))
                rank0, rank1 = self._run(Path(directory), f"tail{rows}")
                for result in (rank0, rank1):
                    self.assertEqual((result["step"], result["tokens"]), (1, rows * 4))
                    self.assertAlmostEqual(result["loss"], expected.loss, places=5)
                    for name, parameter in single.raw_model.named_parameters():
                        torch.testing.assert_close(result["gradients"][name], parameter.grad,
                                                   rtol=2e-4, atol=2e-5)
                    for name, value in single.raw_model.state_dict().items():
                        torch.testing.assert_close(result["model"][name], value,
                                                   rtol=2e-4, atol=2e-5)
                self.assertEqual(rank0["router"]["routing"]["global_expert_counts"],
                                 expected.moe["routing"]["global_expert_counts"])

    def test_backward_runs_outside_cpu_bf16_autocast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rank0, rank1 = self._run(Path(directory), "autocast")
        self.assertEqual(rank0["tokens"], 15)
        self.assertEqual(rank1["tokens"], 15)

    def test_cuda_rng_policy_is_local_and_rejects_divergent_rank_states(self) -> None:
        engine = _engine()
        engine.device = torch.device("cuda:1")
        engine.distributed = DistributedContext(1, 1, 2)
        rng = torch.tensor([7, 8, 9], dtype=torch.uint8)
        with patch("kefir.trainer.state.torch.cuda.is_available", return_value=True), \
             patch("kefir.trainer.state.torch.cuda.get_rng_state", return_value=rng), \
             patch("kefir.trainer.state.torch.cuda.set_rng_state") as set_local:
            saved = _cuda_rng_states(engine)
            self.assertEqual(len(saved), 2)
            self.assertTrue(torch.equal(saved[0], saved[1]))
            _restore_cuda_rng(engine, [rng])
            set_local.assert_called_once()
            self.assertEqual(set_local.call_args.kwargs["device"], torch.device("cuda:1"))
            set_local.reset_mock()
            engine.distributed = None
            engine.device = torch.device("cuda:0")
            _restore_cuda_rng(engine, saved)
            self.assertEqual(set_local.call_args.kwargs["device"], torch.device("cuda:0"))
            with self.assertRaisesRegex(ValueError, "identical"):
                _restore_cuda_rng(engine, [rng, rng + 1])

    def test_final_checkpoint_failure_reaches_the_waiting_follower(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_finalization_worker, args=(directory,), nprocs=2, join=True)
            self.assertIn("synthetic final save failed",
                          (Path(directory) / "final-rank0.txt").read_text())
            self.assertIn("distributed finalization failed",
                          (Path(directory) / "final-rank1.txt").read_text())

    def test_periodic_checkpoint_failure_reaches_the_waiting_follower(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_periodic_failure_worker, args=(directory,), nprocs=2, join=True)
            self.assertIn("synthetic periodic save failed",
                          (Path(directory) / "periodic-rank0.txt").read_text())
            self.assertIn("distributed checkpoint or publication failed",
                          (Path(directory) / "periodic-rank1.txt").read_text())


if __name__ == "__main__":
    unittest.main()
