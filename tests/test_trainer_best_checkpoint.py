"""Tests for validation-loss-based remote best-checkpoint selection."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kefir.trainer.cli import _existing_remote_best_metric, _validation_metric, main


class _Store:
    def __init__(self, pointer=None):
        self.pointer = pointer
        self.paths: list[str] = []
        self.writes: list[tuple[str, dict[str, object]]] = []

    def read_json(self, path: str):
        self.paths.append(path)
        return self.pointer

    def write_json(self, path: str, value):
        self.writes.append((path, dict(value)))


class _Metrics:
    def as_dict(self):
        return {"step": 1, "loss": 3.0}


class _Engine:
    global_step = 0

    def evaluate(self, batches, *, maximum_batches=None):
        list(batches)
        return {
            "loss": 2.0,
            "perplexity": 7.0,
            "target_tokens": 4,
            "blocks": maximum_batches or 1,
        }


class _Session:
    def __init__(self, engine):
        self.engine = engine
        self.saved: list[str] = []
        self.saved_metrics: dict[str, object] = {}

    def step(self):
        self.engine.global_step += 1
        return _Metrics()

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved.append(checkpoint_id)
        self.saved_metrics[checkpoint_id] = kwargs.get("validation_metrics")
        return f"/tmp/{checkpoint_id}"


class _NonPrimarySession(_Session):
    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved.append(checkpoint_id)
        return None


class _ValidationReader:
    def iter_from_start(self, maximum_blocks=None):
        return iter((object(),))


class _Coordinator:
    def publish(self, publisher, *, checkpoint_id: str, drive_manifest, **kwargs):
        return {
            "latest": {
                "checkpoint_id": checkpoint_id,
                "last_prefix": f"run/pretrain-run/checkpoints/{checkpoint_id}/last",
                "checkpoint_manifest": {"version": 1, "files": []},
            },
            "best_updated": False,
        }


class TrainerBestCheckpointTests(unittest.TestCase):
    def test_completed_resume_does_not_attach_new_validation_to_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_id = "step-00000003"
            checkpoint = root / checkpoint_id
            checkpoint.mkdir()
            manifest = checkpoint / "checkpoint.json"
            manifest.write_text('{"validation_metrics": null}')
            original = manifest.read_bytes()
            engine = _Engine()
            engine.global_step = 3
            session = _Session(engine)
            config = SimpleNamespace(evaluation_every_steps=2, checkpoint_every_steps=3)
            args = SimpleNamespace(steps=1, validation_blocks=1, resume=checkpoint_id,
                                   checkpoint_dir=root, best_model_repo="owner/best",
                                   wandb_run_id="pretrain-run")
            with (
                patch("kefir.trainer.cli.parse_args", return_value=args),
                patch("kefir.trainer.cli.setup", return_value=(object(), config, engine, session, _Coordinator())),
                patch.object(session, "step", side_effect=StopIteration),
                patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
                patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
                patch("kefir.trainer.cli.get_dedicated_best_metric", return_value=None),
                patch("kefir.trainer.cli.publish_dedicated_best_model") as publish,
                patch("kefir.trainer.cli.configure_wandb", return_value=None),
                patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(main([]), 0)
            self.assertIn('"validation"', output.getvalue())
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(session.saved, [])
            publish.assert_not_called()

    def test_local_retention_saves_best_between_checkpoints_and_preserves_it_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for resumed in (False, True):
                with self.subTest(resumed=resumed):
                    engine = _Engine()
                    session = _Session(engine)
                    config = SimpleNamespace(evaluation_every_steps=2, checkpoint_every_steps=3)
                    args = SimpleNamespace(steps=5, validation_blocks=1, resume=None,
                                           checkpoint_dir=root, keep_last_checkpoints=1)
                    if resumed:
                        (root / "best_checkpoint.json").write_text(json.dumps({
                            "checkpoint_id": "step-00000000", "metric": -0.5,
                        }))
                    with (
                        patch("kefir.trainer.cli.parse_args", return_value=args),
                        patch("kefir.trainer.cli.setup", return_value=(object(), config, engine, session, _Coordinator())),
                        patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
                        patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
                        patch("kefir.trainer.cli.configure_wandb", return_value=None),
                        patch("kefir.trainer.cli.prune_checkpoints", return_value=[]),
                        patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.assertEqual(main([]), 0)
                    pointer = json.loads((root / "best_checkpoint.json").read_text())
                    self.assertEqual(pointer["checkpoint_id"], "step-00000000" if resumed else "step-00000002")
                    if resumed:
                        self.assertNotIn("step-00000002", session.saved)
                    else:
                        self.assertEqual(session.saved_metrics["step-00000002"]["loss"], 2.0)

    def test_different_cadences_do_not_label_a_later_checkpoint_with_stale_validation(self) -> None:
        store = _Store()
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=store),
            drive_manifest={"run_id": "pretrain-run"}, every_steps=3,
        )
        engine = _Engine()
        session = _Session(engine)
        config = SimpleNamespace(evaluation_every_steps=2, checkpoint_every_steps=3)
        args = SimpleNamespace(steps=7, validation_blocks=1, resume=None,
                               checkpoint_dir=Path("/tmp"))
        with (
            patch("kefir.trainer.cli.parse_args", return_value=args),
            patch("kefir.trainer.cli.setup", return_value=(object(), config, engine, session, _Coordinator())),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication", return_value=remote),
            patch("kefir.trainer.cli.configure_wandb", return_value=None),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)
        self.assertIsNone(session.saved_metrics["step-00000003"])
        self.assertEqual(session.saved_metrics["step-00000006"]["loss"], 2.0)
        self.assertEqual(session.saved_metrics["step-00000007"]["loss"], 2.0)
        self.assertEqual(len(store.writes), 1)
        self.assertEqual(store.writes[0][1]["checkpoint_id"], "step-00000006")

    def test_validation_loss_is_negated_for_higher_is_better_comparison(self) -> None:
        self.assertEqual(_validation_metric({"loss": 2.5}), -2.5)
        self.assertIsNone(_validation_metric(None))

    def test_dedicated_best_model_publishes_only_after_validation_improves(self) -> None:
        engine = _Engine()
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (
            object(),
            trainer_config,
            engine,
            session,
            _Coordinator(),
        )
        with (
            patch("kefir.trainer.cli.parse_args", return_value=args),
            patch("kefir.trainer.cli.setup", return_value=setup_result),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
            patch("kefir.trainer.cli.get_dedicated_best_metric", return_value=None),
            patch(
                "kefir.trainer.cli.publish_dedicated_best_model",
                return_value={
                    "status": "published",
                    "repo_id": args.best_model_repo,
                    "checkpoint_id": "step-00000001",
                },
            ) as publish,
            patch("kefir.trainer.cli.configure_wandb", return_value=None),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)

        publish.assert_called_once()
        kwargs = publish.call_args.kwargs
        self.assertEqual(kwargs["checkpoint_id"], "step-00000001")
        self.assertEqual(kwargs["metric"], -2.0)
        self.assertEqual(kwargs["validation_loss"], 2.0)
        self.assertTrue(kwargs["recreate"])

    def test_non_primary_distributed_rank_skips_external_publication(self) -> None:
        engine = _Engine()
        session = _NonPrimarySession(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (object(), trainer_config, engine, session, _Coordinator())
        with (
            patch("kefir.trainer.cli.parse_args", return_value=args),
            patch("kefir.trainer.cli.setup", return_value=setup_result),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication") as configure_remote,
            patch("kefir.trainer.cli.get_dedicated_best_metric") as get_best_metric,
            patch("kefir.trainer.cli.publish_dedicated_best_model") as publish,
            patch("kefir.trainer.cli.configure_wandb") as configure_wandb,
            patch("kefir.trainer.cli.torch.distributed.is_available", return_value=True),
            patch("kefir.trainer.cli.torch.distributed.is_initialized", return_value=True),
            patch("kefir.trainer.cli.torch.distributed.get_rank", return_value=1),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)

        self.assertEqual(session.saved, ["step-00000001"])
        configure_remote.assert_not_called()
        get_best_metric.assert_not_called()
        publish.assert_not_called()
        configure_wandb.assert_not_called()

    def test_dedicated_best_model_respects_persisted_historical_best(self) -> None:
        engine = _Engine()
        engine.best_validation_loss = 1.5
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (object(), trainer_config, engine, session, _Coordinator())
        with (
            patch("kefir.trainer.cli.parse_args", return_value=args),
            patch("kefir.trainer.cli.setup", return_value=setup_result),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
            patch("kefir.trainer.cli.get_dedicated_best_metric", return_value=None),
            patch("kefir.trainer.cli.publish_dedicated_best_model") as publish,
            patch("kefir.trainer.cli.configure_wandb", return_value=None),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)
        publish.assert_not_called()

    def test_missing_best_repo_repairs_from_resumed_historical_best_checkpoint(self) -> None:
        engine = _Engine()
        engine.best_validation_loss = 2.0
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=1,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume="step-00000000",
            checkpoint_dir=Path("/tmp"),
            best_model_repo="owner/model-best-pretrain-run",
            best_model_recreate=True,
            remote_token_env="HF_TOKEN",
            wandb_run_id="pretrain-run",
        )
        setup_result = (object(), trainer_config, engine, session, _Coordinator())
        with (
            patch("kefir.trainer.cli.parse_args", return_value=args),
            patch("kefir.trainer.cli.setup", return_value=setup_result),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
            patch("kefir.trainer.cli.get_dedicated_best_metric", return_value=None),
            patch("kefir.trainer.cli.checkpoint_validation_loss", return_value=2.0),
            patch(
                "kefir.trainer.cli.publish_dedicated_best_model",
                return_value={"status": "published", "checkpoint_id": args.resume},
            ) as publish,
            patch("kefir.trainer.cli.configure_wandb", return_value=None),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)
        publish.assert_called_once()
        self.assertEqual(publish.call_args.kwargs["checkpoint_id"], args.resume)
        self.assertEqual(publish.call_args.kwargs["validation_loss"], 2.0)

    def test_existing_best_metric_is_read_on_resume(self) -> None:
        store = _Store({"metric": -1.75})
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=store),
            drive_manifest={"run_id": "pretrain-run"},
        )
        self.assertEqual(_existing_remote_best_metric(remote), -1.75)
        self.assertEqual(store.paths, ["run/pretrain-run/best.json"])

    def test_invalid_remote_metric_fails_closed(self) -> None:
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=_Store({"metric": "bad"})),
            drive_manifest={"run_id": "pretrain-run"},
        )
        with self.assertRaisesRegex(RuntimeError, "numeric metric"):
            _existing_remote_best_metric(remote)

    def test_training_writes_pointer_only_best_after_verified_latest(self) -> None:
        store = _Store()
        remote = SimpleNamespace(
            publisher=SimpleNamespace(store=store),
            drive_manifest={"run_id": "pretrain-run"},
            every_steps=1,
        )
        engine = _Engine()
        session = _Session(engine)
        trainer_config = SimpleNamespace(
            evaluation_every_steps=1,
            checkpoint_every_steps=0,
        )
        args = SimpleNamespace(
            steps=1,
            validation_blocks=1,
            resume=None,
            checkpoint_dir=Path("/tmp"),
        )
        setup_result = (
            object(),
            trainer_config,
            engine,
            session,
            _Coordinator(),
        )
        with (
            patch("kefir.trainer.cli.parse_args", return_value=args),
            patch("kefir.trainer.cli.setup", return_value=setup_result),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication", return_value=remote),
            patch("kefir.trainer.cli.configure_wandb", return_value=None),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main([]), 0)

        self.assertEqual(session.saved, ["step-00000001"])
        self.assertEqual(len(store.writes), 1)
        path, pointer = store.writes[0]
        self.assertEqual(path, "run/pretrain-run/best.json")
        self.assertEqual(pointer["metric"], -2.0)
        self.assertEqual(
            pointer["best_prefix"],
            "run/pretrain-run/checkpoints/step-00000001/last",
        )
        self.assertEqual(pointer["checkpoint_manifest"], {"version": 1, "files": []})


if __name__ == "__main__":
    unittest.main()
