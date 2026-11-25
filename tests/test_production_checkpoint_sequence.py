"""Durability contracts for the long production run: drain, retention, auto-resume."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from kefir.data.src.checkpoint_sequence import (
    complete_checkpoint,
    find_latest_complete_checkpoint,
    prune_checkpoints,
)
from kefir.data.src.joint_checkpoint import CheckpointCoordinator
from kefir.trainer.cli import main


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))




class _TrainerState:
    def __init__(self, step: int) -> None:
        self.step = step

    def state_dict(self) -> dict[str, object]:
        return {"global_step": self.step, "consumed_tokens": self.step * 1_024}

    def load_state_dict(self, state) -> None:
        self.step = int(state["global_step"])


class _StreamedTrainerState(_TrainerState):
    def save_checkpoint_state(self, path: Path) -> None:
        torch.save(self.state_dict(), path)


def _coordinator(root: Path) -> CheckpointCoordinator:
    return CheckpointCoordinator(root, configuration_hash="c", source_hash="s", schema_hash="h")


def _write_checkpoint(root: Path, step: int) -> Path:
    return _coordinator(root).save(
        checkpoint_id=f"step-{step:08d}",
        trainer=_TrainerState(step),
        pipeline_state={"last_consumed_block_id": step - 1, "gradient_accumulation_position": 0},
        optimizer_step_complete=True,
    )


def test_latest_accepts_streamed_torch_trainer_state(tmp_path: Path) -> None:
    checkpoint = _coordinator(tmp_path).save(
        checkpoint_id="step-00000001",
        trainer=_StreamedTrainerState(1),
        pipeline_state={"last_consumed_block_id": 0, "gradient_accumulation_position": 0},
        optimizer_step_complete=True,
    )
    latest = find_latest_complete_checkpoint(tmp_path)
    assert latest is not None
    assert latest["path"] == checkpoint
    assert latest["consumed_tokens"] == 1_024


def test_save_rejects_disagreeing_pipeline_cursors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cursor"):
        _coordinator(tmp_path).save(
            checkpoint_id="step-00000001", trainer=_TrainerState(1),
            pipeline_state={"last_consumed_block_id": 0,
                            "gradient_accumulation_position": 0,
                            "consumer": {"kind": "immutable_schema_v2",
                                         "last_consumed_block_id": -1}},
            optimizer_step_complete=True,
        )


def test_checksum_valid_nested_cursor_mismatch_fails_admission(tmp_path: Path) -> None:
    checkpoint = _coordinator(tmp_path).save(
        checkpoint_id="step-00000001", trainer=_TrainerState(1),
        pipeline_state={"last_consumed_block_id": 0,
                        "gradient_accumulation_position": 0,
                        "consumer": {"kind": "immutable_schema_v2",
                                     "last_consumed_block_id": 0}},
        optimizer_step_complete=True,
    )
    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["pipeline_state"]["consumer"]["last_consumed_block_id"] = -1
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = checkpoint / "local_manifest.json"
    local = json.loads(manifest_path.read_text())
    for item in local["files"]:
        if item["name"] == "checkpoint.json":
            item["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="cursor"):
        complete_checkpoint(checkpoint)


def _steps_on_disk(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir())


class _Metrics:
    def __init__(self, step: int) -> None:
        self.step = step

    def as_dict(self) -> dict[str, object]:
        return {"step": self.step, "loss": 1.0}


class _Engine:
    def __init__(self) -> None:
        self.global_step = 0

    def evaluate(self, batches, *, maximum_batches=None):
        list(batches)
        # A monotonically worsening held-out loss keeps the first update the best one.
        return {"loss": 1.0 + self.global_step, "perplexity": 2.0,
                "target_tokens": 4, "blocks": maximum_batches or 1}


class _ValidationReader:
    def iter_from_start(self, maximum_blocks=None):
        return iter((object(),))


class _Session:
    """Fake session whose step costs wall time and whose checkpoints are real."""

    def __init__(self, engine: _Engine, *, step_seconds: float = 0.0,
                 checkpoint_root: Path | None = None) -> None:
        self.engine = engine
        self.step_seconds = step_seconds
        self.checkpoint_root = checkpoint_root
        self.saved: list[str] = []

    def step(self) -> _Metrics:
        if self.step_seconds:
            time.sleep(self.step_seconds)
        self.engine.global_step += 1
        return _Metrics(self.engine.global_step)

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved.append(checkpoint_id)
        if self.checkpoint_root is None:
            return f"/tmp/{checkpoint_id}"
        return _write_checkpoint(self.checkpoint_root, self.engine.global_step)


def _run_cli(argv: list[str], session: _Session, engine: _Engine,
             *, checkpoint_every_steps: int = 0) -> str:
    trainer_config = SimpleNamespace(
        evaluation_every_steps=0,
        checkpoint_every_steps=checkpoint_every_steps,
    )
    setup_result = (object(), trainer_config, engine, session, object())
    stream = io.StringIO()
    with (
        patch("kefir.trainer.cli.setup", return_value=setup_result),
        patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
        patch("kefir.trainer.cli.configure_wandb", return_value=None),
        patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
        contextlib.redirect_stdout(stream),
    ):
        assert main(argv) == 0
    return stream.getvalue()


def _events(output: str, name: str) -> list[dict[str, object]]:
    found = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get(name), dict):
            found.append(value[name])
    return found


class DrainOnWallClockTests(unittest.TestCase):
    def test_target_tokens_stops_after_first_complete_crossing_block(self) -> None:
        engine = _Engine()
        engine.consumed_tokens = 0

        class VariableTokenSession(_Session):
            def step(self) -> _Metrics:
                metric = super().step()
                engine.consumed_tokens += (3, 5, 11)[engine.global_step - 1]
                return metric

        session = VariableTokenSession(engine)
        with tempfile.TemporaryDirectory() as directory:
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory,
                 "--steps", "10", "--target-tokens", "7"], session, engine,
            )
        self.assertEqual(engine.global_step, 2)
        self.assertEqual(engine.consumed_tokens, 8)
        self.assertEqual(session.saved, ["step-00000002"])

    def test_budget_exhaustion_checkpoints_and_exits_zero(self) -> None:
        engine = _Engine()
        session = _Session(engine, step_seconds=0.01)
        with tempfile.TemporaryDirectory() as directory:
            output = _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory,
                 "--steps", "500", "--max-wall-seconds", "0.05"],
                session, engine,
            )
        drained = _events(output, "drained")
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0]["reason"], "max_wall_seconds")
        self.assertEqual(drained[0]["checkpoint_id"], f"step-{engine.global_step:08d}")
        self.assertLess(engine.global_step, 500)
        self.assertGreater(engine.global_step, 0)
        self.assertEqual(session.saved, [f"step-{engine.global_step:08d}"])
        self.assertEqual(json.loads(output.splitlines()[-1]).keys(), {"drained"})

    def test_drain_on_a_checkpoint_boundary_does_not_save_twice(self) -> None:
        engine = _Engine()
        session = _Session(engine, step_seconds=0.01)
        with tempfile.TemporaryDirectory() as directory:
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory,
                 "--steps", "500", "--max-wall-seconds", "0.05",
                 "--checkpoint-every-steps", "1"],
                session, engine, checkpoint_every_steps=1,
            )
        self.assertLess(engine.global_step, 500)
        self.assertEqual(session.saved, sorted(set(session.saved)))
        self.assertEqual(len(session.saved), engine.global_step)

    def test_zero_budget_runs_every_requested_step(self) -> None:
        engine = _Engine()
        session = _Session(engine)
        with tempfile.TemporaryDirectory() as directory:
            output = _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "4"],
                session, engine,
            )
        self.assertEqual(engine.global_step, 4)
        self.assertEqual(_events(output, "drained"), [])


class LocalRetentionTests(unittest.TestCase):
    def test_keep_last_retains_milestones_best_and_the_resume_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 8):
                _write_checkpoint(root, step)
            removed = prune_checkpoints(
                root, keep_last=2, protected={"step-00000001", "step-00000003"},
                milestone_every_steps=4,
            )
            self.assertEqual(removed, ["step-00000005", "step-00000002"])
            self.assertEqual(
                _steps_on_disk(root),
                ["step-00000001", "step-00000003", "step-00000004",
                 "step-00000006", "step-00000007"],
            )

    def test_zero_keeps_every_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 4):
                _write_checkpoint(root, step)
            self.assertEqual(prune_checkpoints(root, keep_last=0), [])
            self.assertEqual(len(_steps_on_disk(root)), 3)

    def test_an_incomplete_newest_checkpoint_blocks_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 4):
                _write_checkpoint(root, step)
            (root / "step-00000009").mkdir()
            self.assertEqual(prune_checkpoints(root, keep_last=1), [])
            self.assertEqual(len(_steps_on_disk(root)), 4)

    def test_trainer_prunes_behind_new_checkpoints_and_keeps_milestones(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(engine, checkpoint_root=root)
            output = _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "5",
                 "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "2",
                 "--milestone-every-steps", "2"],
                session, engine, checkpoint_every_steps=1,
            )
            self.assertEqual(
                _steps_on_disk(root),
                ["step-00000002", "step-00000004", "step-00000005"],
            )
        retention = _events(output, "checkpoint_retention")
        self.assertEqual([event["removed"] for event in retention],
                         [["step-00000001"], ["step-00000003"]])

    def test_trainer_never_deletes_the_published_best_checkpoint(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(engine, checkpoint_root=root)
            argv = [
                "--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "4",
                "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "1",
                "--validation-blocks", "1", "--evaluation-every-steps", "1",
                "--best-model-repo", "private/best", "--best-model-recreate",
                "--wandb-run-id", "moe-prod-001",
            ]
            trainer_config = SimpleNamespace(evaluation_every_steps=1, checkpoint_every_steps=1)
            with (
                patch("kefir.trainer.cli.setup",
                      return_value=(object(), trainer_config, engine, session, object())),
                patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
                patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
                patch("kefir.trainer.cli.configure_wandb", return_value=None),
                patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
                patch("kefir.trainer.cli.get_dedicated_best_metric", return_value=None),
                patch("kefir.trainer.cli.publish_dedicated_best_model",
                      return_value={"published": True}) as publish,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(argv), 0)
            self.assertEqual(publish.call_count, 1)
            self.assertEqual(_steps_on_disk(root), ["step-00000001", "step-00000004"])
            self.assertEqual(json.loads((root / "best_checkpoint.json").read_text()),
                             {"checkpoint_id": "step-00000001", "metric": -2.0})
            restarted_engine = _Engine()
            restarted_engine.global_step = 4
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "3",
                 "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "1",
                 "--resume", "step-00000004"],
                _Session(restarted_engine, checkpoint_root=root), restarted_engine,
                checkpoint_every_steps=1,
            )
            self.assertEqual(_steps_on_disk(root),
                             ["step-00000001", "step-00000004", "step-00000007"])

    def test_retention_off_ignores_the_best_checkpoint_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "best_checkpoint.json"
            marker.write_text("invalid JSON")
            engine = _Engine()
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "1"],
                _Session(engine, checkpoint_root=root), engine,
            )
            self.assertEqual(marker.read_text(), "invalid JSON")

    def test_trainer_never_deletes_the_resume_source(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_checkpoint(root, 1)
            engine.global_step = 1
            session = _Session(engine, checkpoint_root=root)
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "3",
                 "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "1",
                 "--resume", "step-00000001"],
                session, engine, checkpoint_every_steps=1,
            )
            self.assertEqual(
                _steps_on_disk(root), ["step-00000001", "step-00000004"],
            )




def _forbidden_popen(*args, **kwargs):
    raise AssertionError("a completed run must not launch the trainer again")






if __name__ == "__main__":
    unittest.main()


class LocalBestWithoutRemoteRepoTests(unittest.TestCase):
    def test_best_pointer_is_written_without_a_best_model_repo(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(engine, checkpoint_root=root)
            argv = [
                "--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "4",
                "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "1",
                "--validation-blocks", "1", "--evaluation-every-steps", "1",
            ]
            trainer_config = SimpleNamespace(evaluation_every_steps=1, checkpoint_every_steps=1)
            with (
                patch("kefir.trainer.cli.setup",
                      return_value=(object(), trainer_config, engine, session, object())),
                patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
                patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
                patch("kefir.trainer.cli.configure_wandb", return_value=None),
                patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
                patch("kefir.trainer.cli.publish_dedicated_best_model") as publish,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(argv), 0)
            publish.assert_not_called()
            self.assertEqual(json.loads((root / "best_checkpoint.json").read_text()),
                             {"checkpoint_id": "step-00000001", "metric": -2.0})
            self.assertEqual(_steps_on_disk(root), ["step-00000001", "step-00000004"])
