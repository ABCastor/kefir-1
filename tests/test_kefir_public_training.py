"""Public recipe boundary without a GPU, provider, or network service."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kefir import ExecutionSpec, RunSpec, train
from kefir.tokenizer import TOKENIZER_SHA256
from kefir.training import _latest_checkpoint
from kefir.training import _arguments
from kefir.trainer.shard_layout import manifest_identity
from tests.moe_fixtures import tiny_training_config
from tests.trainer_fixtures import payload


def _run() -> RunSpec:
    return RunSpec(
        run_id="kefir-qualification", dataset_dir="prepared", target_tokens=131072,
        max_steps=2, training={
            "optimizer": "hybrid_muon_adamw", "learning_rate": 3e-4,
            "schedule": "wsd", "decay_tokens": 131072,
        },
    )


class PublicTrainingTests(unittest.TestCase):
    def test_local_schema_v2_run_and_completed_resume_use_real_loop(self) -> None:
        tiny = tiny_training_config()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "prepared"
            (data / "train").mkdir(parents=True)
            raw = payload([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
            (data / "train" / "part.bin").write_bytes(raw)
            manifest = {
                "schema_version": 2, "sequence_format": "context_plus_one",
                "context_length": 8, "stored_tokens_per_sequence": 9,
                "sequences_per_block": 1,
                "tokenizer": {"tokenizer_sha256": TOKENIZER_SHA256},
                "shards": [{
                    "filename": "train/part.bin", "split": "train", "byte_size": len(raw),
                    "sequence_count": 1, "checksum": hashlib.sha256(raw).hexdigest(),
                    "first_block_id": 0, "last_block_id": 0,
                }],
            }
            (data / "manifest.json").write_text(json.dumps(manifest))
            run = RunSpec(
                run_id="tiny-public", dataset_dir=str(data), target_tokens=8,
                max_steps=1, global_batch_size=1,
                training={"optimizer": "hybrid_muon_adamw", "learning_rate": 3e-4,
                          "schedule": "constant"},
            )
            execution = ExecutionSpec(
                run_root=str(Path(tmp) / "runs"), microbatch_size=1,
                device="cpu", precision="fp32", checkpoint_every_steps=1,
                evaluation_every_steps=0,
            )
            with patch("kefir.model.setup._model_config_from_args", return_value=tiny):
                result = train(run, execution)
                resumed = train(run, execution, resume="latest")
            self.assertEqual(result.status, "complete")
            self.assertEqual((result.step, result.consumed_tokens), (1, 8))
            self.assertEqual(resumed, result)

    def test_execution_rejects_malformed_precision_and_boolean_controls(self):
        for value in ({"precision": "fp64"}, {"compile_mode": "auto"},
                      {"checkpoint_fsync": "false"}, {"optimizer_telemetry": 0},
                      {"local_incremental_frontier": "true"},
                      {"remote_rolling_latest_only": "true"},
                      {"async_checkpoint_upload": 1},
                      {"keep_last_checkpoints": -1},
                      {"thematic_microbatch_size": 0},
                      {"thematic_microbatch_size": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExecutionSpec(run_root="runs", microbatch_size=4, **value)

    def test_thematic_microbatch_is_sent_to_validation_only(self) -> None:
        execution = ExecutionSpec(
            run_root="runs", microbatch_size=4,
            thematic_panels="/tmp/panels", thematic_microbatch_size=8,
        )
        argv = _arguments(_run(), execution, steps=1, resume=None,
                          checkpoint_dir=Path("/tmp/checkpoints"),
                          experiment_dir=Path("/tmp/artifacts"))
        self.assertEqual(argv[argv.index("--thematic-microbatch") + 1], "8")
        self.assertEqual(argv[argv.index("--microbatch-size") + 1], "4")

    def test_remote_checkpoint_controls_are_execution_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "require remote publication"):
            ExecutionSpec(run_root="runs", microbatch_size=4,
                          remote_rolling_latest_only=True)
        execution = ExecutionSpec(
            run_root="runs", microbatch_size=4,
            remote_checkpoint_bucket="owner/checkpoints",
            remote_drive_manifest="/tmp/drive_manifest.json",
            remote_publish_every_steps=500,
            remote_rolling_latest_only=True,
            async_checkpoint_upload=True,
            keep_last_checkpoints=2,
        )
        argv = _arguments(_run(), execution, steps=1, resume=None,
                          checkpoint_dir=Path("/tmp/checkpoints"),
                          experiment_dir=Path("/tmp/artifacts"))
        self.assertEqual(argv[argv.index("--keep-last-checkpoints") + 1], "2")
        self.assertIn("--remote-rolling-latest-only", argv)
        self.assertIn("--async-checkpoint-upload", argv)

    def test_local_incremental_frontier_is_explicit_and_offline(self) -> None:
        run = replace(_run(), data_manifest="/tmp/manifest.json")
        execution = ExecutionSpec(run_root="runs", microbatch_size=4,
                                  local_incremental_frontier=True)
        argv = _arguments(run, execution, steps=1, resume=None,
                          checkpoint_dir=Path("/tmp/checkpoints"),
                          experiment_dir=Path("/tmp/artifacts"))
        self.assertIn("--local-incremental-frontier", argv)
        self.assertNotIn("--dataset-shard-bucket", argv)
        with self.assertRaisesRegex(ValueError, "cannot use a remote"):
            replace(execution, dataset_shard_bucket="owner/bucket", dataset_shard_run_id="run")

    def test_json_contract_rejects_unknown_scientific_and_execution_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            payload = {name: getattr(_run(), name) for name in
                       ("run_id", "dataset_dir", "target_tokens", "max_steps", "training")}
            payload["training"]["microbatch_size"] = 2
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "execution-only"):
                RunSpec.from_json(path)
            del payload["training"]["microbatch_size"]
            payload["surprise"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                RunSpec.from_json(path)

    def test_public_api_invokes_same_moe_entrypoint_with_explicit_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "kefir-qualification" / "checkpoints" / "step-00000001"
            snapshot = {"step": 1, "consumed_tokens": 131072, "checkpoint_id": checkpoint.name,
                        "path": checkpoint}
            observed: list[list[str]] = []

            def fake_main(argv: list[str]) -> int:
                observed.append(argv)
                return 0

            execution = ExecutionSpec(run_root=tmp, microbatch_size=4, precision="fp32",
                                      max_wall_seconds=30)
            with (patch("kefir.model.__main__.main", side_effect=fake_main),
                  patch("kefir.training._latest_checkpoint",
                        side_effect=[None, snapshot])):
                result = train(_run(), execution)
            args = observed[0]
            self.assertEqual(args[args.index("--microbatch-size") + 1], "4")
            self.assertEqual(args[args.index("--target-tokens") + 1], "131072")
            self.assertEqual(args[args.index("--schedule") + 1], "wsd")
            self.assertEqual(args[args.index("--decay-tokens") + 1], "131072")
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.consumed_tokens, 131072)

    def test_resume_requires_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "kefir-qualification" / "checkpoints"
            (checkpoint_dir / "step-00000001").mkdir(parents=True)
            snapshot = {"step": 1, "consumed_tokens": 65536, "checkpoint_id": "step-00000001",
                        "path": checkpoint_dir / "step-00000001"}
            with patch("kefir.training._latest_checkpoint", return_value=snapshot):
                with self.assertRaisesRegex(ValueError, "pass resume"):
                    train(_run(), ExecutionSpec(run_root=tmp, microbatch_size=1))

    def test_schedule_fork_uses_remaining_absolute_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent" / "checkpoints" / "step-00000001"
            run = replace(_run(), parent_checkpoint=str(parent), training={
                "optimizer": "hybrid_muon_adamw", "learning_rate": 3e-4,
                "schedule": "wsqd", "schedule_anchor_tokens": 65536,
                "cooldown_start_tokens": 120000, "decay_tokens": 100000,
            })
            observed = []
            parent_state = {"step": 1, "consumed_tokens": 65536,
                            "checkpoint_id": parent.name, "path": parent}
            child_state = {"step": 2, "consumed_tokens": 131072,
                           "checkpoint_id": "step-00000002",
                           "path": Path(tmp) / run.run_id / "checkpoints" / "step-00000002"}

            def fake_main(argv: list[str]) -> int:
                observed.extend(argv)
                return 0

            with (patch("kefir.model.__main__.main", side_effect=fake_main),
                  patch("kefir.training.complete_checkpoint", return_value=parent_state),
                  patch("kefir.training._latest_checkpoint", side_effect=[None, child_state])):
                result = train(run, ExecutionSpec(run_root=tmp, microbatch_size=4))
            self.assertEqual(observed[observed.index("--steps") + 1], "1")
            self.assertEqual(observed[observed.index("--fork-checkpoint") + 1], str(parent))
            self.assertEqual(result.status, "complete")

    def test_dataset_manifest_must_name_the_pinned_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "prepared"
            dataset.mkdir()
            (dataset / "manifest.json").write_text(json.dumps({
                "schema_version": 2, "tokenizer": {"tokenizer_sha256": "0" * 64},
            }))
            run = replace(_run(), dataset_dir=str(dataset))
            with patch("kefir.model.__main__.main") as trainer:
                with self.assertRaisesRegex(ValueError, "tokenizer differs"):
                    train(run, ExecutionSpec(run_root=tmp, microbatch_size=1))
            trainer.assert_not_called()

    def test_rolling_dataset_flags_are_execution_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = replace(_run(), data_manifest=str(Path(tmp) / "manifest.json"))
            execution = ExecutionSpec(
                run_root=tmp, microbatch_size=4,
                dataset_shard_bucket="owner/shards", dataset_shard_run_id="corpus-003",
                dataset_shard_prefetch=2, dataset_shard_wait_timeout_seconds=60,
            )
            args = _arguments(run, execution, steps=1, resume=None,
                              checkpoint_dir=Path(tmp) / "checkpoints",
                              experiment_dir=Path(tmp) / "artifacts")
            self.assertEqual(args[args.index("--dataset-shard-bucket") + 1], "owner/shards")
            self.assertEqual(args[args.index("--dataset-shard-run-id") + 1], "corpus-003")
            self.assertEqual(args[args.index("--dataset-shard-prefetch") + 1], "2")
            with self.assertRaisesRegex(ValueError, "both bucket and run ID"):
                replace(execution, dataset_shard_run_id=None)
            with self.assertRaisesRegex(ValueError, "RunSpec.data_manifest"):
                _arguments(replace(run, data_manifest=None), execution, steps=1, resume=None,
                           checkpoint_dir=Path(tmp), experiment_dir=Path(tmp))

    def test_historical_incremental_manifest_requires_matching_parent_cursor_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            manifest = {
                "schema_version": 2, "context_length": 8,
                "sequences_per_block": 64, "incremental_frontier": {"version": 1},
                "shards": [{"filename": "train/part.bin", "split": "train",
                            "byte_size": 1170, "sequence_count": 65,
                            "checksum": "a" * 64, "first_block_id": 0,
                            "last_block_id": 1}],
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            parent = root / "parent"
            parent.mkdir()
            identity = manifest_identity(manifest, 8, 64)
            (parent / "checkpoint.json").write_text(json.dumps({
                "pipeline_state": {"consumer": {"manifest_identity": identity}},
            }))
            run = replace(_run(), dataset_dir=str(dataset), parent_checkpoint=str(parent))
            from kefir.training import _validate_dataset_tokenizer
            _validate_dataset_tokenizer(run)
            with self.assertRaisesRegex(ValueError, "tokenizer differs"):
                _validate_dataset_tokenizer(replace(run, parent_checkpoint=None))
            (parent / "checkpoint.json").write_text(json.dumps({
                "pipeline_state": {"consumer": {"manifest_identity": "0" * 64}},
            }))
            with self.assertRaisesRegex(ValueError, "tokenizer differs"):
                _validate_dataset_tokenizer(run)

    def test_two_rank_preflight_checks_per_rank_microbatch(self) -> None:
        with (tempfile.TemporaryDirectory() as tmp,
              patch.dict("os.environ", {"WORLD_SIZE": "2"})):
            with self.assertRaisesRegex(ValueError, "per-rank"):
                train(_run(), ExecutionSpec(run_root=tmp, microbatch_size=64))

    def test_checkpoint_discovery_does_not_mutate_an_incomplete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step-00000002"
            path.mkdir()
            with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                _latest_checkpoint(Path(tmp))
            self.assertTrue(path.is_dir())


if __name__ == "__main__":
    unittest.main()
