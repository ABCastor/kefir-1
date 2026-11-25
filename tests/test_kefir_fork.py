"""CPU proof that a schedule fork keeps the parent optimizer and data cursor."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import socket
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp

from kefir.data.src.remote import sha256_path
from kefir.model.__main__ import parse_args
from kefir.model.setup import setup
from kefir.trainer.distributed import shutdown_distributed
from kefir.trainer.state import load_trainer_state_file
from tests.moe_fixtures import tiny_moe_config as _tiny
from kefir import RunSpec, ExecutionSpec, train
from kefir.model.config import MoEModelConfig
from kefir.model.setup import _model_config_from_args


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _equal(first, second)
    else:
        assert left == right


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_LLM_CHECKPOINT_FSYNC", "0")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    shard = dataset / "train.bin"
    shard.write_bytes(torch.arange(3 * 4 * 9).remainder(32).to(torch.uint16).numpy().tobytes())
    manifest = {
        "schema_version": 2, "sequence_format": "context_plus_one",
        "context_length": 8, "stored_tokens_per_sequence": 9,
        "sequences_per_block": 4,
        "shards": [{
            "filename": "train.bin", "split": "train", "sequence_count": 12,
            "byte_size": shard.stat().st_size, "checksum": sha256_path(shard),
            "first_block_id": 0, "last_block_id": 2,
        }],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    model_config = replace(
        _tiny(), router_scoring="sqrt_softplus", load_balancing="quantile",
        router_z_loss_coefficient=0.0,
    )
    with patch("kefir.model.setup._model_config_from_args", return_value=model_config):
        yield dataset
    torch.set_num_threads(previous_threads)


def _args(dataset: Path, checkpoint_dir: Path, *extra: str):
    return parse_args([
        "--dataset-dir", str(dataset), "--checkpoint-dir", str(checkpoint_dir),
        "--steps", "1", "--device", "cpu", "--precision", "fp32",
        "--model-size", "accepted", "--sequences-per-block", "4",
        "--optimizer", "hybrid_muon_adamw", "--microbatch-size", "2",
        "--schedule", "wsd", "--stable-tokens", "32", "--decay-tokens", "32",
        *extra,
    ])


def _parent(dataset: Path, root: Path):
    args = _args(dataset, root / "parent" / "checkpoints")
    _, _, engine, session, coordinator = setup(args)
    session.step()
    checkpoint = session.save_checkpoint(coordinator, "step-00000001")
    assert engine.global_step == 1 and engine.consumed_tokens == 32
    return checkpoint


def _child_args(dataset: Path, root: Path, parent: Path, *extra: str):
    args = _args(dataset, root / "child" / "checkpoints", *extra)
    args.schedule = "wsqd"
    args.stable_tokens = 0
    args.schedule_anchor_tokens = 32
    args.cooldown_start_tokens = 80
    args.decay_tokens = 32
    args.fork_checkpoint = parent
    return args


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_accepted_cli_preserves_checkpoint_geometry(precision, tmp_path):
    args = _args(tmp_path, tmp_path / "checkpoints", "--precision", precision)
    assert _model_config_from_args(args) == MoEModelConfig.accepted()


def test_public_fork_resumes_without_parent(prepared, tmp_path):
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["incremental_frontier"] = {"version": 1}
    path.write_text(json.dumps(manifest))
    parent = _parent(prepared, tmp_path)
    run = RunSpec(
        run_id="public-child", dataset_dir=str(prepared), target_tokens=96,
        max_steps=3, global_batch_size=4, parent_checkpoint=str(parent),
        training={"optimizer": "hybrid_muon_adamw", "learning_rate": 3e-4,
                  "schedule": "wsqd", "schedule_anchor_tokens": 32,
                  "cooldown_start_tokens": 80, "decay_tokens": 32},
    )
    execution = ExecutionSpec(run_root=str(tmp_path / "public-runs"),
                              microbatch_size=2, precision="fp32", device="cpu",
                              checkpoint_every_steps=1, evaluation_every_steps=0,
                              max_wall_seconds=1e-6)
    first = train(run, execution)
    assert first.step == 2
    shutil.rmtree(parent.parent)
    resumed = train(run, replace(execution, max_wall_seconds=0), resume="latest")
    assert resumed.step == 3 and resumed.consumed_tokens == 96


def test_fork_preserves_state_cursor_and_parent_bytes(prepared, tmp_path):
    parent = _parent(prepared, tmp_path)
    parent_state_path = parent / "trainer_state.pkl"
    parent_digest = sha256_path(parent_state_path)
    before = load_trainer_state_file(parent_state_path)
    args = _child_args(prepared, tmp_path, parent)
    args.source_commit = "a" * 40
    args.run_source_commit = "b" * 40
    _, _, engine, session, coordinator = setup(args)
    child = args.checkpoint_dir / "step-00000001"
    after = load_trainer_state_file(child / "trainer_state.pkl")
    assert args.resume is None
    assert args.fork_bootstrap_checkpoint == child.name
    assert sha256_path(parent_state_path) == parent_digest
    for name in (
        "model", "optimizer", "scaler", "global_step", "consumed_tokens",
        "overflow_events", "best_validation_loss", "python_rng_state",
        "torch_rng_state", "cuda_rng_states", "model_config",
    ):
        _equal(before[name], after[name])
    assert before["scheduler"]["last_lr"] == after["scheduler"]["last_lr"]
    assert after["scheduler"]["config"]["schedule"] == "wsqd"
    assert engine.global_step == 1 and engine.consumed_tokens == 32
    assert session.source.last_acknowledged_block_id == 0
    parent_meta = json.loads((parent / "checkpoint.json").read_text())
    child_meta = json.loads((child / "checkpoint.json").read_text())
    assert child_meta["configuration_hash"] != parent_meta["configuration_hash"]
    assert child_meta["source_hash"] == parent_meta["source_hash"]
    assert child_meta["schema_hash"] == parent_meta["schema_hash"]
    assert child_meta["pipeline_state"] == parent_meta["pipeline_state"]
    assert child_meta["fork"]["parent_checkpoint_id"] == parent.name
    assert child_meta["fork"]["parent_trainer_state_sha256"] == parent_digest
    assert child_meta["fork"]["parent_configuration_hash"] == parent_meta["configuration_hash"]
    assert child_meta["executor"] == {
        "version": 1, "source_commit": "a" * 40, "run_source_commit": "b" * 40,
    }

    metrics = session.step()
    assert metrics.step == 2 and engine.consumed_tokens == 64
    assert session.source.last_acknowledged_block_id == 1
    assert engine.scheduler.last_lr < before["scheduler"]["last_lr"]
    second = session.save_checkpoint(coordinator, "step-00000002")
    assert json.loads((second / "checkpoint.json").read_text())["fork"] == child_meta["fork"]
    resumed_args = _child_args(prepared, tmp_path, parent)
    resumed_args.fork_checkpoint = None
    resumed_args.resume = "step-00000002"
    _, _, restored, resumed, resumed_coordinator = setup(resumed_args)
    assert restored.global_step == 2 and restored.consumed_tokens == 64
    assert resumed.source.last_acknowledged_block_id == 1
    _equal(engine.state_dict()["model"], restored.state_dict()["model"])
    _equal(engine.state_dict()["optimizer"], restored.state_dict()["optimizer"])
    resumed.step()
    third = resumed.save_checkpoint(resumed_coordinator, "step-00000003")
    assert json.loads((third / "checkpoint.json").read_text())["fork"] == child_meta["fork"]


@pytest.mark.parametrize("change,reason", [
    ({"precision": "bf16"}, "non-schedule"),
    ({"learning_rate": 0.0004}, "continuous"),
    ({"schedule_anchor_tokens": 31}, "anchor"),
])
def test_fork_rejects_incompatible_child(prepared, tmp_path, change, reason):
    parent = _parent(prepared, tmp_path)
    args = _child_args(prepared, tmp_path, parent)
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=reason):
        setup(args)
    assert not args.checkpoint_dir.exists()


def test_ordinary_resume_still_rejects_schedule_change(prepared, tmp_path):
    parent = _parent(prepared, tmp_path)
    args = _child_args(prepared, tmp_path, parent)
    args.fork_checkpoint = None
    args.checkpoint_dir = parent.parent
    args.resume = parent.name
    with pytest.raises(ValueError, match="configuration fields differ"):
        setup(args)


def test_fork_rejects_changed_corpus_without_creating_child(prepared, tmp_path):
    parent = _parent(prepared, tmp_path)
    changed = tmp_path / "changed-dataset"
    shutil.copytree(prepared, changed)
    manifest_path = changed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["checksum"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest))
    args = _child_args(changed, tmp_path, parent)
    with pytest.raises(ValueError, match="source/schema"):
        setup(args)
    assert not args.checkpoint_dir.exists()


def test_fork_rejects_partial_parent_checkpoint(prepared, tmp_path):
    parent = _parent(prepared, tmp_path)
    metadata_path = parent / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["pipeline_state"]["gradient_accumulation_position"] = 1
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = parent / "local_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        if item["name"] == "checkpoint.json":
            item["sha256"] = sha256_path(metadata_path)
    manifest_path.write_text(json.dumps(manifest))
    args = _child_args(prepared, tmp_path, parent)
    with pytest.raises(ValueError, match="completed update"):
        setup(args)
    assert not args.checkpoint_dir.exists()


def test_fork_cli_requires_absolute_distinct_source(prepared, tmp_path):
    with pytest.raises(SystemExit, match="absolute path"):
        _args(prepared, tmp_path / "child", "--fork-checkpoint", "relative/step-00000001")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _args(prepared, tmp_path / "child", "--resume", "step-00000001",
              "--fork-checkpoint", str(tmp_path / "parent" / "step-00000001"))


def _gloo_fork_worker(rank: int, dataset: str, root: str, parent: str, port: int) -> None:
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank),
    })
    torch.set_num_threads(1)
    model_config = replace(
        _tiny(), router_scoring="sqrt_softplus", load_balancing="quantile",
        router_z_loss_coefficient=0.0,
    )
    try:
        with patch("kefir.model.setup._model_config_from_args", return_value=model_config):
            args = _child_args(Path(dataset), Path(root), Path(parent))
            args.sequences_per_block = 64
            args.microbatch_size = 16
            args.schedule_anchor_tokens = 512
            args.cooldown_start_tokens = 1200
            args.decay_tokens = 128
            args.learning_rate = 3e-5
            _, _, engine, session, _ = setup(args)
            (Path(root) / f"rank-{rank}.json").write_text(json.dumps({
                "step": engine.global_step, "tokens": engine.consumed_tokens,
                "cursor": session.source.last_acknowledged_block_id,
                "resume": args.resume,
                "bootstrap": args.fork_bootstrap_checkpoint,
                "checkpoint_exists": (
                    args.checkpoint_dir / args.fork_bootstrap_checkpoint
                ).is_dir(),
            }))
    finally:
        shutdown_distributed()


def test_two_cpu_ranks_share_one_bootstrap_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_LLM_CHECKPOINT_FSYNC", "0")
    dataset = tmp_path / "ddp-dataset"
    dataset.mkdir()
    shard = dataset / "train.bin"
    shard.write_bytes(torch.arange(2 * 64 * 9).remainder(32).to(torch.uint16).numpy().tobytes())
    (dataset / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "sequence_format": "context_plus_one",
        "context_length": 8, "stored_tokens_per_sequence": 9,
        "sequences_per_block": 64,
        "shards": [{
            "filename": "train.bin", "split": "train", "sequence_count": 128,
            "byte_size": shard.stat().st_size, "checksum": sha256_path(shard),
            "first_block_id": 0, "last_block_id": 1,
        }],
    }))
    model_config = replace(
        _tiny(), router_scoring="sqrt_softplus", load_balancing="quantile",
        router_z_loss_coefficient=0.0,
    )
    with patch("kefir.model.setup._model_config_from_args", return_value=model_config):
        args = _args(dataset, tmp_path / "parent" / "checkpoints")
        args.sequences_per_block = 64
        args.microbatch_size = 16
        _, _, engine, session, coordinator = setup(args)
        session.step()
        parent = session.save_checkpoint(coordinator, "step-00000001")
    assert engine.consumed_tokens == 512
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_gloo_fork_worker, args=(str(dataset), str(tmp_path), str(parent), port), nprocs=2)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result == {
            "step": 1, "tokens": 512, "cursor": 0,
            "resume": None, "bootstrap": "step-00000001", "checkpoint_exists": True,
        }
    child = tmp_path / "child" / "checkpoints" / "step-00000001"
    assert json.loads((child / "checkpoint.json").read_text())["fork"][
        "parent_checkpoint_id"
    ] == parent.name
