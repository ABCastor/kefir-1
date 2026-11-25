"""A transported checkpoint must retain the run's scientific identity."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from kefir import ExecutionSpec, RunSpec, train
from kefir.model import setup as model_setup
from kefir.tokenizer import TOKENIZER_SHA256
from tests.moe_fixtures import tiny_training_config
from tests.trainer_fixtures import payload


@pytest.fixture
def active_run(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup, "_model_config_from_args", lambda args: tiny_training_config())
    data = tmp_path / "data"
    data.mkdir()
    raw = payload([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    (data / "train.bin").write_bytes(raw)
    manifest = {
        "schema_version": 2, "sequence_format": "context_plus_one",
        "context_length": 8, "stored_tokens_per_sequence": 9,
        "sequences_per_block": 1,
        "tokenizer": {"tokenizer_sha256": TOKENIZER_SHA256},
        "production": {"configuration_hash": "a" * 64},
        "work_plan_hash": "b" * 64,
        "shards": [{"filename": "train.bin", "split": "train", "byte_size": len(raw),
                    "sequence_count": 2, "checksum": hashlib.sha256(raw).hexdigest(),
                    "first_block_id": 0, "last_block_id": 1}],
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    run = RunSpec(
        "identity-test", str(data), 16, 2,
        {"optimizer": "hybrid_muon_adamw", "learning_rate": 3e-4, "schedule": "constant"},
        global_batch_size=1,
    )
    execution = ExecutionSpec(
        str(tmp_path / "runs"), 1, device="cpu", precision="fp32",
        max_wall_seconds=1e-6,
        checkpoint_every_steps=1, evaluation_every_steps=0,
    )
    first = train(run, execution)
    assert first.status == "incomplete" and first.step == 1
    shutil.rmtree(first.checkpoint.parent.parent / "artifacts")
    return run, replace(execution, max_wall_seconds=0), first, manifest


@pytest.mark.parametrize("change", ["dataset_configuration", "target_tokens", "run_source_commit"])
def test_active_resume_rejects_scientific_identity_change(active_run, change):
    run, execution, first, manifest = active_run
    if change == "dataset_configuration":
        manifest["production"]["configuration_hash"] = "f" * 64
        (Path(run.dataset_dir) / "manifest.json").write_text(json.dumps(manifest))
    elif change == "target_tokens":
        run = replace(run, target_tokens=24)
    else:
        run = replace(run, run_source_commit="d" * 40)
        # Any well-formed commit: the identity change, not the checkout, is under test.
        execution = replace(execution, source_commit="e" * 40)
    with pytest.raises(ValueError, match="identity|contract|configuration|source|target"):
        train(run, execution, resume="latest")
    assert not (first.checkpoint.parent / "step-00000002").exists()


def test_execution_cadence_change_is_allowed(active_run):
    run, execution, _, _ = active_run
    resumed = train(run, replace(execution, checkpoint_every_steps=0), resume="latest")
    assert resumed.step == 2 and resumed.status == "complete"


def test_legacy_checkpoint_requires_exact_external_identities(active_run):
    run, execution, first, _ = active_run
    metadata_path = first.checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["scientific_run"]
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = first.checkpoint / "local_manifest.json"
    local_manifest = json.loads(manifest_path.read_text())
    for item in local_manifest["files"]:
        if item["name"] == "checkpoint.json":
            item["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(local_manifest))

    with pytest.raises(ValueError, match="legacy|no scientific"):
        train(run, execution, resume="latest")
    with pytest.raises(ValueError, match="legacy"):
        train(run, replace(execution, legacy_checkpoint_sha256="0" * 64,
                           legacy_dataset_configuration_hash="a" * 64), resume="latest")

    digest = hashlib.sha256((first.checkpoint / "trainer_state.pkl").read_bytes()).hexdigest()
    guarded = replace(execution, legacy_checkpoint_sha256=digest,
                      legacy_dataset_configuration_hash="a" * 64)
    resumed = train(run, guarded, resume="latest")
    assert resumed.step == 2
    assert json.loads((resumed.checkpoint / "checkpoint.json").read_text())["scientific_run"]["target_tokens"] == 16
