"""Completion must validate the requested run even when no more updates are needed."""

from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from kefir import ExecutionSpec, RunSpec, train
from kefir.model import setup as model_setup
from kefir.tokenizer import TOKENIZER_SHA256
from tests.moe_fixtures import tiny_training_config
from tests.trainer_fixtures import payload


@pytest.fixture
def completed(tmp_path, monkeypatch):
    original_resolver = model_setup._model_config_from_args
    monkeypatch.setattr(model_setup, "_model_config_from_args", lambda args: tiny_training_config())
    data = tmp_path / "data"
    data.mkdir()
    raw = payload([[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2)
    (data / "train.bin").write_bytes(raw)
    manifest = {
        "schema_version": 2, "sequence_format": "context_plus_one",
        "context_length": 8, "stored_tokens_per_sequence": 9, "sequences_per_block": 1,
        "tokenizer": {"tokenizer_sha256": TOKENIZER_SHA256},
        "work_plan_hash": "a" * 64,
        "shards": [{"filename": "train.bin", "split": "train", "byte_size": len(raw),
                    "sequence_count": 2, "checksum": hashlib.sha256(raw).hexdigest(),
                    "first_block_id": 0, "last_block_id": 1}],
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    run = RunSpec("completed-review", str(data), 8, 1,
                  {"optimizer": "hybrid_muon_adamw", "learning_rate": 3e-4, "schedule": "constant"},
                  global_batch_size=1)
    execution = ExecutionSpec(str(tmp_path / "runs"), 1, device="cpu", precision="fp32",
                              checkpoint_every_steps=1, evaluation_every_steps=0)
    result = train(run, execution)
    return run, execution, result, manifest, original_resolver


@pytest.mark.parametrize("change", ["learning_rate", "precision", "dataset", "missing", "model"])
def test_completed_resume_rejects_incompatible_identity(completed, monkeypatch, change):
    run, execution, _, manifest, original_resolver = completed
    if change == "learning_rate":
        run = replace(run, training={**run.training, "learning_rate": 0.01})
    elif change == "precision":
        execution = replace(execution, precision="bf16")
    elif change == "dataset":
        # Preserve tokenizer and source-plan ID: reader identity must still catch changed shards.
        manifest["shards"][0]["checksum"] = "f" * 64
        (Path(run.dataset_dir) / "manifest.json").write_text(json.dumps(manifest))
    elif change == "missing":
        (Path(run.dataset_dir) / "manifest.json").unlink()
    else:
        monkeypatch.setattr(model_setup, "_model_config_from_args", original_resolver)
    with pytest.raises(ValueError, match="configuration|consumer|manifest|identity"):
        train(run, execution, resume="latest")


def test_completed_resume_allows_execution_changes_without_gpu(completed, monkeypatch):
    import kefir.model.__main__ as entrypoint
    run, execution, result, _, _ = completed
    monkeypatch.setattr(entrypoint, "main", lambda args: pytest.fail("completed run launched trainer"))
    execution = replace(execution, device="cuda", checkpoint_every_steps=25, evaluation_every_steps=25)
    assert train(run, execution, resume="latest") == result


def test_completed_resume_validates_scheduler_lr(completed):
    run, execution, result, _, _ = completed
    state_path = result.checkpoint / "trainer_state.pkl"
    with state_path.open("rb") as stream:
        state = pickle.load(stream)
    state["scheduler"]["last_lr"] = 0.01
    with state_path.open("wb") as stream:
        pickle.dump(state, stream)
    manifest_path = result.checkpoint / "local_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        if item["name"] == state_path.name:
            item["sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scheduler state LR"):
        train(run, execution, resume="latest")


def test_checked_in_examples_use_real_kefir_block_geometry(tmp_path):
    from kefir.model.__main__ import parse_args
    from kefir.training import _arguments

    examples = Path(__file__).resolve().parents[1] / "examples/configs"
    run = RunSpec.from_json(examples / "local-run.json")
    execution = ExecutionSpec.from_json(examples / "local-execution.json")
    args = parse_args(_arguments(run, execution, steps=run.max_steps, resume=None,
                                checkpoint_dir=tmp_path / "checkpoints", experiment_dir=tmp_path / "artifacts"))
    model = model_setup._model_config_from_args(args)
    assert run.global_batch_size == args.sequences_per_block == 64
    assert model.max_seq_len == 2048
    assert run.target_tokens == run.global_batch_size * model.max_seq_len == 131072
    assert run.global_batch_size % execution.microbatch_size == 0
    assert execution.precision == "bf16"
    assert model.dense.gdn_chunk_size == 32


def test_completed_budget_extension_requires_explicit_flag(completed):
    from kefir.training import _verify_scientific_run
    run, execution, result, _, _ = completed
    longer = replace(run, target_tokens=16, max_steps=2)
    with pytest.raises(ValueError, match="scientific identity"):
        _verify_scientific_run(longer, execution, result.checkpoint)
    _verify_scientific_run(replace(longer, extend_budget=True), execution, result.checkpoint)


@pytest.mark.parametrize("change", ["lr", "batch", "tokens_only", "steps_only", "shorter"])
def test_budget_extension_preserves_recipe_and_increases_both_bounds(completed, change):
    from kefir.training import _verify_scientific_run
    run, execution, result, _, _ = completed
    longer = replace(run, target_tokens=16, max_steps=2, extend_budget=True)
    if change == "lr":
        longer = replace(longer, training={**longer.training, "learning_rate": 1e-5})
    elif change == "batch":
        longer = replace(longer, global_batch_size=2)
    elif change == "tokens_only":
        longer = replace(longer, max_steps=1)
    elif change == "steps_only":
        longer = replace(longer, target_tokens=8)
    else:
        longer = replace(longer, target_tokens=4)
    with pytest.raises(ValueError, match="scientific identity"):
        _verify_scientific_run(longer, execution, result.checkpoint)


def test_budget_extension_reaches_real_trainer_and_resumes_full_state(completed):
    run, execution, result, _, _ = completed
    extended = replace(run, target_tokens=16, max_steps=2, extend_budget=True)
    resumed = train(extended, execution, resume="latest")
    assert resumed.step == 2
    assert resumed.consumed_tokens == 16
    metadata = json.loads((resumed.checkpoint / "checkpoint.json").read_text())
    assert metadata["scientific_run"]["max_steps"] == 2
    assert metadata["scientific_run"]["target_tokens"] == 16
