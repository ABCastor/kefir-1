"""User-facing entry contracts: example files, new outputs, recipe validation at parse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kefir.sft.recipe import Execution, Recipe
from kefir.training import ExecutionSpec, RunSpec
from tests import sft_fixtures as fx

ROOT = Path(__file__).resolve().parents[1]
FAKE = {"path": "missing.jsonl", "sha256": "ab" * 32}

# Every JSON file under examples/ and the parser its command reads it with.
EXAMPLES = {
    "configs/local-run.json": RunSpec.from_json,
    "configs/local-execution.json": ExecutionSpec.from_json,
    "configs/sft-execution.json": Execution.from_json,
    "configs/sft-recipe-k1-b32-s17.json": Recipe.from_json,
    "sft-smoke/recipe.json": Recipe.from_json,
    "sft-smoke/execution.json": Execution.from_json,
}
# A record of the run, not runnable: three input hashes were never recorded publicly.
HISTORICAL = "configs/sft-recipe-k1-b32-s17.json"


def test_every_example_json_parses_with_its_parser():
    found = {str(p.relative_to(ROOT / "examples")) for p in (ROOT / "examples").rglob("*.json")}
    assert found == set(EXAMPLES)
    for name, parse in EXAMPLES.items():
        if name == HISTORICAL:
            with pytest.raises(ValueError, match="64 lowercase hex characters, got 'unknown'"):
                parse(ROOT / "examples" / name)
        else:
            parse(ROOT / "examples" / name)


# Item: outputs are refused before work and never left partial -------------------------

def test_evaluate_chat_refuses_an_existing_output_before_loading_anything(tmp_path):
    from kefir.sft.cli import evaluate_chat_main
    out = tmp_path / "eval"
    out.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        evaluate_chat_main(["--weights", str(tmp_path / "missing.safetensors"),
                            "--benchmark", str(tmp_path / "missing"), "--output", str(out)])
    with pytest.raises(ValueError, match="expected a Kefir weights file"):
        evaluate_chat_main(["--weights", str(tmp_path / "missing.safetensors"),
                            "--benchmark", str(tmp_path / "missing"), "--output", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_failed_render_leaves_no_file_and_a_retry_succeeds(tmp_path):
    from kefir.sft.data import render_rows
    from kefir.sft.renderer import ContextOverflow
    too_long = {"conversation_id": "long", "messages": [
        {"role": "user", "content": "word " * 3000}, {"role": "assistant", "content": "ok"}]}
    output = tmp_path / "train-tokens.jsonl"
    with pytest.raises(ContextOverflow):
        render_rows([fx.conversation(0), too_long], output)
    assert list(tmp_path.iterdir()) == []
    assert render_rows([fx.conversation(0)], output)["conversations"] == 1

    def untouched():
        raise AssertionError("conversations read before the output was checked")
        yield
    with pytest.raises(FileExistsError, match="output already exists"):
        render_rows(untouched(), output)


def test_failed_bench_build_leaves_no_output(tmp_path):
    from kefir.bench.build import build_suite
    out = tmp_path / "suite"
    with pytest.raises(FileNotFoundError, match="missing pinned source"):
        build_suite(out, tmp_path / "empty-cache")
    assert not out.exists() and [p.name for p in tmp_path.iterdir()] == ["empty-cache"]


# Item: recipe fields are validated at parse, before any data is read --------------------

def recipe(**changes) -> dict:
    return {**fx.recipe({"parent": FAKE, "segments": [FAKE], "validation": FAKE}), **changes}


@pytest.mark.parametrize("changes, message", [
    ({"optimizer": {"optimizer": "hybrid_muon_adamw"}}, "must set 'optimizer' and 'learning_rate'"),
    ({"optimizer": {"learning_rate": 3e-5}}, "must set 'optimizer' and 'learning_rate'"),
    ({"optimizer": {"optimizer": "sgd", "learning_rate": 3e-5}}, "adamw or hybrid_muon_adamw"),
    ({"initial_loss_scale": 0}, "initial_loss_scale"),
    ({"initial_loss_scale": True}, "initial_loss_scale"),
    ({"initial_loss_scale": "256"}, "initial_loss_scale"),
    ({"parent": {"path": "p", "sha256": "AB" * 32}}, "64 lowercase hex"),
    ({"validation": {"path": "v", "sha256": "0" * 64}}, "placeholder hash"),
    ({"schedule": {"kind": "wsd", "decay_start_chats": 8, "decay_end_chats": 8}}, "decay_start_chats < decay_end_chats"),
    ({"schedule": {"kind": "wsd", "warmup_chats": 9, "decay_start_chats": 8, "decay_end_chats": 12}}, "warmup_chats <="),
    ({"schedule": {"kind": "wsd", "decay_end_chats": 12}}, "integer"),
    ({"schedule": {"kind": "wsd", "decay_start_chats": 1, "decay_end_chats": 2, "minimum_lr_ratio": 2}},
     "minimum_lr_ratio"),
])
def test_recipe_rejects_invalid_fields_at_parse(changes, message):
    Recipe(**recipe())  # the unchanged recipe is valid without reading any file
    with pytest.raises(ValueError, match=message):
        Recipe(**recipe(**changes))


def test_input_hash_mismatch_names_expected_and_actual(tmp_path):
    from kefir.sft.train import _checked
    (tmp_path / "rows.jsonl").write_text("x\n")
    actual = hashlib.sha256(b"x\n").hexdigest()
    with pytest.raises(ValueError, match=f"expected {'ab' * 32}, got {actual}"):
        _checked(tmp_path, {"path": "rows.jsonl", "sha256": "ab" * 32})


# Precision, device and rank checks at entry --------------------------------------------

def test_fp16_evaluation_off_cuda_is_refused():
    from kefir.sft.evaluation import ModelBackend
    with pytest.raises(ValueError, match="fp16 evaluation requires a CUDA device"):
        ModelBackend(None, device="cpu", precision="fp16")


def test_reduced_precision_off_cuda_is_refused_before_the_weights_load(tmp_path, capsys):
    from kefir.generate import main as generate_main
    from kefir.sft.cli import evaluate_chat_main
    missing = str(tmp_path / "missing.safetensors")  # loading it would raise a different error
    for precision in ("fp16", "bf16"):
        with pytest.raises(SystemExit) as stop:
            generate_main(["--weights", missing, "--prompt", "x", "--precision", precision])
        assert stop.value.code == 2 and "needs a CUDA device" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stop:
        evaluate_chat_main(["--weights", missing, "--benchmark", str(tmp_path), "--precision", "fp16",
                            "--output", str(tmp_path / "out")])
    assert stop.value.code == 2 and "fp16 evaluation requires a CUDA device" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_bench_build_without_pyarrow_fails_before_any_work(monkeypatch, tmp_path):
    import importlib.util
    from kefir.bench.build import build_suite
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pyarrow" else real(name, *a))
    with pytest.raises(ModuleNotFoundError, match=r"bench extra"):
        build_suite(tmp_path / "suite", tmp_path / "cache", download=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("world, device, precision, message", [
    ("2", "cuda:1", "fp16", "cuda:LOCAL_RANK"),
    ("1", "cpu", "fp16", "fp16 recipe trains on CUDA only"),
])
def test_sft_train_rejects_execution_before_reading_inputs(monkeypatch, tmp_path, world, device, precision, message):
    from kefir.sft.train import train
    monkeypatch.setenv("WORLD_SIZE", world)
    with pytest.raises(ValueError, match=message):
        train(Recipe(**recipe(precision=precision)), tmp_path, Execution(output=str(tmp_path / "o"), device=device))


def test_sft_command_usage_exit_codes(capsys):
    from kefir.sft.cli import main
    assert main([]) == 2 and main(["nope"]) == 2 and main(["--help"]) == 0
    assert capsys.readouterr().out == ""


def test_plain_cuda_resolves_to_an_indexed_device():
    # torch.cuda.set_device rejects an un-indexed "cuda" (observed on an H200 job, 2026-11-20).
    from kefir.sft.train import resolve_device
    assert resolve_device("cuda", 0) == "cuda:0"
    assert resolve_device("cuda", 1) == "cuda:1"
    assert resolve_device("cuda:0", 1) == "cuda:0" and resolve_device("cpu", 0) == "cpu"
