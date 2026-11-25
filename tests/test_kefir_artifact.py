from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from kefir.artifact import load_weights, read_weights, save_weights
from kefir.evaluate import evaluate_checkpoint
from kefir.export import export_checkpoint
from kefir.generate import generate_text
from kefir.layers.gdn2_pairwise import PairwiseGDN2Backend
from kefir.tokenizer import TOKENIZER_SHA256
from kefir.model.model import MoESmallLLM
from tests.test_moe_evaluation import _tiny_moe_config
from tests.trainer_fixtures import payload


@pytest.fixture
def saved_model(tmp_path):
    torch.manual_seed(43)
    model = MoESmallLLM(_tiny_moe_config()).eval()
    path = save_weights(model, tmp_path / "model.safetensors")
    return model, path


def test_weights_round_trip_preserves_logits_and_tied_embedding(saved_model):
    model, path = saved_model
    loaded, config, info = load_weights(path, device="cpu")
    inputs = torch.tensor([[5, 8, 21], [9, 13, 2]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(inputs), model(inputs), rtol=2e-5, atol=2e-5)
    assert loaded.embedding is loaded.lm_head
    assert config.as_dict() == model.config.as_dict()
    assert info["tokenizer_sha256"] == TOKENIZER_SHA256
    assert all(isinstance(block.mixer.backend, PairwiseGDN2Backend)
               for block in loaded.blocks if hasattr(block.mixer, "backend"))
    assert len(generate_text(path, ["Hello"], max_new_tokens=2)) == 1


def test_released_form_without_tokenizer_metadata_loads(saved_model, tmp_path):
    """The published SFT weights carry only model_config (plus free-form provenance)."""
    model, _ = saved_model
    released = tmp_path / "released.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(released),
              metadata={"model_config": json.dumps(model.config.as_dict(), sort_keys=True,
                                                    separators=(",", ":")),
                        "format": "pt", "training_exposure_chats": "15000"})
    loaded, _, info = load_weights(released)
    for key, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], tensor)
    assert "tokenizer_sha256" not in info


def test_failed_save_leaves_no_partial_file(saved_model, tmp_path, monkeypatch):
    model, _ = saved_model
    target = tmp_path / "retry" / "model.safetensors"
    from kefir import artifact

    original = artifact.save_file

    def fail_once(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(artifact, "save_file", fail_once)
    with pytest.raises(OSError, match="synthetic"):
        save_weights(model, target)
    assert list(target.parent.iterdir()) == []
    monkeypatch.setattr(artifact, "save_file", original)
    assert save_weights(model, target) == target
    with pytest.raises(FileExistsError):
        save_weights(model, target)


def test_weights_reject_wrong_checksum_and_tokenizer(saved_model, tmp_path):
    model, path = saved_model
    with pytest.raises(ValueError, match="checksum"):
        load_weights(path, expected_sha256="0" * 64)
    other = tmp_path / "other.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(other),
              metadata={"model_config": json.dumps(model.config.as_dict()),
                        "tokenizer_sha256": "0" * 64})
    with pytest.raises(ValueError, match="tokenizer"):
        load_weights(other)
    bare = tmp_path / "bare.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(bare))
    with pytest.raises(ValueError, match="model_config"):
        load_weights(bare)


class _TouchOnUnpickle:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


def test_public_inference_never_deserializes_untrusted_pickle(tmp_path):
    marker = tmp_path / "executed"
    state = tmp_path / "trainer_state.pkl"
    state.write_bytes(pickle.dumps(_TouchOnUnpickle(marker)))
    with pytest.raises(ValueError, match="weights file"):
        generate_text(state, ["Hello"])
    with pytest.raises(ValueError, match="weights file"):
        evaluate_checkpoint(state, tmp_path)
    with pytest.raises(ValueError, match="trusted=True"):
        export_checkpoint(state, tmp_path / "output.safetensors")
    with pytest.raises(ValueError, match="weights file"):
        read_weights(tmp_path)
    assert not marker.exists()


def test_export_of_explicitly_trusted_state_preserves_model(saved_model, tmp_path):
    model, _ = saved_model
    state = tmp_path / "trainer_state.pkl"
    torch.save({"version": 1, "model": model.state_dict(),
                "model_config": model.config.as_dict(), "global_step": 1,
                "consumed_tokens": 16, "optimizer": {"not_for_inference": 42}}, state)
    output = export_checkpoint(state, tmp_path / "exported.safetensors", trusted=True)
    loaded, _, info = load_weights(output)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
    with safe_open(str(output), framework="pt") as reader:
        assert "optimizer" not in reader.keys()
    assert info["source_global_step"] == "1"


def test_public_evaluation_reads_real_shards_and_weights(saved_model, tmp_path):
    model, weights = saved_model
    sequences = [[(index + offset) % 31 + 1 for index in range(9)] for offset in (0, 7, 19)]
    raw = payload(sequences)
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "validation.bin").write_bytes(raw)
    import hashlib
    (dataset / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "sequence_format": "context_plus_one",
        "tokenizer": {"tokenizer_sha256": TOKENIZER_SHA256},
        "context_length": 8, "stored_tokens_per_sequence": 9,
        "sequences_per_block": 2,
        "shards": [{"filename": "validation.bin", "split": "validation",
                    "byte_size": len(raw), "sequence_count": 3,
                    "checksum": hashlib.sha256(raw).hexdigest(),
                    "first_block_id": 0, "last_block_id": 1}],
    }))
    tokens = torch.tensor(sequences)
    with torch.inference_mode():
        expected = F.cross_entropy(model(tokens[:, :-1]).reshape(-1, 8000),
                                   tokens[:, 1:].reshape(-1)).item()
    result = evaluate_checkpoint(weights, dataset, validation_blocks=2, microbatch_size=2)
    assert result["held_out"]["loss"] == pytest.approx(expected, abs=2e-5)
    assert result["held_out"]["target_tokens"] == 24
    assert result["held_out"]["blocks"] == 2
