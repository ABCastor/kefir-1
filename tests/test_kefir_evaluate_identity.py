import json

import pytest

from kefir.evaluate import evaluate_checkpoint


@pytest.mark.parametrize("tokenizer", [None, {"tokenizer_sha256": "0" * 64}])
def test_held_out_dataset_requires_kefir_tokenizer(tmp_path, monkeypatch, tokenizer):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"tokenizer": tokenizer}))

    monkeypatch.setattr(
        "kefir.evaluate.load_weights",
        lambda *args, **kwargs: (object(), object(), {}),
    )
    with pytest.raises(ValueError, match="tokenizer"):
        evaluate_checkpoint(tmp_path / "checkpoint", data)
