"""Load the one tokenizer used for Kefir v1 text and generation."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json

from tokenizers import Tokenizer


TOKENIZER_SHA256 = "068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb"
SEMANTIC_VOCAB_SIZE = 8000
EOD_TOKEN_ID = 7992


class KefirTokenizer:
    def __init__(self, source: bytes) -> None:
        actual = hashlib.sha256(source).hexdigest()
        if actual != TOKENIZER_SHA256:
            raise ValueError(f"Kefir tokenizer hash mismatch: {actual}")
        original = json.loads(source)
        ordinary = dict(original)
        ordinary["added_tokens"] = []
        ordinary_model = dict(original["model"])
        ordinary_model["vocab"] = {
            token: token_id for token, token_id in ordinary_model["vocab"].items()
            if token_id < EOD_TOKEN_ID
        }
        ordinary["model"] = ordinary_model
        self._text = Tokenizer.from_str(json.dumps(ordinary, ensure_ascii=False))
        self._decoder = Tokenizer.from_str(source.decode("utf-8"))

    def encode(self, text: str) -> list[int]:
        ids = self._text.encode(text, add_special_tokens=False).ids
        if any(token_id >= EOD_TOKEN_ID for token_id in ids):
            raise RuntimeError("ordinary text encoded as a reserved control token")
        return ids

    def decode(self, token_ids: list[int]) -> str:
        if any(not 0 <= token_id < SEMANTIC_VOCAB_SIZE for token_id in token_ids):
            raise ValueError("token ID is outside the semantic vocabulary")
        return self._decoder.decode(token_ids, skip_special_tokens=False)


def load_tokenizer() -> KefirTokenizer:
    asset = files("kefir.assets").joinpath("superbpe_8000_v2.json")
    return KefirTokenizer(asset.read_bytes())


__all__ = ["EOD_TOKEN_ID", "KefirTokenizer", "SEMANTIC_VOCAB_SIZE", "TOKENIZER_SHA256",
           "load_tokenizer"]
