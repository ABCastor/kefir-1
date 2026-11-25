#!/usr/bin/env python3
"""Verify the bundled Kefir 1 tokenizer: hash, layout and encode/decode round trips.

Needs only `tokenizers`. With --model, also re-exports a producer `.model` file (for example
`superbpe_8000_final.model` from the Hugging Face tokenizer repo) with export_hf.py and checks
that the result is byte-identical to the bundled file; that step needs `boundlessbpe`.

    python verify_tokenizer.py [--tokenizer PATH] [--model superbpe_8000_final.model]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SHA256 = "068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb"
ASSET = Path("src/kefir/assets/superbpe_8000_v2.json")
VOCAB_SIZE = 8000
STAGE1_VOCAB = 7192      # 243 byte entries + 6,949 merges
SUPERMERGES = 800
EOD = 7992
SPECIAL = ["<|endoftext|>", "<think>", "</think>", "<answer>",
           "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>", "<|reserved_3|>"]
# The 13 bytes that cannot occur in valid UTF-8 are absent from the byte alphabet.
ABSENT_BYTES = [0xC0, 0xC1, *range(0xF5, 0x100)]
SAMPLES = [
    "Hello, world. The cat sat on the mat, and that was the end of it.",
    "By the way, I think you know what kind of day it is.",
    "In 1969, 1,234,567 people watched; 3.14159 is close to pi.",
    "Tabs\tand\nnewlines\r\nand  double  spaces   stay.",
    "Café, naïve, Straße, 水, ∀x λ, emoji 🐍🙂.",
    "Literal <|endoftext|> and <think> in text are ordinary characters.",
    "".join(chr(i) for i in range(128)),
    " leading space and trailing space ",
    "",
]


def find_asset(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ASSET).is_file():
            return parent / ASSET
    raise FileNotFoundError(f"{ASSET} not found above {start}; pass --tokenizer")


def ordinary_tokenizer(data: dict):
    """The text encoder used by Kefir: special tokens and ids >= 7992 removed."""
    from tokenizers import Tokenizer
    plain = dict(data, added_tokens=[])
    plain["model"] = dict(data["model"], vocab={t: i for t, i in data["model"]["vocab"].items() if i < EOD})
    return Tokenizer.from_str(json.dumps(plain, ensure_ascii=False))


def verify(path: Path) -> dict:
    from tokenizers import Tokenizer
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SHA256:
        raise SystemExit(f"hash mismatch: {actual}")
    data = json.loads(raw)
    model = data["model"]
    vocab, merges = model["vocab"], model["merges"]
    assert model["type"] == "BPE" and model["unk_token"] is None and not model["byte_fallback"]
    assert sorted(vocab.values()) == list(range(VOCAB_SIZE)), "ids are not exactly 0..7999"
    assert len(merges) == STAGE1_VOCAB - (256 - len(ABSENT_BYTES)) + SUPERMERGES == 7749
    assert [(t["id"], t["content"]) for t in data["added_tokens"]] == list(enumerate(SPECIAL, EOD))
    steps = [p["type"] for p in data["pre_tokenizer"]["pretokenizers"]]
    assert steps == ["Digits", "Split", "ByteLevel"] and data["pre_tokenizer"]["pretokenizers"][2]["use_regex"] is False
    inv = {i: t for t, i in vocab.items()}
    superwords = [inv[i] for i in range(STAGE1_VOCAB, EOD)]
    # Stage-2 entries span a space inside the token; stage-1 entries never do.
    assert sum("Ġ" in t[1:] for t in superwords) >= 0.9 * SUPERMERGES
    assert not any("Ġ" in inv[i][1:] for i in range(STAGE1_VOCAB) if len(inv[i]) > 1 and inv[i].strip("Ġ"))

    full = Tokenizer.from_str(raw.decode("utf-8"))
    text = ordinary_tokenizer(data)
    n_tokens = 0
    for s in SAMPLES:
        ids = text.encode(s, add_special_tokens=False).ids
        assert all(0 <= i < EOD for i in ids), f"ordinary text produced a control id: {s!r}"
        assert full.decode(ids, skip_special_tokens=False) == s, f"round trip failed: {s!r}"
        n_tokens += len(ids)
    assert full.encode("<|endoftext|>").ids == [EOD]
    assert full.decode([EOD], skip_special_tokens=False) == "<|endoftext|>"
    return {"sha256": actual, "vocab": len(vocab), "merges": len(merges), "superwords": len(superwords),
            "round_trips": len(SAMPLES), "sample_tokens": n_tokens}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--model", type=Path, help="producer .model to re-export and compare")
    a = ap.parse_args()
    path = a.tokenizer or find_asset(Path(__file__).resolve().parent)
    result = verify(path)
    if a.model:
        import export_hf
        data, _, _ = export_hf.convert(str(a.model))
        exported = hashlib.sha256(export_hf.dumps(data).encode()).hexdigest()
        if exported != SHA256:
            raise SystemExit(f"re-export differs: {exported}")
        result["re_export_sha256"] = exported
    print(json.dumps(result))


if __name__ == "__main__":
    main()
