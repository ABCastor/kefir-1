#!/usr/bin/env python3
"""Convert a two-stage fastboundlessbpe `.model` into a Hugging Face `tokenizer.json`.

The layout follows the public SuperBPE release: a BPE model whose merges are the stage-1
merges then the stage-2 supermerges, in training order; pre-tokenizer = Digits, a split into
groups of three digits from the right, then ByteLevel(use_regex=False) so that merges across
spaces are reachable at inference; ByteLevel decoder. Special tokens keep the ids the trainer
gave them.

The output is written as `json.dumps(data, indent=2) + "\n"`, the exact byte format of the
published file: converting the published `superbpe_8000_final.model` with this script gives
SHA-256 068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb.

    python export_hf.py superbpe_8000_final.model tokenizer.json [--parity heldout.jsonl --docs 5000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def convert(model_path: str):
    from boundlessbpe import Tokenizer
    from boundlessbpe.util import frombytes, byte_map
    ref = Tokenizer()
    ref.load(model_path)
    # The byte alphabet must be the GPT-2 bytes_to_unicode table that HF ByteLevel uses.
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    gpt2 = {bytes([b]): chr(c) for b, c in zip(bs, cs)}
    mismatches = [b for b in range(256) if byte_map[bytes([b])] != gpt2[bytes([b])]]
    vocab = {frombytes(tok): i for i, tok in enumerate(ref.vocab.tokens)}
    specials = sorted(ref.vocab.special_tokens.items(), key=lambda kv: kv[1])
    for s, i in specials:
        vocab[s] = i

    def pairs(section):
        out = []
        for key in sorted(section.merges):        # index -> ((left, right), count, flag), training order
            (a, b) = section.merges[key][0]
            out.append([frombytes(a), frombytes(b)])
        return out

    merges = pairs(ref.words) + (pairs(ref.superwords) if ref.superwords else [])
    data = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for s, i in specials],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": False},
            {"type": "Split", "pattern": {"Regex": "(?=(\\d{3})+(?!\\d))"}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }
    meta = {"word_merges": len(pairs(ref.words)),
            "supermerges": len(ref.superwords.merges) if ref.superwords else 0,
            "byte_map_mismatches_vs_hf": len(mismatches)}
    return data, ref, meta


def dumps(data: dict) -> str:
    """The byte format of the published tokenizer.json."""
    return json.dumps(data, indent=2) + "\n"


def parity(hf_json: str, ref, heldout: str, docs: int) -> dict:
    """Compare the exported file with the reference implementation on held-out JSONL text."""
    from tokenizers import Tokenizer as HfTokenizer
    hf = HfTokenizer.from_file(hf_json)
    texts = []
    with open(heldout, encoding="utf-8") as f:
        for line in f:
            if len(texts) >= docs:
                break
            texts.append(json.loads(line)["text"])
    t0 = time.time()
    hf_ids = [e.ids for e in hf.encode_batch(texts)]
    t_hf = time.time() - t0
    t0 = time.time()
    ref_ids = [ref.encode(t, allowed_special="none") for t in texts]
    t_ref = time.time() - t0
    same = sum(1 for a, b in zip(hf_ids, ref_ids) if a == b)
    nb = sum(len(t.encode()) for t in texts)
    rt = sum(1 for t, ids in zip(texts, hf_ids) if hf.decode(ids) != t)
    return {"docs": len(texts), "identical_docs": same, "identical_pct": round(100 * same / len(texts), 3),
            "hf_tokens": sum(map(len, hf_ids)), "ref_tokens": sum(map(len, ref_ids)),
            "bytes_per_token_hf": round(nb / sum(map(len, hf_ids)), 4),
            "bytes_per_token_ref": round(nb / sum(map(len, ref_ids)), 4),
            "roundtrip_failures_hf": rt, "seconds_hf": round(t_hf, 1), "seconds_ref": round(t_ref, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--parity", help="held-out JSONL with a 'text' field")
    ap.add_argument("--docs", type=int, default=5000)
    a = ap.parse_args()
    data, ref, meta = convert(a.model)
    Path(a.out).write_text(dumps(data), encoding="utf-8")
    print(json.dumps({"out": a.out, "vocab": len(data["model"]["vocab"]),
                      "merges": len(data["model"]["merges"]), **meta}))
    if a.parity:
        print(json.dumps({"parity": parity(a.out, ref, a.parity, a.docs)}))


if __name__ == "__main__":
    main()
