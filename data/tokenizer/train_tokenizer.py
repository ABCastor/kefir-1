#!/usr/bin/env python3
"""Train the Kefir 1 SuperBPE tokenizer in two stages, on one CPU, and export tokenizer.json.

Stage 1 is ordinary byte-level BPE under the GPT-4o pre-tokenization regex, so tokens never
cross a pre-token boundary; it stops at 7,192 entries (243 byte entries + 6,949 merges).
Stage 2 starts from the stage-1 model, lifts the whitespace boundary and learns 800
supermerges that may span several words. Eight special tokens follow, ids 7992-7999.

The defaults are the parameters of the published tokenizer. The calls match the original
fitting script line for line; only file handling and the contract checks were rewritten.

    python train_tokenizer.py --stage1 stage1.jsonl --stage2 stage2.jsonl --out work/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import export_hf

SPECIAL = ["<|endoftext|>", "<think>", "</think>", "<answer>",
           "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>", "<|reserved_3|>"]


def count_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train(stage1: Path, stage2: Path, out: Path, *, stage1_vocab: int = 7192, supermerges: int = 800,
          tau: float = 1.1, max_bytes_stage1: int = 11_000_000_000,
          max_bytes_stage2: int = 1_100_000_000) -> dict:
    from boundlessbpe import FastBpeTrainer, FastSuperBpeTrainer, GPT4O_REGEX
    out.mkdir(parents=True, exist_ok=True)
    word = out / "word_final.model"
    final = out / "superbpe_8000_final.model"
    n1, n2 = count_lines(stage1), count_lines(stage2)
    started = time.time()

    # Stage 1: subwords. tau > 1 disables PickyBPE deletions, so this is plain BPE.
    w = FastBpeTrainer(main_regex=GPT4O_REGEX)
    w.train(tau=tau, filepath=str(stage1), outprefix=str(out / "word"), num_lines=n1,
            vocab_size=stage1_vocab, recalc=0, blowup=False, max_bytes=max_bytes_stage1, verbose=False)
    w.register_special_tokens(SPECIAL)
    w.save(str(word.with_suffix("")))
    del w
    t1 = time.time()

    # Stage 2: superwords, learned on top of the stage-1 model.
    s = FastSuperBpeTrainer(main_regex=GPT4O_REGEX)
    s.train(filepath=str(stage2), outprefix=str(out / "super"), num_lines=n2, vocab_size=supermerges,
            recalc=0, word_model_file=str(word), max_bytes=max_bytes_stage2, verbose=False)
    s.register_special_tokens(SPECIAL)
    s.save(str(final.with_suffix("")))
    del s
    t2 = time.time()

    data, _, meta = export_hf.convert(str(final))
    path = out / "tokenizer.json"
    path.write_text(export_hf.dumps(data), encoding="utf-8")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    learned = len(data["model"]["vocab"]) - len(SPECIAL)
    if learned != stage1_vocab + supermerges:
        raise RuntimeError(f"{learned} learned entries, expected {stage1_vocab} + {supermerges}: "
                           "a stage stopped early, the training text is too small")
    if meta["supermerges"] != supermerges or meta["byte_map_mismatches_vs_hf"] != 0:
        raise RuntimeError(f"invalid vocabulary contract: {meta}")
    for i, s_tok in enumerate(SPECIAL, learned):
        if tok.token_to_id(s_tok) != i:
            raise RuntimeError(f"special token {s_tok!r} is not at id {i}")
    receipt = {"stage1": {"docs": n1, "sha256": sha256(stage1)}, "stage2": {"docs": n2, "sha256": sha256(stage2)},
               "stage1_vocab": stage1_vocab, "supermerges": supermerges, "tau": tau,
               "max_bytes_stage1": max_bytes_stage1, "max_bytes_stage2": max_bytes_stage2,
               "vocab_size": tok.get_vocab_size(), "eod_id": learned, **meta,
               "stage1_seconds": round(t1 - started, 1), "stage2_seconds": round(t2 - t1, 1),
               "tokenizer_sha256": sha256(path)}
    (out / "train-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, required=True, help="JSONL with a 'text' field")
    ap.add_argument("--stage2", type=Path, required=True, help="JSONL with a 'text' field")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage1-vocab", type=int, default=7192,
                    help="stage-1 vocabulary including the byte entries (switch point)")
    ap.add_argument("--supermerges", type=int, default=800)
    ap.add_argument("--tau", type=float, default=1.1)
    ap.add_argument("--max-bytes-stage1", type=int, default=11_000_000_000)
    ap.add_argument("--max-bytes-stage2", type=int, default=1_100_000_000)
    ap.add_argument("--parity", type=Path, help="held-out JSONL for the parity gate")
    ap.add_argument("--parity-docs", type=int, default=5000)
    a = ap.parse_args()
    receipt = train(a.stage1, a.stage2, a.out, stage1_vocab=a.stage1_vocab, supermerges=a.supermerges,
                    tau=a.tau, max_bytes_stage1=a.max_bytes_stage1, max_bytes_stage2=a.max_bytes_stage2)
    print(json.dumps(receipt, indent=2))
    if a.parity:
        _, ref, _ = export_hf.convert(str(a.out / "superbpe_8000_final.model"))
        par = export_hf.parity(str(a.out / "tokenizer.json"), ref, str(a.parity), a.parity_docs)
        print(json.dumps({"parity": par}))
        # The original gate: no round-trip failure, token counts within 0.5% of the reference.
        if par["roundtrip_failures_hf"] or abs(par["hf_tokens"] - par["ref_tokens"]) > 0.005 * max(1, par["ref_tokens"]):
            raise SystemExit("parity gate failed")


if __name__ == "__main__":
    main()
