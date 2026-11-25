#!/usr/bin/env python3
"""Filter English prose into the two tokenizer training files, stage1.jsonl and stage2.jsonl.

Input: JSONL files with a 'text' field, validation files and training files separately.
For each document, in order (validation files first, then training files):

1. `filters.ProseFilter` (english-prose-v1): whole-document rejection of code, markup, LaTeX,
   symbolic maths, equations, programming topics, PII patterns, special-token literals,
   control characters, short or repetitive text, non-Latin text, link-heavy text, and any
   2,000-character chunk that fastText does not label English with p >= 0.90;
2. exact deduplication on the SHA-256 of the NFKC-casefolded, whitespace-collapsed text,
   which is also the document id;
3. training documents sharing a paragraph of 40+ words with an accepted validation document
   are dropped; validation keeps at most 1,000 documents per primary keyword topic;
4. the english-prose-v2 refinement (`V2_PATTERNS` below) drops what survived with function
   notation, inline assignments, numeric formulas or ASCII logic.

stage1.jsonl holds every surviving training document in input order. stage2.jsonl holds the
documents whose id satisfies int(id[:8], 16) % 5 == 0, in the same order, until 1,000,000,000
UTF-8 bytes. Validation documents never enter either stage.

    python prepare_seed.py --lid lid.176.bin --validation heldout/*.jsonl --train train/*.jsonl --out seed/
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from filters import ProseFilter, normalized, VERSION

STAGE2_MAX_UTF8_BYTES = 1_000_000_000
VALIDATION_MAX_DOCS_PER_TOPIC = 1000
V2_VERSION = "english-prose-v2"
V2_PATTERNS = {
    "function_notation": re.compile(r"\b[A-Za-z]\([A-Za-z0-9, ]{1,20}\)"),
    "inline_assignment": re.compile(r"\b\w+\s*\+=|\b\w+\s*-=|\b(?:return|yield)\s+[a-zA-Z_][\w.]*\s*[+*/-]"),
    "numeric_formula": re.compile(r"\b\d+\s*(?:\*\*|[×÷])\s*\d+"),
    "ascii_symbolic_logic": re.compile(r"\b[A-Za-z]\s*(?:<->|->|[&|])\s*[A-Za-z]\b"),
}


def digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def paragraphs(text: str) -> set[str]:
    return {digest(normalized(p)) for p in text.split("\n") if len(p.split()) >= 40}


def v2_reason(text: str) -> str | None:
    return next((name for name, p in V2_PATTERNS.items() if p.search(text)), None)


def stage2_selected(doc_id: str) -> bool:
    return int(doc_id[:8], 16) % 5 == 0


def stage_line(text: str) -> str:
    return json.dumps({"text": text}, ensure_ascii=False) + "\n"


def build(validation: list[Path], train: list[Path], out: Path, prose_filter) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    eval_paragraphs: set[str] = set()
    val_topics: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    n1 = n2 = b1 = b2 = 0
    with (out / "validation.jsonl").open("w", encoding="utf-8") as val_out, \
         (out / "stage1.jsonl").open("w", encoding="utf-8") as s1, \
         (out / "stage2.jsonl").open("w", encoding="utf-8") as s2:
        for split, paths in (("validation", validation), ("train", train)):
            for path in paths:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        text = json.loads(line)["text"]
                        counts[split + "_input"] += 1
                        reason, info = prose_filter.assess(text)
                        if not reason:
                            h = digest(normalized(text))
                            if h in seen:
                                reason = "exact_duplicate"
                            elif split == "train" and paragraphs(text) & eval_paragraphs:
                                reason = "validation_paragraph_overlap"
                            elif split == "validation" and val_topics[info["topics"][0]] >= VALIDATION_MAX_DOCS_PER_TOPIC:
                                reason = "validation_bucket_cap"
                        if reason:
                            counts["reject_" + reason] += 1
                            continue
                        # v1-accepted documents count for deduplication, overlap and caps
                        # even if the v2 refinement drops them, as in the original two passes.
                        seen.add(h)
                        if split == "validation":
                            eval_paragraphs.update(paragraphs(text))
                            val_topics[info["topics"][0]] += 1
                        reason = v2_reason(text)
                        if reason:
                            counts["reject_v2_" + reason] += 1
                            continue
                        counts[split + "_accepted"] += 1
                        if split == "validation":
                            val_out.write(json.dumps({"text": text, "id": h, **info}, ensure_ascii=False) + "\n")
                            continue
                        size = len(text.encode())
                        s1.write(stage_line(text))
                        n1 += 1
                        b1 += size
                        if stage2_selected(h) and b2 + size <= STAGE2_MAX_UTF8_BYTES:
                            s2.write(stage_line(text))
                            n2 += 1
                            b2 += size
    receipt = {"filter": VERSION, "refinement": V2_VERSION, "counts": dict(counts),
               "stage1_docs": n1, "stage1_utf8_bytes": b1, "stage2_docs": n2, "stage2_utf8_bytes": b2}
    (out / "seed-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lid", type=Path, required=True, help="fastText lid.176.bin")
    ap.add_argument("--validation", type=Path, nargs="*", default=[])
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import fasttext
    receipt = build(a.validation, a.train, a.out, ProseFilter(fasttext.load_model(str(a.lid))))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
