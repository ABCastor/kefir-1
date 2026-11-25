"""Reproduction check against the published labelled pool.

For every row of the given Parquet parts it recomputes, from the row's own fields:
``record_id`` and ``content_hash`` (from source, lineage and messages), the eight scores
(from ``answers_json``), and ``strict_pass`` / ``intermediate_pass``, and compares them
with the published values. With ``--membership`` (all 20 parts, 200,000 rows) it also
reruns the selection and compares the train / validation / unused membership.

The pool does not store the groups that deduplication inherited from dropped duplicates
(about 1 % of selected rows had some); the membership rerun rebuilds groups from source,
lineage and first prompt only, so a small disagreement there is expected and reported.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from .questions import SCHEMA_SHA256, scores
from .records import make_record
from .selection import flags, membership

COLUMNS = ("record_id", "messages", "source_json", "lineage_json", "content_hash", "wave", "stratum",
           "inclusion_probability", "model_id", "schema_hash", "answers_json", "scores_json",
           "strict_pass", "intermediate_pass", "membership")


def rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        yield from pq.read_table(path, columns=list(COLUMNS)).to_pylist()


def check(paths, *, with_membership=False, overlap_keys=None) -> dict:
    counts, mismatches, records, published = Counter(), Counter(), [], {}
    for row in rows(paths):
        counts["rows"] += 1
        record = make_record(json.loads(row["source_json"]), row["messages"], json.loads(row["lineage_json"]))
        mismatches["record_id"] += record.record_id != row["record_id"]
        mismatches["content_hash"] += record.content_hash != row["content_hash"]
        mismatches["schema_hash"] += row["schema_hash"] != SCHEMA_SHA256
        record = replace(record, answers=json.loads(row["answers_json"]),
                         sample={"wave": row["wave"], "stratum": row["stratum"],
                                 "inclusion_probability": row["inclusion_probability"]})
        mismatches["scores"] += scores(record.answers) != json.loads(row["scores_json"])
        f = flags(record)
        mismatches["strict_pass"] += f["strict_pass"] != row["strict_pass"]
        mismatches["intermediate_pass"] += f["intermediate_pass"] != row["intermediate_pass"]
        mismatches["selected_but_not_strict"] += row["membership"] != "unused" and not row["strict_pass"]
        counts.update({"strict": f["strict_pass"], "intermediate": f["intermediate_pass"],
                       "membership:" + row["membership"]: 1, "model:" + row["model_id"]: 1})
        published[row["record_id"]] = row["membership"]
        if with_membership:
            records.append(record)
    report = {"parts": [Path(p).name for p in paths], "counts": dict(counts), "mismatches": dict(mismatches)}
    if with_membership:
        keys = json.loads(Path(overlap_keys).read_text())
        chosen, selection_counts = membership(records, set(keys["benchmark"]), set(keys["open_panel"]))
        recomputed = {rid: chosen.get(rid, ("unused",))[0] for rid in published}
        confusion = Counter((published[rid], recomputed[rid]) for rid in published)
        report["membership"] = {"selection_counts": selection_counts,
                                "published_vs_recomputed": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
                                "agreement": sum(n for (a, b), n in confusion.items() if a == b) / len(published)}
    return report
