"""Stratified candidate sampling before labelling.

Base cells are source x Assistant-turn bin (1, 2-3, 4-6, 7+) x rendered-length bin
(<=512, <=1024, >1024 positions) x System presence. Supported cells (at least 40 members)
are split further by available source metadata: native category and source, instruction
constraints, threaded-source ratings, alias origins and message-length shape. A split keeps
only children of at least 20 members, puts the rest in a residual child, uses each
dimension at most 6 times and stops at 600 cells. Missing metadata is its own value.

Each cell gets min(2, size) conversations, the rest proportionally (largest remainders);
within a cell the lowest seeded hashes are taken. ``inclusion_probability`` is the cell's
sample share. Nested milestones (for example 25k, 50k, 75k, 100k) allocate each increment
from what remains, so every milestone prefix is itself a stratified sample.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math

from .records import turn_bin

MAX_CELLS, MAX_SPLITS, MIN_PARENT, MIN_CHILD = 600, 6, 40, 20


def length_bin(rendered_tokens: int) -> str:
    return "short" if rendered_tokens <= 512 else "medium" if rendered_tokens <= 1024 else "long"


def stratum(record) -> str:
    return json.dumps([record.source["dataset"], turn_bin(record.turns),
                       length_bin(record.rendered_tokens), record.has_system], separators=(",", ":"))


def _compact(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= 120 else "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _constraints(value):
    if isinstance(value, list):
        for item in value:
            yield from _constraints(item)
    elif isinstance(value, dict):
        key = next((k for k in ("instruction_id", "type", "name", "constraint_type")
                    if isinstance(value.get(k), str)), None)
        if key:
            yield _compact(value[key])
        elif isinstance(value.get("instruction_id_list"), list):
            for item in value["instruction_id_list"]:
                yield _compact(item)
        else:
            for k in value:
                yield "field:" + k
    elif isinstance(value, str):
        yield _compact(value)


def metadata(record) -> dict:
    """Refinement values from source metadata; absent fields are simply not present."""
    lineage, native, out = record.lineage, record.lineage.get("native", {}), {}
    for key, value in (("native_source", record.source.get("native_source")),
                       ("native_category", lineage.get("native_category")),
                       ("category", native.get("category"))):
        if value is not None:
            out[key] = _compact(value)
    constraints = sorted(set(_constraints(native.get("constraints"))))
    out["constraint_count"] = str(min(len(constraints), 5))
    out["constraint_signature"] = _compact(constraints)
    for c in constraints:
        out["constraint:" + c] = "present"
    values = defaultdict(list)
    for item in lineage.get("native_metadata", []):
        role = item.get("role", "unknown")
        for key in ("lang", "review_result", "deleted", "rank", "synthetic"):
            if item.get(key) is not None:
                values[f"{role}/{key}"].append(item[key])
        labels = item.get("labels")
        pairs = (zip(labels.get("name", []), labels.get("value", [])) if isinstance(labels, dict) else
                 [(x.get("name"), x.get("value")) for x in labels if isinstance(x, dict)]
                 if isinstance(labels, list) else [])
        for key, value in pairs:
            if key is not None and isinstance(value, (int, float)):
                values[f"{role}/label/{key}"].append(value)
    for key, vs in values.items():
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vs):
            out["native/" + key] = str(round(sum(vs) / len(vs) * 4) / 4)
        else:
            out["native/" + key] = _compact(sorted(set(map(str, vs))))
    aliases = lineage.get("aliases", [])
    out["origin_datasets"] = _compact(sorted({record.source["dataset"]} | {a["dataset"] for a in aliases}))
    out["alias_count"] = str(min(len(aliases), 5))
    lengths = [len(m["content"]) for m in record.messages if m["role"] != "system"]
    out["max_message_chars_bin"] = str(next((b for b in (128, 512, 2048) if max(lengths) <= b), 8192))
    out["review_unresolved"] = str(lineage.get("review_unresolved", False))
    return out


def _dimension_order(key):
    tier = (0 if key in ("native_source", "native_category", "category", "origin_datasets") else
            1 if key in ("constraint_count", "constraint_signature", "max_message_chars_bin",
                         "alias_count", "review_unresolved") else
            2 if key.startswith("native/") else 3)
    return tier, key


def cells(records) -> dict:
    """Map cell name -> record indices, refining supported base cells by metadata."""
    items = [(stratum(r), metadata(r)) for r in records]
    groups = defaultdict(list)
    for i, (base, _) in enumerate(items):
        groups[base].append(i)
    used = Counter()
    for dim in sorted({k for _, f in items for k in f}, key=_dimension_order):
        for parent, members in sorted(list(groups.items()), key=lambda x: (-len(x[1]), x[0])):
            if used[dim] >= MAX_SPLITS:
                break
            if len(members) < MIN_PARENT:
                continue
            bins = defaultdict(list)
            for i in members:
                bins[items[i][1].get(dim, "MISSING")].append(i)
            if len(bins) < 2:
                continue
            children = {k: v for k, v in bins.items() if len(v) >= MIN_CHILD}
            rest = [i for v in bins.values() if len(v) < MIN_CHILD for i in v]
            if rest:
                children["RESIDUAL"] = rest
            if len(children) < 2 or len(groups) + len(children) - 1 > MAX_CELLS:
                continue
            del groups[parent]
            for value, ids in children.items():
                groups[f"{parent}|{dim}={value}"] = ids
            used[dim] += 1
    return dict(groups)


def allocation(counts: dict, n: int) -> dict:
    if n > sum(counts.values()):
        raise ValueError("sample larger than population")
    quota = {k: min(2, v) for k, v in counts.items()}
    if sum(quota.values()) > n:
        raise ValueError("cell minimums exceed the sample")
    left, remaining = n - sum(quota.values()), sum(counts[k] - quota[k] for k in counts)
    if left and remaining:
        ideal = {k: left * (counts[k] - quota[k]) / remaining for k in counts}
        for k in quota:
            quota[k] += math.floor(ideal[k])
        for k in sorted(quota, key=lambda k: (-(ideal[k] - math.floor(ideal[k])), k))[:n - sum(quota.values())]:
            quota[k] += 1
    return quota


def nested_quotas(counts: dict, milestones) -> dict:
    quotas, result, previous = {k: 0 for k in counts}, {}, 0
    for milestone in milestones:
        if milestone <= previous:
            raise ValueError("milestones must increase")
        step = allocation({k: counts[k] - quotas[k] for k in counts}, milestone - previous)
        quotas = {k: quotas[k] + step[k] for k in counts}
        result[milestone], previous = quotas, milestone
    return result


def _rank(seed, record_id):
    return hashlib.sha256((seed + "\0" + record_id).encode()).digest()


def draw(records, milestones, *, seed: str, wave: str):
    """Return the sampled records in labelling order, each with its ``sample`` facts."""
    records = list(records)
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("duplicate record identity")
    groups = cells(records)
    counts = {k: len(v) for k, v in groups.items()}
    quotas = nested_quotas(counts, milestones)
    last = quotas[milestones[-1]]
    ranked = {g: sorted(ix, key=lambda i: (_rank(seed, records[i].record_id), records[i].record_id))[:last[g]]
              for g, ix in groups.items()}
    order, previous = [], {g: 0 for g in groups}
    for milestone in milestones:
        batch = []
        for g, ix in ranked.items():
            chosen = ix[previous[g]:quotas[milestone][g]]
            batch += [((j + .5) / len(chosen), _rank(seed + "-interleave", records[i].record_id), i, g)
                      for j, i in enumerate(chosen)]
            previous[g] = quotas[milestone][g]
        order += sorted(batch)
    return [replace(records[i], sample={"wave": wave, "stratum": g,
                                        "inclusion_probability": last[g] / counts[g]})
            for _, _, i, g in order]
