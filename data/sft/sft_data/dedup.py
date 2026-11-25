"""Deduplication and held-out isolation over all sources together.

Conversations are joined into families when they share an isolation group, have identical
content, or are near duplicates (Jaccard >= 0.9 over distinct case-folded 5-word shingles
of the role-tagged text, conversations of at least 24 words and 120 characters; every
candidate pair is verified exactly). Then:

- a family that contains any held-out conversation is removed whole;
- identical conversations keep one row (No Robots, then OpenAssistant, then SmolTalk, then
  the others; ties by ``record_id``) and the survivor inherits the others' groups;
- two different surviving conversations that are near duplicates are both rejected, unless
  they are branches of one OpenAssistant tree or one is a whole-message prefix of the other.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
import re

from .records import with_groups

PREFERENCE = {"no_robots": 0, "openassistant": 1, "smol_smoltalk": 2}


@dataclass(frozen=True)
class Near:
    threshold: float = 0.9
    shingle_words: int = 5
    min_words: int = 24
    min_chars: int = 120
    max_pairs: int = 10_000_000


def shingles(messages, near: Near) -> frozenset:
    text = "\n".join(m["role"] + ": " + m["content"] for m in messages)
    words = re.findall(r"\w+", text.casefold(), re.UNICODE)
    if len(words) < near.min_words or len(text) < near.min_chars:
        return frozenset()
    out = frozenset(tuple(words[i:i + near.shingle_words]) for i in range(len(words) - near.shingle_words + 1))
    return out if len(out) >= near.min_words - near.shingle_words + 1 else frozenset()


class _Union:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def near_pairs(items: dict, near: Near):
    """Verified near-duplicate pairs; an exact prefix filter, never a probabilistic one."""
    sets = {rid: shingles(r.messages, near) for rid, r in items.items()}
    inverted = defaultdict(list)
    for rid, sh in sets.items():
        for s in sh:
            inverted[s].append(rid)
    pairs = set()
    for rid, sh in sets.items():
        if not sh:
            continue
        size = len(sh)
        prefix = sorted(sh, key=lambda s: (len(inverted[s]), s))[:size - math.ceil(near.threshold * size - 1e-12) + 1]
        for s in prefix:
            for other in inverted[s]:
                n = len(sets[other])
                if other != rid and min(n, size) / max(n, size) >= near.threshold:
                    pairs.add(tuple(sorted((rid, other))))
                    if len(pairs) > near.max_pairs:
                        raise RuntimeError("near-duplicate pair budget exhausted; no partial result")
    return [(a, b) for a, b in sorted(pairs)
            if len(sets[a] & sets[b]) / len(sets[a] | sets[b]) >= near.threshold]


def _is_prefix(a, b) -> bool:
    short, long = sorted((a.messages, b.messages), key=len)
    return len(short) < len(long) and all(x == y for x, y in zip(short, long))


def deduplicate(records, near: Near = Near()) -> tuple[list, dict]:
    """Return (training candidates, counts); held-out records only isolate."""
    items = {r.record_id: r for r in records}
    if len(items) != len(records):
        raise ValueError("duplicate record_id")
    union, first_by_group, first_by_hash = _Union(items), {}, {}
    for rid in sorted(items):
        r = items[rid]
        for g in r.group_ids:
            union.union(rid, first_by_group.setdefault(g, rid))
        union.union(rid, first_by_hash.setdefault(r.content_hash, rid))
    pairs = near_pairs(items, near)
    for a, b in pairs:
        union.union(a, b)
    families = defaultdict(list)
    for rid in sorted(items):
        families[union.find(rid)].append(rid)
    counts = defaultdict(int)
    kept = {}
    for members in families.values():
        if any(items[r].source["split"] != "train" for r in members):
            counts["heldout_family_removed"] += sum(items[r].source["split"] == "train" for r in members)
            continue
        by_hash = defaultdict(list)
        for r in members:
            by_hash[items[r].content_hash].append(r)
        for same in by_hash.values():
            same.sort(key=lambda r: (PREFERENCE.get(items[r].source["family"], 99), r))
            survivor = items[same[0]]
            aliases = [{"record_id": r, "dataset": items[r].source["dataset"]} for r in same[1:]]
            for r in same[1:]:
                survivor = with_groups(survivor, items[r].group_ids)
            if aliases:
                survivor = replace(survivor, lineage={**survivor.lineage, "aliases": aliases})
            kept[survivor.record_id] = survivor
            counts["exact_duplicate_removed"] += len(same) - 1
    rejected = set()
    for a, b in pairs:
        if a in kept and b in kept and kept[a].content_hash != kept[b].content_hash:
            same_tree = {g for g in kept[a].group_ids if g.startswith("oasst-tree:")} & set(kept[b].group_ids)
            if not same_tree and not _is_prefix(kept[a], kept[b]):
                rejected.update((a, b))
    counts["near_duplicate_rejected"] = len(rejected)
    out = [kept[r] for r in sorted(kept) if r not in rejected]
    counts["candidates"] = len(out)
    return out, dict(counts)
