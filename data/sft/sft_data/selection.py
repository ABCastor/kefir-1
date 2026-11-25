"""From labelled candidates to the training and validation conversations.

1. Filter flags: all five thresholds of a level must hold (strict or intermediate).
2. Overlap removal, in this order: a User message identical to a benchmark message (or,
   for the second wave, to an open-panel prompt); then family isolation, visiting
   conversations in a seeded hash order and keeping one only if it shares no isolation
   group and no first User message with a conversation already kept.
3. Stratified cohorts from the independent pool: Hamilton allocation by source x
   capability, then the best of 64 seeded draws for the margins of source, task, subject,
   length, turns, System presence and capability.

``membership`` reproduces the Kefir 1 recipe: the first labelling wave gives 10,000
training, 512 validation and 5,000 continuation conversations; every remaining
independent conversation of both waves forms the long continuation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math

from .questions import categories, scores
from .records import turn_bin

STRICT = {"english_only": .99, "english_quality": .90, "instruction_fidelity": .95,
          "content_error": .05, "excluded_formalism": .01}
INTERMEDIATE = {"english_only": .95, "english_quality": .80, "instruction_fidelity": .80,
                "content_error": .20, "excluded_formalism": .05}
AT_MOST = ("content_error", "excluded_formalism")  # lower is better; the others: at least
SEED = "kefir-sft-10k-strict-v1"
CONTINUATION_SEED = "kefir-sft-continuation-5k-v1"
TRIALS = 64


def passes(values: dict, thresholds: dict) -> bool:
    return all(math.isfinite(values[k]) and 0 <= values[k] <= 1 and
               (values[k] <= t if k in AT_MOST else values[k] >= t)
               for k, t in thresholds.items())


def flags(record) -> dict:
    s = scores(record.answers)
    return {"strict_pass": passes(s, STRICT), "intermediate_pass": passes(s, INTERMEDIATE)}


def rank(seed: str, record_id: str) -> bytes:
    return hashlib.sha256((seed + "\0" + record_id).encode()).digest()


def overlaps(record, keys) -> bool:
    return any(k in keys for k in record.user_keys())


def isolate(records, seen_groups: set, seen_prompts: set, seed: str = SEED):
    """Keep one conversation per family; updates the seen sets in place."""
    kept = []
    for record in sorted(records, key=lambda r: rank(seed + ":isolation", r.record_id)):
        first = record.user_keys()[0]
        if set(record.group_ids) & seen_groups or first in seen_prompts:
            continue
        seen_groups.update(record.group_ids)
        seen_prompts.add(first)
        kept.append(record)
    return kept


def length_bin(record) -> str:
    """Rendered-length bin of the sampling stratum: short <=512, medium <=1024, long."""
    return json.loads(record.sample["stratum"].split("|", 1)[0])[2]


def features(record) -> dict:
    s, c = scores(record.answers), categories(record.answers)
    return {"source": record.source["dataset"], "token_bin": length_bin(record),
            "turn_bin": turn_bin(record.turns), "system": "present" if record.has_system else "absent",
            "task": c["task"], "subject": c["subject"],
            "capability": ("R" if s["reasoning_required"] >= .8 else "-") +
                          ("C" if s["context_required"] >= .8 else "-")}


def cohort(records, count: int, seed: str):
    """Proportional design sample (not a probability sample); returns (chosen, margins)."""
    if len({r.record_id for r in records}) != len(records) or not 0 < count <= len(records):
        raise ValueError("invalid cohort or count")
    rows = sorted(records, key=lambda r: r.record_id)
    feats = {r.record_id: features(r) for r in rows}
    groups = defaultdict(list)
    for r in rows:
        groups[(feats[r.record_id]["source"], feats[r.record_id]["capability"])].append(r)
    allocation = {k: len(v) * count // len(rows) for k, v in groups.items()}
    spare = count - sum(allocation.values())
    for k in sorted(groups, key=lambda k: (-(len(groups[k]) * count % len(rows)), k))[:spare]:
        allocation[k] += 1
    population = Counter((k, v) for f in feats.values() for k, v in f.items())
    expected = {k: n * count / len(rows) for k, n in population.items()}
    best = None
    for trial in range(TRIALS):
        chosen = []
        for k, group in groups.items():
            chosen.extend(sorted(group, key=lambda r: rank(f"{seed}:{trial}", r.record_id))[:allocation[k]])
        observed = Counter((k, v) for r in chosen for k, v in feats[r.record_id].items())
        loss = sum((observed[k] - n) ** 2 / max(1, n) for k, n in expected.items())
        if best is None or loss < best[0]:
            best = (loss, trial, chosen, observed)
    loss, trial, chosen, observed = best
    margins = {"seed": seed, "trial": trial, "balance_loss": loss,
               "max_share_gap": max(abs(population[k] / len(rows) - observed[k] / count) for k in population)}
    return sorted(chosen, key=lambda r: rank(seed + ":final", r.record_id)), margins


def membership(records, benchmark_keys, panel_keys, *, first_wave="original",
               sizes=(10000, 512, 5000)) -> tuple[dict, dict]:
    """Return ({record_id: (split, segment)}, counts) for the two-wave Kefir 1 recipe."""
    counts = Counter()
    waves = defaultdict(list)
    for r in records:
        counts["labelled"] += 1
        if not flags(r)["strict_pass"]:
            continue
        counts["strict"] += 1
        wave = "first" if r.sample["wave"] == first_wave else "second"
        if overlaps(r, benchmark_keys):
            counts[wave + "_benchmark_overlap"] += 1
            continue
        waves[wave].append(r)
    groups, prompts = set(), set()
    pool = isolate(waves["first"], groups, prompts)
    counts["first_independent"] = len(pool)
    train_n, validation_n, continuation_n = sizes
    train, _ = cohort(pool, train_n, SEED + ":train")
    used = {r.record_id for r in train}
    validation, _ = cohort([r for r in pool if r.record_id not in used], validation_n, SEED + ":validation")
    used |= {r.record_id for r in validation}
    continuation, _ = cohort([r for r in pool if r.record_id not in used], continuation_n, CONTINUATION_SEED)
    used |= {r.record_id for r in continuation}
    old_unused = [r for r in pool if r.record_id not in used and not overlaps(r, panel_keys)]
    fresh = []
    for r in waves["second"]:
        if overlaps(r, panel_keys):
            counts["second_panel_overlap"] += 1
        else:
            fresh.append(r)
    new_pool = isolate(fresh, groups, prompts)
    counts["second_independent"] = len(new_pool)
    out = {r.record_id: ("train", "first") for r in train}
    out.update({r.record_id: ("validation", "validation") for r in validation})
    out.update({r.record_id: ("train", "continuation") for r in continuation})
    out.update({r.record_id: ("train", "long") for r in old_unused + new_pool})
    counts.update(Counter(split for split, _ in out.values()))
    return out, dict(counts)
