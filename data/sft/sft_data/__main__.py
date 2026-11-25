"""Command line for the fine-tuning data recipe: ``PYTHONPATH=data/sft python -m sft_data <command>``.

Commands run in order; each reads JSON Lines of ``Record`` and writes a new file (an
existing output is refused, never overwritten):

  sources     print the pinned sources and the download commands
  prepare     source files -> candidates (2,048-token cap, source domain, deduplication)
  prefilter   candidates -> retained (scope rules, then fastText + Lingua)
  sample      retained -> stratified sample, excluding earlier samples
  label       sample -> labelled, asking the ten questions to a judge
  select      labelled -> membership (strict filter, overlap removal, cohorts)
  check       recompute identities, scores, flags (and membership) of the published pool
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from .records import Record

DEFAULT_KEYS = Path(__file__).with_name("assets") / "overlap-keys.json"


def read(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                yield Record.from_json(line)


class Output:
    """A new file written whole: a temporary name, renamed only on success."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise SystemExit(f"refusing to overwrite {self.path}")
        self.tmp = self.path.with_name(self.path.name + ".partial")

    def __enter__(self):
        self.stream = self.tmp.open("x", encoding="utf-8")
        return self

    def write(self, record: Record):
        self.stream.write(record.to_json() + "\n")

    def __exit__(self, kind, *_):
        self.stream.close()
        if kind is None:
            os.replace(self.tmp, self.path)
        else:
            self.tmp.unlink()
        return False


def report(path, value):
    text = json.dumps(value, indent=2, sort_keys=True)
    if path:
        Output(path)  # same refusal rule
        Path(path).write_text(text + "\n")
    print(text)


def cmd_sources(args):
    from .sources import SOURCES
    for dataset, s in SOURCES.items():
        files = " ".join(s.train + s.heldout)
        print(f"hf download {dataset} {files} --repo-type dataset --revision {s.revision} "
              f"--local-dir {args.root}/{dataset}    # {s.licence}")


def cmd_prepare(args):
    from kefir.sft.renderer import ChatRenderer, ContextOverflow
    from .dedup import deduplicate
    from .prefilter import source_domain
    from .sources import SOURCES, ancestor, load
    renderer, counts, records = ChatRenderer.load(), Counter(), []

    def fit(record):
        lengths = [len(record.messages)]
        if record.source["family"] == "openassistant":
            lengths += list(range(len(record.messages) - 2, 1, -2))
        for keep in lengths:
            candidate = record if keep == len(record.messages) else ancestor(record, keep)
            try:
                tokens = len(renderer.render(list(candidate.messages))["token_ids"])
            except ContextOverflow:
                continue
            return replace(candidate, rendered_tokens=tokens)
        return None

    for dataset in args.sources or list(SOURCES):
        for record, split in load(Path(args.root), dataset):
            counts[f"{dataset}:{split}:rows"] += 1
            if record is None:
                counts[f"{dataset}:{split}:rejected_structure"] += 1
            elif split != "train":
                records.append(record)
            elif source_domain(record):
                counts[f"{dataset}:source_domain"] += 1
            elif (fitted := fit(record)) is None:
                counts[f"{dataset}:over_2048"] += 1
            else:
                counts[f"{dataset}:ancestor_recovered"] += fitted.record_id != record.record_id
                records.append(fitted)
    candidates, dedup_counts = deduplicate(records)
    with Output(args.out) as out:
        for record in candidates:
            out.write(record)
    report(args.report, {"rows": dict(counts), "dedup": dedup_counts})


def cmd_prefilter(args):
    from .prefilter import exclusion, language_detectors
    check = None if args.no_language else language_detectors(args.fasttext)
    counts = Counter()
    with Output(args.out) as out:
        for record in read(args.inputs):
            reason = exclusion(record, check)
            counts[(reason or "retained").split(":")[0]] += 1
            if reason is None:
                out.write(record)
    report(args.report, dict(counts))


def cmd_sample(args):
    from .sample import draw
    excluded = {r.record_id for r in read(args.exclude or [])}
    population = [r for r in read(args.inputs) if r.record_id not in excluded]
    milestones = tuple(int(x) for x in args.milestones.split(","))
    chosen = draw(population, milestones, seed=args.seed, wave=args.wave)
    with Output(args.out) as out:
        for record in chosen:
            out.write(record)
    report(args.report, {"population": len(population), "excluded": len(excluded), "sample": len(chosen)})


def cmd_label(args):
    from .judge import ChatJudge
    from .questions import QUESTIONS, state
    judge = ChatJudge(args.base_url, args.model, api_key_env=args.api_key_env)
    done = 0
    with Output(args.out) as out:
        for record in read(args.inputs):
            out.write(replace(record, answers=judge.answer(state(list(record.messages)), QUESTIONS)))
            done += 1
    report(args.report, {"labelled": done})


def cmd_select(args):
    from .selection import membership
    keys = json.loads(Path(args.overlap_keys).read_text())
    sizes = tuple(int(x) for x in args.sizes.split(","))
    chosen, counts = membership(list(read(args.inputs)), set(keys["benchmark"]), set(keys["open_panel"]),
                                first_wave=args.first_wave, sizes=sizes)
    Output(args.out)
    Path(args.out).write_text(json.dumps({rid: {"split": s, "segment": g} for rid, (s, g) in sorted(chosen.items())},
                                         indent=0) + "\n")
    report(args.report, counts)


def cmd_check(args):
    from .check_pool import check
    report(args.report, check(args.parquet, with_membership=args.membership, overlap_keys=args.overlap_keys))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m sft_data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, func, help):
        p = sub.add_parser(name, help=help)
        p.set_defaults(func=func)
        p.add_argument("--report", help="also write the JSON report to this new file")
        return p

    p = command("sources", cmd_sources, "pinned sources and download commands")
    p.add_argument("--root", default="data/sft/sources")
    p = command("prepare", cmd_prepare, "source files -> candidates")
    p.add_argument("--root", required=True, help="directory with <dataset>/<file> at the pinned revisions")
    p.add_argument("--sources", nargs="*", help="subset of datasets (default: all eight)")
    p.add_argument("--out", required=True)
    p = command("prefilter", cmd_prefilter, "scope and language pre-filters")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--fasttext", help="path to lid.176.bin")
    p.add_argument("--no-language", action="store_true", help="scope rules only (tests, dry runs)")
    p = command("sample", cmd_sample, "stratified candidate sample")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--exclude", nargs="*", help="earlier samples whose conversations are not drawn again")
    p.add_argument("--milestones", required=True, help="e.g. 25000,50000,75000,100000")
    p.add_argument("--seed", required=True)
    p.add_argument("--wave", required=True, help="labelling wave name, e.g. original or extension")
    p.add_argument("--out", required=True)
    p = command("label", cmd_label, "ask the ten questions to an OpenAI-compatible judge")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", help="name of the environment variable holding the key")
    p.add_argument("--out", required=True)
    p = command("select", cmd_select, "filter flags, overlap removal, cohorts")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--first-wave", default="original")
    p.add_argument("--sizes", default="10000,512,5000", help="first training, validation, continuation")
    p.add_argument("--overlap-keys", default=str(DEFAULT_KEYS))
    p = command("check", cmd_check, "reproduce the published pool's fields")
    p.add_argument("parquet", nargs="+")
    p.add_argument("--membership", action="store_true", help="rerun selection (needs all 20 parts)")
    p.add_argument("--overlap-keys", default=str(DEFAULT_KEYS))
    args = parser.parse_args(argv)
    if args.command == "prefilter" and not (args.fasttext or args.no_language):
        parser.error("prefilter needs --fasttext PATH (or --no-language)")
    try:
        args.func(args)
    except ValueError as error:  # invalid input or request: nothing was written
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
