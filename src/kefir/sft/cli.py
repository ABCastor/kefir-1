"""``kefir sft render|train`` and ``kefir evaluate chat``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def render_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kefir sft render",
                                description="Render {conversation_id, messages} JSONL into token rows")
    p.add_argument("--conversations", required=True)
    p.add_argument("--output", required=True, help="new token-row JSONL file")
    a = p.parse_args(argv)
    from .data import render_rows
    records = (json.loads(line) for line in Path(a.conversations).read_text().splitlines() if line.strip())
    print(json.dumps(render_rows(records, a.output)))
    return 0


def train_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kefir sft train", description="Fine-tune Kefir 1 on whole conversations")
    p.add_argument("--recipe", required=True, help="scientific recipe JSON")
    p.add_argument("--execution", required=True, help="hardware/output JSON")
    p.add_argument("--resume", help="checkpoint directory (exposure-NNNNN) to resume or fork from")
    a = p.parse_args(argv)
    from .recipe import Execution, Recipe
    from .train import is_primary, train
    recipe, base = Recipe.from_json(a.recipe)
    result = train(recipe, base, Execution.from_json(a.execution), resume=a.resume)
    if is_primary():
        print(json.dumps({"sft_result": result}, sort_keys=True), flush=True)
    return 0


def main(argv=None) -> int:
    commands = {"render": render_main, "train": train_main}
    argv = list(argv) if argv is not None else []
    if not argv or argv[0] not in commands:
        print("usage: kefir sft {render,train} ...", file=sys.stderr)
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    return commands[argv[0]](argv[1:])


def evaluate_chat_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kefir evaluate chat",
                                description="Frozen chat suite, raw retention and held-out NLL for SFT weights")
    p.add_argument("--weights", required=True, help=".safetensors weights (or a trusted trainer state)")
    p.add_argument("--trusted-state", action="store_true")
    p.add_argument("--benchmark", required=True, help="suite built by `kefir bench build`")
    p.add_argument("--split", choices=("dev", "test", "reserve"), default="dev")
    p.add_argument("--confirm-held-out-split", action="store_true",
                   help="required for test/reserve: they are not for recipe tuning")
    p.add_argument("--validation", help="held-out token rows for Assistant NLL")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--output", required=True, help="new directory")
    a = p.parse_args(argv)
    if a.split != "dev" and not a.confirm_held_out_split:
        p.error("test/reserve require --confirm-held-out-split")
    from .evaluation import check_precision
    try:
        check_precision(a.precision, a.device)
    except ValueError as error:
        p.error(str(error))
    from kefir.artifact import load_weights
    from kefir.bench.build import load_split
    from kefir.staging import staged_output
    from kefir.bench.io import write_jsonl
    from .data import read_rows
    from .evaluation import evaluate_model
    with staged_output(a.output) as out:
        model, _, info = load_weights(a.weights, device=a.device, trusted=a.trusted_state)
        result = evaluate_model(model, cases=load_split(a.benchmark, a.split),
                                validation_rows=read_rows(a.validation) if a.validation else None,
                                precision=a.precision, device=a.device)
        out.mkdir()
        write_jsonl(out / "responses.jsonl", result["responses"])
        write_jsonl(out / "scored.jsonl", result["scored"])
        summary = {**result["summary"], "split": a.split, "weights_sha256": info["sha256"]}
        (out / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if result["retention"] is not None:
            (out / "raw-retention.json").write_text(
                json.dumps(result["retention"], indent=2, allow_nan=False) + "\n")
    print(json.dumps({"benchmark": summary["benchmark"].get("categories", summary["benchmark"]),
                      "validation": summary["validation"]}, allow_nan=False))
    return 0
