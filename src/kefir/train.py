"""Run a Kefir segment from local JSON configuration or under torchrun."""

from __future__ import annotations

import argparse
import json
import os

from .training import ExecutionSpec, RunSpec, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kefir pretrain", description="Train or resume Kefir 1 pretraining")
    parser.add_argument("--run", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    result = train(RunSpec.from_json(args.run), ExecutionSpec.from_json(args.execution),
                   resume=args.resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"training_result": {
            "status": result.status, "run_id": result.run_id, "step": result.step,
            "consumed_tokens": result.consumed_tokens, "checkpoint": str(result.checkpoint),
        }}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
