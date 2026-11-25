"""A small local caller of the public pretraining API (same path as `kefir pretrain`)."""

from __future__ import annotations

import argparse

from kefir import ExecutionSpec, RunSpec, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Scientific recipe JSON")
    parser.add_argument("execution", help="Hardware and segment JSON")
    parser.add_argument("--resume", help="Checkpoint ID or 'latest'")
    args = parser.parse_args()
    result = train(RunSpec.from_json(args.run), ExecutionSpec.from_json(args.execution),
                   resume=args.resume)
    print(f"{result.status}: step={result.step} tokens={result.consumed_tokens} "
          f"checkpoint={result.checkpoint}")


if __name__ == "__main__":
    main()
