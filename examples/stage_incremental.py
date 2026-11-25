"""Stage the checkpoint-aligned lead window before starting a GPU segment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kefir.data.incremental_stage import stage_incremental_window_when_ready
from kefir.data.src.hf_bucket_shards import HuggingFaceBucketShardStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--start-block", type=int, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    if args.start_block < 0:
        parser.error("--start-block must be non-negative")
    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"{args.token_env} is required")
    store = HuggingFaceBucketShardStore(args.bucket, token=token, create_bucket=False)
    result = stage_incremental_window_when_ready(
        store=store, run_id=args.run_id, destination=args.dataset_dir,
        start_block_id=args.start_block, ensure_bucket=False,
    )
    print(result)


if __name__ == "__main__":
    main()
