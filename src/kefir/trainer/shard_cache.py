"""Construct the optional rolling shard cache for a prepared corpus."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping


def rolling_cache(args: object) -> object | None:
    bucket_id = getattr(args, "dataset_shard_bucket", None)
    if not bucket_id:
        return None
    run_id = getattr(args, "dataset_shard_run_id", None)
    manifest_path = getattr(args, "dataset_manifest", None)
    if not run_id or manifest_path is None:
        raise RuntimeError("rolling dataset shards require a run ID and manifest")
    token_env = getattr(args, "dataset_shard_token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is required for rolling dataset shard reads")
    prefetch = getattr(args, "dataset_shard_prefetch", 1)
    if isinstance(prefetch, bool) or prefetch < 1:
        raise RuntimeError("rolling dataset shard prefetch must be at least one")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("rolling dataset manifest must contain a JSON object")
    production = payload.get("production")
    if not isinstance(production, Mapping) or production.get("run_id") != run_id:
        raise RuntimeError("rolling dataset manifest run ID mismatch")

    from kefir.data.src.hf_bucket_shards import HuggingFaceBucketShardStore

    store = HuggingFaceBucketShardStore(bucket_id, token=token, create_bucket=False)
    incremental = payload.get("incremental_frontier")
    if isinstance(incremental, Mapping):
        from kefir.data.incremental_cache import IncrementalRollingShardCache
        from kefir.data.incremental_frontier import RUN_CONTRACT_FILENAME

        contract_path = manifest_path.parent / RUN_CONTRACT_FILENAME
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError("incremental rolling dataset has no readable run contract") from error
        if not isinstance(contract, Mapping):
            raise RuntimeError("incremental rolling dataset run contract is not an object")
        if contract.get("run_id") != run_id:
            raise RuntimeError("incremental rolling dataset run contract ID mismatch")
        if incremental.get("contract_sha256") != contract.get("contract_sha256"):
            raise RuntimeError("incremental rolling dataset manifest/contract identity mismatch")
        return IncrementalRollingShardCache(
            root=args.dataset_dir,
            run_id=run_id,
            contract=contract,
            store=store,
            prefetch_shards=prefetch,
            wait_timeout_seconds=getattr(args, "dataset_shard_wait_timeout_seconds", 0.0),
        )

    from kefir.data.rolling_cache import RollingShardCache

    return RollingShardCache(
        root=args.dataset_dir,
        run_id=run_id,
        manifest=payload,
        store=store,
        prefetch_shards=prefetch,
        evict_consumed=True,
    )


__all__ = ["rolling_cache"]
