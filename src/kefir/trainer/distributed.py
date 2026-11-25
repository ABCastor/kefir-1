"""Process-local distributed execution for one logical training run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def vote(self, value: bool, device: torch.device) -> bool:
        flag = torch.tensor(int(value), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def agree(self, values: tuple[int, ...], device: torch.device) -> None:
        local = torch.tensor(values, dtype=torch.int64, device=device)
        minimum, maximum = local.clone(), local.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.equal(minimum, maximum):
            raise RuntimeError("distributed ranks disagree on the training block")


_owns_group = False


def initialize_distributed(device: str | torch.device | None = None) -> DistributedContext | None:
    """Join a torchrun group; a normal Python invocation stays single process."""

    global _owns_group
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 1:
        raise RuntimeError("WORLD_SIZE must be positive")
    if world <= 1:
        if dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError("initialized process group has no matching torchrun environment")
        return None
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("distributed training requires RANK and LOCAL_RANK from torchrun")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if not 0 <= rank < world or local_rank < 0:
        raise RuntimeError("invalid distributed rank geometry")
    requested = torch.device(device).type if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA distributed training requires a visible CUDA device")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK has no visible CUDA device")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    elif requested == "cpu":
        backend = "gloo"
    else:
        raise RuntimeError("distributed training supports CPU and CUDA devices")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
        _owns_group = True
    if dist.get_rank() != rank or dist.get_world_size() != world:
        raise RuntimeError("torchrun and process-group ranks disagree")
    return DistributedContext(rank, local_rank, world)


def shutdown_distributed() -> None:
    global _owns_group
    if _owns_group and dist.is_initialized():
        dist.destroy_process_group()
    _owns_group = False


class FollowerCache:
    """Read a verified shard already staged by rank zero in shared storage."""

    def __init__(self, root: Path, *, timeout: float = 3 * 60 * 60) -> None:
        from kefir.data.incremental_frontier import _frontier_shards

        self.root = root
        contract = json.loads((root / "run_contract.json").read_text())
        frontier = json.loads((root / "shard_frontier.json").read_text())
        if (frontier.get("contract_sha256") != contract.get("contract_sha256")
                or frontier.get("run_id") != contract.get("run_id")
                or frontier.get("producer_complete") is not True):
            raise RuntimeError("follower corpus frontier is not a complete, matching run")
        self.planned_block_count = int(contract["planned_train_blocks"])
        self.shards = _frontier_shards(frontier, "ready_train_shards")
        if not self.shards or self.shards[-1].last_block_id < self.planned_block_count - 1:
            raise RuntimeError("follower corpus frontier does not cover the run")
        self.timeout = timeout
        self._verified: set[str] = set()

    def shard_for_block(self, block_id: int):
        from kefir.data.incremental_frontier import _train_index_for_block

        if not 0 <= block_id < self.planned_block_count:
            raise RuntimeError("follower block is outside the frozen corpus")
        index = _train_index_for_block(self.shards, block_id)
        if index is None:
            raise RuntimeError("follower block is absent from the verified frontier")
        return self.shards[index]

    def ensure_block(self, block_id: int) -> None:
        from kefir.data.incremental_frontier import _file_matches

        shard = self.shard_for_block(block_id)
        if shard.filename in self._verified:
            return
        deadline = time.monotonic() + self.timeout
        while not _file_matches(self.root, shard):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"rank zero did not provide shard {shard.filename}")
            time.sleep(1)
        self._verified.add(shard.filename)

    def acknowledge(self, block_id: int) -> None:
        self.shard_for_block(block_id)

    def restore_after_acknowledged(self, block_id: int) -> None:
        if block_id + 1 < self.planned_block_count:
            self.shard_for_block(block_id + 1)


__all__ = ["DistributedContext", "FollowerCache", "initialize_distributed", "shutdown_distributed"]
