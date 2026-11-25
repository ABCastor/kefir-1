"""Exact token accounting and immutable shard writing for the corpus003 packer."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping

from kefir.data import config
from .storage import sha256_file

STREAM_CACHE_SCHEMA_VERSION = 2
SEQUENCE_FORMAT = "context_plus_one"


def _lcm(left: int, right: int) -> int:
    return left // math.gcd(left, right) * right


def normalize_cluster_weights(weights: Mapping[int | str, object]) -> tuple[dict[int, str], dict[int, int]]:
    """Validate all accepted-cluster weights and return supplied and exact units.

    A mapping must contain *only* the accepted clusters.  In particular, cluster
    11 is not a harmless zero: including it is rejected so an accidental
    production configuration cannot conceal a policy error.
    """

    supplied: dict[int, str] = {}
    fractions: dict[int, Fraction] = {}
    for raw_cluster, raw_weight in weights.items():
        cluster = int(raw_cluster)
        if cluster in supplied:
            raise ValueError(f"duplicate cluster weight {cluster}")
        if isinstance(raw_weight, bool):
            raise ValueError(f"cluster {cluster} weight must be a positive finite number")
        try:
            value = Fraction(str(raw_weight))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"cluster {cluster} has invalid weight {raw_weight!r}") from error
        if value <= 0:
            raise ValueError(f"cluster {cluster} weight must be positive")
        supplied[cluster] = str(raw_weight)
        fractions[cluster] = value
    expected = set(config.ACCEPTED_CLUSTER_IDS)
    actual = set(fractions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "cluster weights must contain exactly accepted clusters "
            f"(missing={missing}, extra={extra}; cluster 11 is excluded)"
        )
    denominator = 1
    for value in fractions.values():
        denominator = _lcm(denominator, value.denominator)
    units = {cluster: int(value * denominator) for cluster, value in fractions.items()}
    divisor = 0
    for value in units.values():
        divisor = math.gcd(divisor, value)
    units = {cluster: value // divisor for cluster, value in sorted(units.items())}
    return dict(sorted(supplied.items())), units


@dataclass(frozen=True)
class SourceDocument:
    """A validated accepted document with a source identity stable across runs."""

    source_id: str
    cluster_id: int
    tokens: tuple[int, ...]
    work_item_index: int = 0
    record_start: int = 0

    @property
    def source_token_count(self) -> int:
        return len(self.tokens)


class TokenDeficitScheduler:
    """Whole-document, exact-integer largest-deficit scheduler."""

    def __init__(self, weights: Mapping[int | str, object], tie_break_seed: str) -> None:
        self.supplied_weights, self.weight_units = normalize_cluster_weights(weights)
        self.weight_total = sum(self.weight_units.values())
        self.tie_break_seed = tie_break_seed
        self.emitted_source_tokens = {cluster: 0 for cluster in self.weight_units}
        self.total_emitted_source_tokens = 0
        self.tie_break_counter = 0

    def deficit_numerator(self, cluster_id: int) -> int:
        return (
            self.weight_units[cluster_id] * self.total_emitted_source_tokens
            - self.emitted_source_tokens[cluster_id] * self.weight_total
        )

    def choose(self, available_clusters: Iterable[int]) -> int | None:
        available = sorted(set(available_clusters))
        if not available:
            return None
        unknown = set(available) - set(self.weight_units)
        if unknown:
            raise ValueError(f"scheduler received unaccepted clusters {sorted(unknown)}")
        highest = max(self.deficit_numerator(cluster) for cluster in available)
        tied = [cluster for cluster in available if self.deficit_numerator(cluster) == highest]
        if len(tied) == 1:
            return tied[0]
        return min(
            tied,
            key=lambda cluster: hashlib.sha256(
                f"{self.tie_break_seed}\0{self.tie_break_counter}\0{cluster}".encode("utf-8")
            ).digest(),
        )

    def emit(self, document: SourceDocument) -> None:
        if document.cluster_id not in self.weight_units:
            raise ValueError(f"cannot schedule excluded cluster {document.cluster_id}")
        tokens = document.source_token_count
        if tokens <= 0:
            raise ValueError("scheduled document must have source tokens")
        self.emitted_source_tokens[document.cluster_id] += tokens
        self.total_emitted_source_tokens += tokens
        # A scheduling decision is committed only when its whole document is
        # emitted.  Keeping ``choose`` pure makes repeated availability probes
        # deterministic and makes this counter safe to checkpoint.
        self.tie_break_counter += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "weight_units": {str(k): v for k, v in self.weight_units.items()},
            "emitted_source_tokens": {str(k): v for k, v in self.emitted_source_tokens.items()},
            "total_emitted_source_tokens": self.total_emitted_source_tokens,
            "tie_break_seed": self.tie_break_seed,
            "tie_break_counter": self.tie_break_counter,
        }

    @classmethod
    def from_state(cls, weights: Mapping[int | str, object], state: Mapping[str, object]) -> "TokenDeficitScheduler":
        instance = cls(weights, str(state["tie_break_seed"]))
        expected = {str(k): v for k, v in instance.weight_units.items()}
        if state.get("weight_units") != expected:
            raise ValueError("scheduler state weights do not match this run")
        raw = state.get("emitted_source_tokens")
        if not isinstance(raw, Mapping):
            raise ValueError("scheduler state has no emitted_source_tokens")
        instance.emitted_source_tokens = {cluster: int(raw[str(cluster)]) for cluster in instance.weight_units}
        instance.total_emitted_source_tokens = int(state["total_emitted_source_tokens"])
        instance.tie_break_counter = int(state["tie_break_counter"])
        if sum(instance.emitted_source_tokens.values()) != instance.total_emitted_source_tokens:
            raise ValueError("scheduler state total does not equal per-cluster counters")
        return instance


@dataclass(frozen=True)
class PreparedBlock:
    block_id: int
    split: str
    sequence_count: int
    token_count: int
    payload: bytes
    cumulative_source_tokens: int
    per_cluster_source_tokens: dict[int, int]
    first_source_id: str
    last_source_id: str
    schema_version: int = STREAM_CACHE_SCHEMA_VERSION


@dataclass(frozen=True)
class ShardMetadata:
    filename: str
    split: str
    byte_size: int
    token_count: int
    sequence_count: int
    checksum: str
    first_block_id: int
    last_block_id: int
    context_length: int
    int_type: str
    byte_order: str
    cumulative_cluster_source_tokens: dict[int, int]
    shard_cluster_source_tokens: dict[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.__dict__, "cumulative_cluster_source_tokens": {str(k): v for k, v in self.cumulative_cluster_source_tokens.items()},
            "shard_cluster_source_tokens": {str(k): v for k, v in self.shard_cluster_source_tokens.items()},
        }


class ImmutableShardWriter:
    """Write only temporary active files; finalized shards are never reopened."""

    def __init__(self, output_dir: Path, *, split: str, target_bytes: int, context_length: int, start_index: int | None = None) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.directory = output_dir / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split, self.target_bytes, self.context_length = split, target_bytes, context_length
        self.shards: list[ShardMetadata] = []
        if start_index is not None:
            self._index = start_index
        else:
            existing = [
                int(p.stem.split("-")[-1])
                for p in self.directory.glob(f"{split}-*.bin")
                if p.stem.split("-")[-1].isdigit()
            ]
            self._index = max(existing) + 1 if existing else 0
        self._handle = None
        self._blocks: list[PreparedBlock] = []
        self._cluster_counts: dict[int, int] = {}
        self._cumulative_counts: dict[int, int] = {}

    @property
    def active_path(self) -> Path:
        return self.directory / f".{self.split}-{self._index:06d}.bin.tmp"

    def write_block(self, block: PreparedBlock) -> list[ShardMetadata]:
        if block.split != self.split:
            raise ValueError("block split does not match shard writer")
        finalized: list[ShardMetadata] = []
        if self._blocks and self._byte_size() + len(block.payload) > self.target_bytes:
            finalized.append(self.finalize_active())
        if self._handle is None:
            if self.active_path.exists():
                raise RuntimeError(
                    f"mutable active shard already exists and is not recoverable: {self.active_path}"
                )
            self._handle = self.active_path.open("xb")
        written = self._handle.write(block.payload)
        if written != len(block.payload):
            raise OSError("short shard write")
        # Persist each complete block before the caller can checkpoint its progress.
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._blocks.append(block)
        for cluster, count in block.per_cluster_source_tokens.items():
            self._cluster_counts[cluster] = self._cluster_counts.get(cluster, 0) + count
            self._cumulative_counts[cluster] = self._cumulative_counts.get(cluster, 0) + count
        return finalized

    def _byte_size(self) -> int:
        return sum(len(block.payload) for block in self._blocks)

    def finalize_active(self) -> ShardMetadata:
        if self._handle is None or not self._blocks:
            raise RuntimeError("no active shard to finalize")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        final_name = f"{self.split}-{self._index:06d}.bin"
        final_path = self.directory / final_name
        os.replace(self.active_path, final_path)
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        metadata = ShardMetadata(
            filename=f"{self.split}/{final_name}", split=self.split, byte_size=final_path.stat().st_size,
            token_count=final_path.stat().st_size // 2,
            sequence_count=sum(block.sequence_count for block in self._blocks),
            checksum=sha256_file(final_path), first_block_id=self._blocks[0].block_id,
            last_block_id=self._blocks[-1].block_id, context_length=self.context_length,
            int_type=config.INT_TYPE, byte_order=config.BYTE_ORDER,
            cumulative_cluster_source_tokens=dict(self._cumulative_counts),
            shard_cluster_source_tokens=dict(self._cluster_counts),
        )
        self.shards.append(metadata)
        self._index += 1
        self._handle = None
        self._blocks = []
        self._cluster_counts = {}
        return metadata

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def finish(self) -> list[ShardMetadata]:
        return [self.finalize_active()] if self._blocks else []
