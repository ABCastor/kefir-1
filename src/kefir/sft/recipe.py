"""The SFT recipe (scientific identity) and execution (hardware), parsed once at entry.

The recipe names its schedule in conversations ("chats"), the unit the Kefir 1
experiments were designed in; ``trainer_config`` converts it once into the token
spans of ``TrainerConfig`` on the committed Assistant-target clock, which is what
every checkpoint stores and every resume or fork compares.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
import re
from typing import Any

from kefir.trainer.config import TrainerConfig

from .data import Stream

# TrainerConfig fields a recipe may set in "optimizer". Schedule spans come only from
# "schedule"; microbatch size comes only from the execution file.
OPTIMIZER_FIELDS = frozenset({
    "optimizer", "learning_rate", "weight_decay", "beta1", "beta2", "adam_epsilon",
    "muon_momentum", "muon_lr_multiplier", "muon_update_rms", "muon_weight_decay",
    "max_grad_norm", "max_overflow_retries",
})
SCHEDULE_KINDS = ("constant", "wsd")


def _object(path: str | Path, cls: type) -> Any:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{source} has unknown fields: {sorted(unknown)}")
    return cls(**value), source.parent


def _pinned(value: Any, name: str) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or "sha256" not in value:
        raise ValueError(f"{name} must be {{'path': ..., 'sha256': <64 hex>}}")
    digest = value["sha256"]
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError(f"{name}.sha256 must be 64 lowercase hex characters, got {digest!r}")
    if digest == "0" * 64:
        raise ValueError(f"{name}.sha256 is a placeholder hash; pin the file's real SHA-256 (sha256sum)")
    return value


@dataclass(frozen=True)
class Recipe:
    """Everything that decides the trained weights."""
    run_id: str
    parent: dict
    segments: list
    validation: dict
    global_batch_size: int
    seed: int
    optimizer: dict
    schedule: dict
    precision: str = "fp16"
    initial_loss_scale: float = 256.0
    observe_at_chats: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id required")
        _pinned(self.parent, "parent")
        _pinned(self.validation, "validation")
        if not isinstance(self.segments, list) or not self.segments:
            raise ValueError("at least one training segment required")
        for index, segment in enumerate(self.segments):
            _pinned(segment, f"segments[{index}]")
            if set(segment) - {"path", "sha256", "order"}:
                raise ValueError(f"segments[{index}] has unknown fields")
            if segment.get("order") is not None:
                _pinned(segment["order"], f"segments[{index}].order")
        for name in ("global_batch_size", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "global_batch_size" else 0):
                raise ValueError(f"{name} must be a {'positive' if name == 'global_batch_size' else 'non-negative'} integer")
        if not isinstance(self.optimizer, dict) or set(self.optimizer) - OPTIMIZER_FIELDS:
            raise ValueError(f"optimizer may only set {sorted(OPTIMIZER_FIELDS)}")
        if not {"optimizer", "learning_rate"} <= self.optimizer.keys():
            raise ValueError("optimizer must set 'optimizer' and 'learning_rate'")
        kind = self.schedule.get("kind") if isinstance(self.schedule, dict) else None
        allowed = {"constant": {"kind"},
                   "wsd": {"kind", "warmup_chats", "decay_start_chats", "decay_end_chats",
                           "minimum_lr_ratio"}}
        if kind not in SCHEDULE_KINDS or set(self.schedule) - allowed[kind]:
            raise ValueError("schedule must be {'kind': 'constant'} or "
                             "{'kind': 'wsd', 'warmup_chats', 'decay_start_chats', "
                             "'decay_end_chats', 'minimum_lr_ratio'}")
        probe = {}
        if kind == "wsd":
            bounds = [self.schedule.get(name, 0 if name == "warmup_chats" else None)
                      for name in ("warmup_chats", "decay_start_chats", "decay_end_chats")]
            if not all(type(x) is int for x in bounds) or not 0 <= bounds[0] <= bounds[1] < bounds[2]:
                raise ValueError("wsd needs integer 0 <= warmup_chats <= decay_start_chats < decay_end_chats")
            probe = dict(schedule="wsd", decay_tokens=1,
                         minimum_lr_ratio=self.schedule.get("minimum_lr_ratio", 0.1))
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        scale = self.initial_loss_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("initial_loss_scale must be a positive finite number")
        # The core's own checks of optimizer values and the LR floor, before any data is read.
        TrainerConfig(**self.optimizer, precision=self.precision, seed=self.seed, **probe)
        if (not isinstance(self.observe_at_chats, list)
                or any(type(x) is not int or x <= 0 for x in self.observe_at_chats)
                or self.observe_at_chats != sorted(set(self.observe_at_chats))):
            raise ValueError("observe_at_chats must be increasing positive integers")

    @classmethod
    def from_json(cls, path: str | Path) -> tuple["Recipe", Path]:
        """Return the recipe and the directory its relative paths resolve against."""
        return _object(path, cls)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def trainer_config(self, stream: Stream, *, microbatch_size: int) -> TrainerConfig:
        """Normalize the chat-denominated schedule onto the committed-target clock."""
        common = dict(self.optimizer, precision=self.precision, seed=self.seed,
                      microbatch_size=microbatch_size)
        if self.schedule["kind"] == "constant":
            return TrainerConfig(**common, schedule="constant")
        spec = self.schedule
        tokens = [stream.targets_at_chats(x) for x in
                  (spec.get("warmup_chats", 0), spec["decay_start_chats"], spec["decay_end_chats"])]
        return TrainerConfig(**common, schedule="wsd", warmup_tokens=tokens[0],
                             stable_tokens=tokens[1] - tokens[0],
                             decay_tokens=tokens[2] - tokens[1],
                             minimum_lr_ratio=spec.get("minimum_lr_ratio", 0.1))


@dataclass(frozen=True)
class Execution:
    """Hardware choices that may change between segments of the same run."""
    output: str
    device: str = "cuda"
    microbatch_size: int = 2
    threads: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("output directory required")
        for name in ("microbatch_size", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_json(cls, path: str | Path) -> "Execution":
        return _object(path, cls)[0]


__all__ = ["Execution", "OPTIMIZER_FIELDS", "Recipe", "SCHEDULE_KINDS"]
