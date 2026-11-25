"""Trainer engine adapter for the Kefir MoE model."""

from __future__ import annotations

from dataclasses import asdict
import math
import torch
from pathlib import Path
from typing import Mapping

from kefir.trainer.config import MOE_RESUME_EXECUTION_FIELDS, TrainerConfig
from kefir.trainer.engine import TrainerEngine
from kefir.trainer.state import load_trainer_state_file, save_engine_checkpoint_state
from kefir.trainer.schedule import TokenLRScheduler

from .accounting import count_moe_parameters
from .model import MoESmallLLM
from .optimizer import build_moe_optimizer
from .step import moe_train_step


_FORK_SCHEDULE_FIELDS = frozenset({
    "learning_rate", "schedule", "warmup_tokens", "stable_tokens", "decay_tokens",
    "minimum_lr_ratio", "schedule_anchor_tokens", "cooldown_start_tokens",
    "settle_tokens", "settle_lr_ratio", "base_power",
})


class MoETrainerEngine(TrainerEngine):
    def __init__(
        self,
        model: MoESmallLLM,
        config: TrainerConfig,
        *,
        device: str | torch.device | None = None,
        optimizer_telemetry: bool = True,
    ) -> None:
        resolved = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        model.to(resolved)
        optimizer = build_moe_optimizer(model, config, telemetry=optimizer_telemetry)
        super().__init__(model, config, device=resolved, optimizer=optimizer)
        self.parameter_accounting = count_moe_parameters(
            model, num_experts=model.config.num_experts, top_k=model.config.top_k
        )

    def train_batch(self, batch: object):
        if self.distributed is not None:
            from .distributed import distributed_moe_train_step
            return distributed_moe_train_step(self, batch)
        return moe_train_step(self, batch)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        # A logical block is the optimizer batch. Its microbatch partition and
        # observation cadence may change across single-GPU providers; the
        # scientific recipe, model, optimizer moments and token clock may not.
        if self.raw_model.config.version == 3:
            saved = state.get("config")
            target = self.config.as_dict()
            scheduler = state.get("scheduler")
            if not isinstance(saved, Mapping) or not isinstance(scheduler, Mapping):
                raise ValueError("checkpoint has no trainer/scheduler configuration")
            if set(saved) != set(target):
                raise ValueError("checkpoint trainer configuration fields differ")
            TrainerConfig(**saved)
            if scheduler.get("config") != saved:
                raise ValueError("checkpoint scheduler and trainer configurations disagree")
            changed = {key for key in set(saved) | set(target) if saved.get(key) != target.get(key)}
            if changed - MOE_RESUME_EXECUTION_FIELDS:
                raise ValueError(f"checkpoint scientific configuration mismatch: {sorted(changed - MOE_RESUME_EXECUTION_FIELDS)}")
            state = {**state, "config": target, "scheduler": {**scheduler, "config": target}}
        super().load_state_dict(state)

    def load_checkpoint_state(self, path: Path | str) -> None:
        self.load_state_dict(load_trainer_state_file(path, map_location="cpu"))

    def save_checkpoint_state(self, path: Path | str) -> None:
        if self.distributed is not None and self.model is self.raw_model:
            save_engine_checkpoint_state(self, path)
        else:
            super().save_checkpoint_state(path)

    def load_fork_state_dict(self, state: Mapping[str, object]) -> None:
        """Keep a completed version-3 training state while changing only its LR curve."""

        if self.raw_model.config.version != 3:
            raise ValueError("schedule forks require the accepted version-3 MoE")
        saved = state.get("config")
        scheduler = state.get("scheduler")
        if not isinstance(saved, Mapping) or not isinstance(scheduler, Mapping):
            raise ValueError("fork checkpoint has no trainer/scheduler configuration")
        old = TrainerConfig(**saved)
        if dict(saved) != old.as_dict() or scheduler.get("config") != saved:
            raise ValueError("fork checkpoint trainer/scheduler configurations disagree")
        target = self.config.as_dict()
        changed = {
            name for name in asdict(old)
            if getattr(old, name) != getattr(self.config, name)
        }
        forbidden = changed - _FORK_SCHEDULE_FIELDS - MOE_RESUME_EXECUTION_FIELDS
        if forbidden:
            raise ValueError(f"fork changes non-schedule training fields: {sorted(forbidden)}")
        if not changed.intersection(_FORK_SCHEDULE_FIELDS):
            raise ValueError("schedule fork must change the learning-rate recipe")
        if state.get("model_config") != self.raw_model.config.as_dict():
            raise ValueError("fork model configuration differs from its parent")
        tokens = state.get("consumed_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("fork checkpoint has no completed token clock")
        if self.config.schedule == "wsqd" and self.config.schedule_anchor_tokens != tokens:
            raise ValueError("WSqD fork anchor must equal parent committed tokens")
        if scheduler.get("committed_tokens") != tokens:
            raise ValueError("fork scheduler and trainer token clocks disagree")
        cuda_states = state.get("cuda_rng_states", [])
        if (not isinstance(cuda_states, list) or len(cuda_states) > 2
                or any(not isinstance(item, torch.Tensor) for item in cuda_states)
                or any(not torch.equal(cuda_states[0], item) for item in cuda_states[1:])):
            raise ValueError("fork requires identical portable CUDA RNG states")

        witness = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=old.learning_rate)
        old_schedule = TokenLRScheduler(witness, old)
        old_schedule.load_state_dict(scheduler)
        new_lr = self.scheduler._lr_at(tokens)
        if not math.isclose(new_lr, old_schedule.last_lr, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError("fork learning rate must be continuous at the parent token clock")
        child_scheduler = {
            "version": TokenLRScheduler.VERSION,
            "config": target,
            "committed_tokens": tokens,
            "last_lr": new_lr,
        }
        super().load_state_dict({**state, "config": target, "scheduler": child_scheduler})


__all__ = ["MoETrainerEngine"]
