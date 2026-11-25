"""A small, provider-independent entry point to the existing MoE trainer."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal

from kefir.data.src.checkpoint_sequence import (
    CHECKPOINT_ID, INVALID_CHECKPOINT, checkpoint_step, complete_checkpoint,
)
from kefir.trainer.config import MOE_RESUME_EXECUTION_FIELDS, TrainerConfig
from kefir.trainer.shard_layout import manifest_identity
from kefir.trainer.identity import dataset_configuration_hash


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_SHA = re.compile(r"[0-9a-f]{40}")
_V2_TOKENIZER_SHA256 = "068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb"
_SCIENTIFIC_FIELDS = frozenset({
    "optimizer", "learning_rate", "weight_decay", "muon_momentum", "muon_lr_multiplier",
    "muon_update_rms", "muon_weight_decay", "max_grad_norm", "schedule", "warmup_tokens",
    "stable_tokens", "decay_tokens", "minimum_lr_ratio", "schedule_anchor_tokens",
    "cooldown_start_tokens", "settle_tokens", "settle_lr_ratio", "base_power", "seed",
})


def _read_json(path: str | Path, cls: type[Any]) -> Any:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    known = {field.name for field in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"{source} has unknown fields: {sorted(unknown)}")
    return cls(**value)


@dataclass(frozen=True, slots=True)
class RunSpec:
    """The scientific identity and cumulative stopping point of a Kefir run."""

    run_id: str
    dataset_dir: str
    target_tokens: int
    max_steps: int
    training: dict[str, Any]
    data_manifest: str | None = None
    run_source_commit: str | None = None
    parent_checkpoint: str | None = None
    tokenizer_sha256: str = _V2_TOKENIZER_SHA256
    global_batch_size: int = 64
    model_size: Literal["accepted"] = "accepted"
    extend_budget: bool = False

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id contains invalid characters")
        if not self.dataset_dir:
            raise ValueError("dataset_dir is required")
        for name in ("target_tokens", "max_steps", "global_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.extend_budget) is not bool:
            raise ValueError("extend_budget must be a boolean")
        if self.model_size != "accepted":
            raise ValueError("Kefir v1 uses the accepted MoE configuration")
        if _SHA256.fullmatch(self.tokenizer_sha256) is None:
            raise ValueError("tokenizer_sha256 must be a SHA-256 digest")
        if self.tokenizer_sha256 != _V2_TOKENIZER_SHA256:
            raise ValueError("Kefir v1 requires the pinned SuperBPE v2 tokenizer")
        if self.run_source_commit is not None and _SOURCE_SHA.fullmatch(self.run_source_commit) is None:
            raise ValueError("run_source_commit must be a full lowercase Git SHA")
        if self.parent_checkpoint is not None and not self.parent_checkpoint:
            raise ValueError("parent_checkpoint must name a checkpoint directory")
        if self.parent_checkpoint is not None and not Path(self.parent_checkpoint).is_absolute():
            raise ValueError("parent_checkpoint must be an absolute path")
        if not isinstance(self.training, dict):
            raise ValueError("training must be a JSON object")
        unknown = set(self.training) - _SCIENTIFIC_FIELDS
        if unknown:
            raise ValueError(f"training has unknown or execution-only fields: {sorted(unknown)}")
        if not {"optimizer", "learning_rate", "schedule"} <= self.training.keys():
            raise ValueError("training must explicitly set optimizer, learning_rate and schedule")
        TrainerConfig(**self.training)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunSpec":
        return _read_json(path, cls)


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Hardware, segment length, and optional transport for one invocation."""

    run_root: str
    microbatch_size: int
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    device: str = "cuda"
    source_commit: str | None = None
    compile_mode: Literal["off", "blocks"] = "off"
    optimizer_telemetry: bool = True
    checkpoint_fsync: bool = True
    validation_microbatch_size: int = 1
    validation_blocks: int = 0
    checkpoint_every_steps: int = 1000
    keep_last_checkpoints: int = 0
    evaluation_every_steps: int = 1000
    max_wall_seconds: float = 0.0
    thematic_panels: str = ""
    thematic_microbatch_size: int = 32
    wandb_mode: Literal["disabled", "online", "offline"] = "disabled"
    wandb_project: str = "Kefir"
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: Literal["never", "allow", "must", "auto"] = "never"
    remote_checkpoint_bucket: str | None = None
    remote_drive_manifest: str | None = None
    remote_publish_every_steps: int = 0
    remote_rolling_latest_only: bool = False
    async_checkpoint_upload: bool = False
    remote_token_env: str = "HF_TOKEN"
    dataset_shard_bucket: str | None = None
    dataset_shard_run_id: str | None = None
    dataset_shard_token_env: str = "HF_TOKEN"
    dataset_shard_prefetch: int = 1
    dataset_shard_wait_timeout_seconds: float = 0.0
    local_incremental_frontier: bool = False
    legacy_checkpoint_sha256: str | None = None
    legacy_dataset_configuration_hash: str | None = None

    def __post_init__(self) -> None:
        for name, allowed in {
            "precision": {"fp32", "fp16", "bf16"},
            "compile_mode": {"off", "blocks"},
            "wandb_mode": {"disabled", "online", "offline"},
            "wandb_resume": {"never", "allow", "must", "auto"},
        }.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"unsupported {name}: {getattr(self, name)!r}")
        for name in ("optimizer_telemetry", "checkpoint_fsync", "local_incremental_frontier",
                     "remote_rolling_latest_only", "async_checkpoint_upload"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not self.run_root:
            raise ValueError("run_root is required")
        if self.source_commit is not None and _SOURCE_SHA.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a full lowercase Git SHA")
        for name in ("microbatch_size", "validation_microbatch_size",
                     "thematic_microbatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("validation_blocks", "checkpoint_every_steps", "keep_last_checkpoints",
                     "evaluation_every_steps",
                     "remote_publish_every_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds must be finite and non-negative")
        if self.remote_publish_every_steps and not (
            self.remote_checkpoint_bucket and self.remote_drive_manifest
        ):
            raise ValueError("remote publication requires a bucket and drive manifest")
        if (self.remote_rolling_latest_only or self.async_checkpoint_upload) and not self.remote_publish_every_steps:
            raise ValueError("remote checkpoint controls require remote publication")
        if self.wandb_resume != "never" and not self.wandb_run_id:
            raise ValueError("W&B resume requires wandb_run_id")
        if bool(self.dataset_shard_bucket) != bool(self.dataset_shard_run_id):
            raise ValueError("rolling dataset requires both bucket and run ID")
        if self.local_incremental_frontier and self.dataset_shard_bucket:
            raise ValueError("local incremental frontier cannot use a remote shard bucket")
        legacy = (self.legacy_checkpoint_sha256, self.legacy_dataset_configuration_hash)
        if any(value is not None for value in legacy) and not all(
            isinstance(value, str) and _SHA256.fullmatch(value) for value in legacy
        ):
            raise ValueError("legacy resume requires checkpoint and dataset configuration SHA-256")
        if not self.dataset_shard_token_env:
            raise ValueError("dataset_shard_token_env is required")
        if isinstance(self.dataset_shard_prefetch, bool) or self.dataset_shard_prefetch < 1:
            raise ValueError("dataset_shard_prefetch must be positive")
        if (not math.isfinite(self.dataset_shard_wait_timeout_seconds)
                or self.dataset_shard_wait_timeout_seconds < 0):
            raise ValueError("dataset_shard_wait_timeout_seconds must be finite and non-negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "ExecutionSpec":
        return _read_json(path, cls)


@dataclass(frozen=True, slots=True)
class TrainResult:
    status: Literal["complete", "incomplete"]
    run_id: str
    step: int
    consumed_tokens: int
    checkpoint: Path


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _scientific_run(run: RunSpec, execution: ExecutionSpec) -> dict[str, object]:
    config = TrainerConfig(**{
        **run.training, "precision": execution.precision,
        "microbatch_size": execution.microbatch_size,
        "checkpoint_every_steps": execution.checkpoint_every_steps,
        "evaluation_every_steps": execution.evaluation_every_steps,
    })
    return {
        "version": 1, "run_id": run.run_id, "target_tokens": run.target_tokens,
        "max_steps": run.max_steps, "run_source_commit": run.run_source_commit,
        "tokenizer_sha256": run.tokenizer_sha256,
        "global_batch_size": run.global_batch_size, "model_size": run.model_size,
        "dataset_configuration_hash": dataset_configuration_hash(run.dataset_dir, run.data_manifest),
        "trainer": {key: value for key, value in config.as_dict().items()
                    if key not in MOE_RESUME_EXECUTION_FIELDS},
    }


def _verify_scientific_contract(saved: dict, expected: dict, checkpoint: Path,
                                *, extend_budget: bool = False) -> None:
    if saved == expected:
        return
    if not isinstance(expected, dict):
        raise ValueError("checkpoint scientific identity contract differs from the requested run")
    budget_keys = {"target_tokens", "max_steps"}
    same_recipe = ({key: value for key, value in saved.items() if key not in budget_keys}
                   == {key: value for key, value in expected.items() if key not in budget_keys})
    if (extend_budget and same_recipe
            and all(expected[key] > saved[key] for key in budget_keys)):
        complete = complete_checkpoint(checkpoint)
        if (complete["step"] >= saved["max_steps"]
                and complete["consumed_tokens"] >= saved["target_tokens"]):
            return
    raise ValueError("checkpoint scientific identity contract differs from the requested run")


def _verify_scientific_run(run: RunSpec, execution: ExecutionSpec, checkpoint: Path) -> None:
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    expected = _scientific_run(run, execution)
    saved = metadata.get("scientific_run")
    if saved is not None:
        _verify_scientific_contract(saved, expected, checkpoint, extend_budget=run.extend_budget)
        return
    if execution.legacy_checkpoint_sha256 is None:
        raise ValueError("checkpoint has no scientific run contract; provide verified legacy identities")
    with (checkpoint / "trainer_state.pkl").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (digest != execution.legacy_checkpoint_sha256
            or expected["dataset_configuration_hash"] != execution.legacy_dataset_configuration_hash):
        raise ValueError("legacy checkpoint or dataset configuration identity differs")


def _arguments(run: RunSpec, execution: ExecutionSpec, *, steps: int,
               resume: str | None, checkpoint_dir: Path, experiment_dir: Path) -> list[str]:
    scientific = TrainerConfig(**run.training)
    TrainerConfig(**{
        **run.training,
        "microbatch_size": execution.microbatch_size,
        "precision": execution.precision,
        "checkpoint_every_steps": execution.checkpoint_every_steps,
        "evaluation_every_steps": execution.evaluation_every_steps,
    })
    argv = [
        "--dataset-dir", run.dataset_dir,
        "--checkpoint-dir", str(checkpoint_dir),
        "--experiment-dir", str(experiment_dir),
        "--model-size", run.model_size,
        "--steps", str(steps),
        "--target-tokens", str(run.target_tokens),
        "--sequences-per-block", str(run.global_batch_size),
        "--device", execution.device,
        "--precision", execution.precision,
        "--compile", execution.compile_mode,
        "--microbatch-size", str(execution.microbatch_size),
        "--validation-microbatch-size", str(execution.validation_microbatch_size),
        "--validation-blocks", str(execution.validation_blocks),
        "--checkpoint-every-steps", str(execution.checkpoint_every_steps),
        "--keep-last-checkpoints", str(execution.keep_last_checkpoints),
        "--evaluation-every-steps", str(execution.evaluation_every_steps),
        "--max-wall-seconds", str(execution.max_wall_seconds),
        "--wandb-mode", execution.wandb_mode,
        "--wandb-project", execution.wandb_project,
        "--scientific-run-contract", json.dumps(_scientific_run(run, execution), sort_keys=True),
    ]
    for name in sorted(_SCIENTIFIC_FIELDS):
        value = getattr(scientific, name)
        argv.extend((_flag(name), str(value)))
    if run.extend_budget:
        argv.append("--extend-budget")
    if not execution.checkpoint_fsync:
        argv.append("--no-checkpoint-fsync")
    if not execution.optimizer_telemetry:
        argv.append("--no-optimizer-telemetry")
    if run.data_manifest:
        argv.extend(("--dataset-manifest", run.data_manifest))
    if execution.local_incremental_frontier:
        if not run.data_manifest:
            raise ValueError("local incremental frontier requires RunSpec.data_manifest")
        argv.append("--local-incremental-frontier")
    if execution.dataset_shard_bucket:
        if not run.data_manifest:
            raise ValueError("rolling dataset requires RunSpec.data_manifest")
        for name in ("dataset_shard_bucket", "dataset_shard_run_id", "dataset_shard_token_env",
                     "dataset_shard_prefetch", "dataset_shard_wait_timeout_seconds"):
            argv.extend((_flag(name), str(getattr(execution, name))))
    if execution.source_commit:
        argv.extend(("--source-commit", execution.source_commit))
    if execution.legacy_checkpoint_sha256 is not None:
        argv.extend(("--legacy-checkpoint-sha256", execution.legacy_checkpoint_sha256,
                     "--legacy-dataset-configuration-hash",
                     execution.legacy_dataset_configuration_hash))
    if run.run_source_commit:
        if execution.source_commit is None:
            raise ValueError("run_source_commit requires an executor source_commit")
        argv.extend(("--run-source-commit", run.run_source_commit))
    if resume:
        argv.extend(("--resume", resume))
    elif run.parent_checkpoint:
        argv.extend(("--fork-checkpoint", run.parent_checkpoint))
    if execution.thematic_panels:
        argv.extend(("--thematic-panels", execution.thematic_panels,
                     "--thematic-microbatch", str(execution.thematic_microbatch_size)))
    for name in ("wandb_entity", "wandb_run_id"):
        value = getattr(execution, name)
        if value:
            argv.extend((_flag(name), value))
    if execution.wandb_resume != "never":
        argv.extend(("--wandb-resume", execution.wandb_resume))
    if execution.remote_publish_every_steps:
        argv.extend(("--remote-checkpoint-bucket", str(execution.remote_checkpoint_bucket)))
        argv.extend(("--remote-drive-manifest", str(execution.remote_drive_manifest)))
        argv.extend(("--remote-publish-every-steps", str(execution.remote_publish_every_steps)))
        argv.extend(("--remote-token-env", execution.remote_token_env))
        if execution.remote_rolling_latest_only:
            argv.append("--remote-rolling-latest-only")
        if execution.async_checkpoint_upload:
            argv.append("--async-checkpoint-upload")
    return argv


def _latest_checkpoint(checkpoint_dir: Path) -> dict[str, object] | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(
        (path for path in checkpoint_dir.iterdir() if CHECKPOINT_ID.fullmatch(path.name)),
        key=lambda path: checkpoint_step(path) or -1,
        reverse=True,
    )
    return complete_checkpoint(candidates[0]) if candidates else None


def _validate_dataset_tokenizer(run: RunSpec, checkpoint: Path | None = None) -> None:
    manifest_path = Path(run.data_manifest) if run.data_manifest else Path(run.dataset_dir) / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = manifest.get("tokenizer") if isinstance(manifest, dict) else None
    if isinstance(tokenizer, dict) and tokenizer.get("tokenizer_sha256") == run.tokenizer_sha256:
        return
    checkpoint = checkpoint or (Path(run.parent_checkpoint) if run.parent_checkpoint else None)
    if tokenizer is None and checkpoint is not None and isinstance(manifest, dict):
        metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
        consumer = metadata.get("pipeline_state", {}).get("consumer", {})
        if (manifest.get("incremental_frontier") is not None
                and consumer.get("manifest_identity") == manifest_identity(
                    manifest, int(manifest["context_length"]), int(manifest["sequences_per_block"])
                )):
            return
    raise ValueError("dataset manifest tokenizer differs from the pinned Kefir v1 tokenizer")


def _validate_completed_resume(run: RunSpec, execution: ExecutionSpec, checkpoint: Path,
                               checkpoint_dir: Path, experiment_dir: Path) -> None:
    from kefir.model.__main__ import parse_args
    from kefir.model.setup import _model_config_from_args
    from kefir.trainer.config import MOE_RESUME_EXECUTION_FIELDS
    from kefir.trainer.identity import checkpoint_identity, saved_checkpoint_identity
    from kefir.trainer.shards import SchemaV2ShardReader
    from kefir.trainer.state import load_trainer_state_file

    args = parse_args(_arguments(run, execution, steps=1, resume=checkpoint.name,
                                checkpoint_dir=checkpoint_dir, experiment_dir=experiment_dir))
    model_config = _model_config_from_args(args)
    target = TrainerConfig(**{
        **run.training, "precision": execution.precision,
        "microbatch_size": execution.microbatch_size,
        "checkpoint_every_steps": execution.checkpoint_every_steps,
        "evaluation_every_steps": execution.evaluation_every_steps,
    })
    state = load_trainer_state_file(checkpoint / "trainer_state.pkl", map_location="cpu")
    saved = state.get("config")
    if state.get("version") != 1 or not isinstance(saved, dict) or set(saved) != set(target.as_dict()):
        raise ValueError("checkpoint trainer configuration fields differ")
    saved_config = TrainerConfig(**saved)
    changed = {key for key, value in target.as_dict().items() if saved[key] != value}
    if changed - MOE_RESUME_EXECUTION_FIELDS:
        raise ValueError(f"checkpoint scientific configuration mismatch: {sorted(changed - MOE_RESUME_EXECUTION_FIELDS)}")
    if state.get("model_config") != model_config.as_dict():
        raise ValueError("checkpoint model configuration mismatch")
    scheduler = state.get("scheduler", {})
    if (not isinstance(scheduler, dict) or scheduler.get("config") != saved
            or scheduler.get("committed_tokens") != state.get("consumed_tokens")):
        raise ValueError("checkpoint scheduler and trainer state disagree")
    from kefir.trainer.schedule import TokenLRScheduler
    import torch

    probe = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=target.learning_rate)
    TokenLRScheduler(probe, saved_config).load_state_dict(scheduler)
    current = checkpoint_identity(
        run.dataset_dir, model_config=model_config, trainer_config=saved_config,
        manifest_path=run.data_manifest, context_length=model_config.max_seq_len,
        sequences_per_block=run.global_batch_size,
    )
    if current != saved_checkpoint_identity(checkpoint):
        raise ValueError("completed checkpoint dataset identity differs from the current corpus")
    reader = SchemaV2ShardReader(
        run.dataset_dir, manifest_path=run.data_manifest,
        semantic_vocab_size=model_config.semantic_vocab_size,
        sequences_per_block=run.global_batch_size,
        context_length=model_config.max_seq_len if run.data_manifest else None,
    )
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    consumer = metadata["pipeline_state"].get("consumer", {})
    if consumer.get("kind") == "immutable_schema_v2":
        for key, value in reader.state_dict().items():
            if key != "last_consumed_block_id" and consumer.get(key) != value:
                raise ValueError(f"completed checkpoint data consumer mismatch: {key}")


def train(run: RunSpec, execution: ExecutionSpec, *, resume: str | None = None) -> TrainResult:
    """Run one bounded segment; call in every rank when launched by torchrun."""

    from kefir.model.__main__ import main as train_moe

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in {1, 2}:
        raise ValueError("Kefir v1 supports one or two ranks")
    if world == 2 and run.global_batch_size != 64:
        raise ValueError("two-rank Kefir v1 requires a 64-sequence logical block")
    if run.global_batch_size % (world * execution.microbatch_size):
        raise ValueError("microbatch_size must divide the per-rank logical block")
    checkpoint_dir = Path(execution.run_root) / run.run_id / "checkpoints"
    experiment_dir = Path(execution.run_root) / run.run_id / "artifacts"
    if resume is not None and resume != "latest" and CHECKPOINT_ID.fullmatch(resume) is None:
        raise ValueError("resume must be latest or a checkpoint ID")
    latest = _latest_checkpoint(checkpoint_dir) if resume in {None, "latest"} else None
    if resume is None and checkpoint_dir.is_dir() and any(
        CHECKPOINT_ID.fullmatch(path.name) for path in checkpoint_dir.iterdir()
    ):
        raise ValueError("run already has checkpoints; pass resume='latest' or a checkpoint ID")
    if resume == "latest":
        if latest is None:
            raise ValueError("no complete checkpoint to resume")
        selected = latest
    elif resume is not None:
        selected = complete_checkpoint(checkpoint_dir / resume)
    else:
        selected = None
    parent = None
    if selected is None and run.parent_checkpoint:
        parent = complete_checkpoint(Path(run.parent_checkpoint))
    origin = selected if selected is not None else parent
    _validate_dataset_tokenizer(run, Path(origin["path"]) if origin is not None else None)
    if selected is not None:
        _verify_scientific_run(run, execution, Path(selected["path"]))
    step = 0 if origin is None else int(origin["step"])
    tokens = 0 if origin is None else int(origin["consumed_tokens"])
    if parent is not None and (step >= run.max_steps or tokens >= run.target_tokens):
        raise ValueError("fork target must exceed parent step and token count")
    if step >= run.max_steps and tokens < run.target_tokens:
        raise ValueError("max_steps reached before target_tokens")
    if tokens >= run.target_tokens:
        if selected is None:
            raise RuntimeError("target reached without a checkpoint")
        _validate_completed_resume(run, execution, Path(selected["path"]),
                                   checkpoint_dir, experiment_dir)
        return TrainResult("complete", run.run_id, step, tokens, Path(selected["path"]))
    argv = _arguments(run, execution, steps=run.max_steps - step,
                      resume=None if selected is None else str(selected["checkpoint_id"]),
                      checkpoint_dir=checkpoint_dir, experiment_dir=experiment_dir)
    result = train_moe(argv)
    if result != 0:
        raise RuntimeError(f"trainer exited with status {result}")
    latest = _latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise RuntimeError("trainer exited without a complete checkpoint")
    tokens = int(latest["consumed_tokens"])
    return TrainResult("complete" if tokens >= run.target_tokens else "incomplete",
                       run.run_id, int(latest["step"]), tokens, Path(latest["path"]))
