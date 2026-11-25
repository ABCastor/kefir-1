"""Build the dedicated MoE model while reusing the validated data/checkpoint stack."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping

import torch
import torch.distributed as dist

from kefir.data.src.checkpoint_sequence import complete_checkpoint
from kefir.data.src.joint_checkpoint import CheckpointCoordinator
from kefir.trainer.shard_cache import rolling_cache
from kefir.trainer.config import MOE_RESUME_EXECUTION_FIELDS, TrainerConfig
from kefir.trainer.distributed import FollowerCache, initialize_distributed
from kefir.trainer.engine import TrainingSession, seed_everything
from kefir.trainer.identity import checkpoint_identity, dataset_configuration_hash, saved_checkpoint_identity
from kefir.trainer.shards import SchemaV2ShardReader
from kefir.trainer.state import load_trainer_state_file

from .compile_lane import DEFAULT_COMPILE_MODE, apply_compile_lane
from .config import MoEModelConfig
from .engine import MoETrainerEngine
from .initialization import initialize_moe_model
from .model import MoESmallLLM


def _model_config_from_args(args: object) -> MoEModelConfig:
    """Resolve an explicit MoE identity without silently changing accepted production settings."""

    if args.model_size != "accepted":
        raise ValueError("Kefir v1 supports only the accepted model")
    if getattr(args, "load_balancing", None) not in {None, "quantile"}:
        raise ValueError("accepted model requires Quantile Balancing")
    if getattr(args, "balancing_step_size", 0.0) != 0.0:
        raise ValueError("accepted model does not use balancing_step_size")
    return MoEModelConfig.accepted(
        architecture=args.architecture, gdn_chunk_size=args.gdn_chunk_size or 32,
    )


def _load_fork_checkpoint(
    args: object, session: TrainingSession,
    current_identity: tuple[str, str, str], distributed: object | None,
) -> dict[str, object]:
    parent = Path(args.fork_checkpoint)
    error: Exception | None = None
    try:
        if not parent.is_absolute() or parent.parent.resolve() == args.checkpoint_dir.resolve():
            raise ValueError("fork requires a checkpoint in a different run")
        if args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()):
            raise ValueError("fork destination already contains checkpoints")
        verified = complete_checkpoint(parent, verify_state=False)
        parent_identity = saved_checkpoint_identity(parent)
        if parent_identity[1:] != current_identity[1:]:
            raise ValueError("fork dataset source/schema identity differs from parent")
        metadata = json.loads((parent / "checkpoint.json").read_text(encoding="utf-8"))
        manifest = json.loads((parent / "local_manifest.json").read_text(encoding="utf-8"))
        state_digest = next(
            item["sha256"] for item in manifest["files"]
            if item["name"] == "trainer_state.pkl"
        )
        state = load_trainer_state_file(parent / "trainer_state.pkl", map_location="cpu")
        try:
            if state.get("global_step") != verified["step"]:
                raise ValueError("fork trainer step differs from checkpoint identity")
            if (args.target_tokens is not None
                    and args.target_tokens <= state.get("consumed_tokens", -1)):
                raise ValueError("fork target_tokens must exceed parent committed tokens")
            session.engine.load_fork_state_dict(state)
        finally:
            del state
        pipeline = metadata.get("pipeline_state")
        if not isinstance(pipeline, Mapping):
            raise ValueError("fork checkpoint has no pipeline cursor")
        session.source.load_pipeline_state(pipeline)
        if session.source.last_acknowledged_block_id != session.engine.global_step - 1:
            raise ValueError("fork reader cursor and trainer step disagree")
        lineage = {
            "version": 1,
            "parent_checkpoint_id": verified["checkpoint_id"],
            "parent_trainer_state_sha256": state_digest,
            "parent_configuration_hash": parent_identity[0],
        }
    except Exception as caught:
        error = caught
    if distributed is not None and distributed.vote(error is not None, session.engine.device):
        raise RuntimeError("fork validation failed on a training rank") from error
    if error is not None:
        raise error
    return lineage


def _save_fork_checkpoint(
    args: object, session: TrainingSession, coordinator: CheckpointCoordinator,
    distributed: object | None,
) -> None:

    checkpoint_id = f"step-{session.engine.global_step:08d}"
    error = None
    if distributed is None or distributed.primary:
        try:
            session.save_checkpoint(coordinator, checkpoint_id)
        except Exception as caught:
            error = caught
    if distributed is not None:
        if distributed.vote(error is not None, session.engine.device):
            raise RuntimeError("fork bootstrap checkpoint failed on rank zero") from error
        dist.barrier()
    elif error is not None:
        raise error
    args.fork_bootstrap_checkpoint = checkpoint_id


def _saved_fork_lineage(checkpoint: Path) -> dict[str, object] | None:
    payload = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    lineage = payload.get("fork")
    if lineage is None:
        return None
    if (not isinstance(lineage, dict)
            or set(lineage) != {
                "version", "parent_checkpoint_id", "parent_trainer_state_sha256",
                "parent_configuration_hash",
            }
            or lineage.get("version") != 1
            or not isinstance(lineage.get("parent_checkpoint_id"), str)
            or re.fullmatch(r"step-\d{1,12}", lineage["parent_checkpoint_id"]) is None
            or any(
                not isinstance(lineage.get(name), str)
                or re.fullmatch(r"[0-9a-f]{64}", lineage[name]) is None
                for name in ("parent_trainer_state_sha256", "parent_configuration_hash")
            )):
        raise ValueError("resumed checkpoint has invalid fork provenance")
    return dict(lineage)


def setup(args: object):
    distributed = initialize_distributed(args.device)
    if distributed is not None:
        if distributed.world_size != 2:
            raise RuntimeError("Kefir v1 currently supports one or two training ranks")
        args.device = (
            f"cuda:{distributed.local_rank}" if torch.device(args.device).type == "cuda" else "cpu"
        )
        if args.sequences_per_block != 64:
            raise RuntimeError("distributed Kefir v1 requires a 64-sequence logical block")
    seed_everything(args.seed)
    model_config = _model_config_from_args(args)

    model = MoESmallLLM(model_config)
    initialize_moe_model(model, args.initialization)
    # Execution lane, applied before the engine builds the optimizer. Compiling a
    # block in place never rebinds its parameters, so the optimizer below captures
    # the same tensors and the checkpoint contract is untouched.
    apply_compile_lane(model, getattr(args, "compile_mode", DEFAULT_COMPILE_MODE))
    trainer_config = TrainerConfig(
        optimizer=args.optimizer,
        microbatch_size=args.microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_momentum=args.muon_momentum,
        muon_lr_multiplier=args.muon_lr_multiplier,
        muon_update_rms=args.muon_update_rms,
        muon_weight_decay=args.muon_weight_decay,
        max_grad_norm=args.max_grad_norm,
        precision=args.precision,
        schedule=args.schedule,
        warmup_tokens=args.warmup_tokens,
        stable_tokens=args.stable_tokens,
        decay_tokens=args.decay_tokens,
        minimum_lr_ratio=args.minimum_lr_ratio,
        schedule_anchor_tokens=args.schedule_anchor_tokens,
        cooldown_start_tokens=args.cooldown_start_tokens,
        settle_tokens=args.settle_tokens,
        settle_lr_ratio=args.settle_lr_ratio,
        base_power=args.base_power,
        checkpoint_every_steps=args.checkpoint_every_steps,
        evaluation_every_steps=args.evaluation_every_steps,
        seed=args.seed,
    )

    if getattr(args, "local_incremental_frontier", False):
        cache_manager = FollowerCache(args.dataset_dir, timeout=10)
    elif distributed is not None and not distributed.primary and (
        getattr(args, "dataset_shard_bucket", None)
    ):
        cache_manager = FollowerCache(args.dataset_dir)
    else:
        cache_manager = rolling_cache(args)
    source = SchemaV2ShardReader(
        args.dataset_dir,
        split="train",
        sequences_per_block=args.sequences_per_block,
        semantic_vocab_size=model_config.semantic_vocab_size,
        manifest_path=args.dataset_manifest,
        context_length=(
            model_config.max_seq_len if args.dataset_manifest is not None else None
        ),
        cache_manager=cache_manager,
    )
    engine = MoETrainerEngine(model, trainer_config, device=args.device,
                              optimizer_telemetry=not getattr(args, "no_optimizer_telemetry", False))
    engine.checkpoint_fsync = not getattr(args, "no_checkpoint_fsync", False)
    engine.distributed = distributed
    session = TrainingSession(engine, source)

    fork_parent = getattr(args, "fork_checkpoint", None)
    if fork_parent is not None and args.resume:
        raise ValueError("fork checkpoint and resume are mutually exclusive")
    checkpoint_root = args.checkpoint_dir / args.resume if args.resume else None
    if checkpoint_root is not None and (checkpoint_root / "checkpoint.json").is_file():
        identities = saved_checkpoint_identity(checkpoint_root)
    else:
        identities = checkpoint_identity(
            args.dataset_dir,
            model_config=model_config,
            trainer_config=trainer_config,
            manifest_path=args.dataset_manifest,
            context_length=model_config.max_seq_len,
            sequences_per_block=args.sequences_per_block,
        )
    lineage = (
        _load_fork_checkpoint(args, session, identities, distributed)
        if fork_parent is not None else None
    )
    coordinator = CheckpointCoordinator(
        args.checkpoint_dir,
        configuration_hash=identities[0],
        source_hash=identities[1],
        schema_hash=identities[2],
        fsync=engine.checkpoint_fsync,
    )
    raw_contract = getattr(args, "scientific_run_contract", None)
    contract = json.loads(raw_contract) if raw_contract else None
    if contract is not None:
        trainer_science = {key: value for key, value in trainer_config.as_dict().items()
                           if key not in MOE_RESUME_EXECUTION_FIELDS}
        if (not isinstance(contract, dict) or contract.get("version") != 1
                or contract.get("target_tokens") != args.target_tokens
                or contract.get("global_batch_size") != args.sequences_per_block
                or contract.get("trainer") != trainer_science
                or contract.get("dataset_configuration_hash") != dataset_configuration_hash(
                    args.dataset_dir, args.dataset_manifest)):
            raise ValueError("scientific run contract differs from the trainer inputs")
        coordinator.scientific_run = contract
    if args.resume:
        metadata = json.loads((checkpoint_root / "checkpoint.json").read_text(encoding="utf-8"))
        saved_contract = metadata.get("scientific_run")
        if saved_contract is not None:
            from kefir.training import _verify_scientific_contract
            _verify_scientific_contract(saved_contract, contract, checkpoint_root,
                                        extend_budget=getattr(args, "extend_budget", False))
        if saved_contract is None and contract is not None:
            expected_sha = getattr(args, "legacy_checkpoint_sha256", None)
            expected_dataset = getattr(args, "legacy_dataset_configuration_hash", None)
            if expected_sha is None or expected_dataset is None:
                raise ValueError("legacy checkpoint requires verified checkpoint and dataset identities")
            with (checkpoint_root / "trainer_state.pkl").open("rb") as stream:
                actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if (actual_sha != expected_sha
                    or expected_dataset != dataset_configuration_hash(args.dataset_dir, args.dataset_manifest)):
                raise ValueError("legacy checkpoint or dataset configuration identity differs")
        session.load_checkpoint(coordinator, args.resume)
        coordinator.fork_metadata = _saved_fork_lineage(checkpoint_root)
        if model_config.version == 3:
            # New checkpoints describe the actual execution configuration after
            # migration; the loaded snapshot remains immutable.
            current = checkpoint_identity(
                args.dataset_dir, model_config=model_config, trainer_config=trainer_config,
                manifest_path=args.dataset_manifest, context_length=model_config.max_seq_len,
                sequences_per_block=args.sequences_per_block,
            )
            if current[1:] != identities[1:]:
                raise ValueError("resumed MoE dataset identity differs from the current corpus")
            coordinator.configuration_hash = current[0]
    elif fork_parent is not None:
        coordinator.fork_metadata = lineage
        if getattr(args, "source_commit", None):
            coordinator.executor_metadata = {
                "version": 1, "source_commit": args.source_commit,
                "run_source_commit": getattr(args, "run_source_commit", None) or args.source_commit,
            }
        _save_fork_checkpoint(args, session, coordinator, distributed)
    if distributed is not None:
        from .distributed import prepare_distributed_engine
        prepare_distributed_engine(engine, distributed)
        if not distributed.primary:
            args.experiment_dir = None
    return model_config, trainer_config, engine, session, coordinator


def validation_reader(args: object, model_config: MoEModelConfig) -> SchemaV2ShardReader:
    return SchemaV2ShardReader(
        args.dataset_dir,
        split="validation",
        sequences_per_block=args.sequences_per_block,
        semantic_vocab_size=model_config.semantic_vocab_size,
        manifest_path=args.dataset_manifest,
        context_length=(
            model_config.max_seq_len if args.dataset_manifest is not None else None
        ),
    )


__all__ = ["setup", "validation_reader", "_model_config_from_args"]
