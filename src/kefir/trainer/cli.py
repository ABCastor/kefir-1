"""Bounded smoke-training CLI for immutable schema-v2 shards."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from collections.abc import Mapping
from pathlib import Path

import torch

from kefir.data.src.checkpoint_sequence import CHECKPOINT_ID, prune_checkpoints
from kefir.data.src.storage import read_json, write_json_atomic

from .cli_args import parse_args
from .best_model import (
    checkpoint_validation_loss,
    get_dedicated_best_metric,
    publish_dedicated_best_model,
)
from .remote_publication import cleanup_remote_publication, configure_remote_publication, promote_best_pointer
from .wandb_logging import configure_wandb

setup = validation_reader = None


def _tree_byte_size(value: object) -> int | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if path.is_symlink() or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _validation_metric(validation: Mapping[str, object] | None) -> float | None:
    """Return a higher-is-better remote metric from held-out loss."""

    if validation is None:
        return None
    loss = validation.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise RuntimeError("validation result has no numeric loss for best-checkpoint selection")
    loss_value = float(loss)
    if loss_value < 0 or not math.isfinite(loss_value):
        raise RuntimeError("validation loss for best-checkpoint selection is invalid")
    return -loss_value


def _existing_remote_best_metric(remote: object | None) -> float | None:
    """Read the persisted best metric so resumed runs cannot overwrite it blindly."""

    if remote is None or bool(getattr(remote, "rolling_latest_only", False)):
        return None
    publisher = getattr(remote, "publisher", None)
    store = getattr(publisher, "store", None)
    read_json = getattr(store, "read_json", None)
    drive_manifest = getattr(remote, "drive_manifest", None)
    if not callable(read_json) or not isinstance(drive_manifest, Mapping):
        return None
    run_id = drive_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("remote publication has no valid run_id")
    pointer = read_json(f"run/{run_id}/best.json")
    if pointer is None:
        return None
    if not isinstance(pointer, Mapping):
        raise RuntimeError("remote best.json is not a JSON object")
    metric = pointer.get("metric")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        raise RuntimeError("remote best.json has no numeric metric")
    value = float(metric)
    if not math.isfinite(value):
        raise RuntimeError("remote best.json metric is non-finite")
    return value


def _is_primary_process() -> bool:
    """Return whether this process owns external side effects."""

    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return True
    try:
        if not distributed.is_available() or not distributed.is_initialized():
            return True
        return int(distributed.get_rank()) == 0
    except RuntimeError:
        return True


def main(
    argv: list[str] | None = None, *, setup_fn=None, validation_reader_fn=None,
    parse_args_fn=None,
) -> int:
    started_wall = time.monotonic()
    args = (parse_args_fn or parse_args)(argv)
    build = setup_fn or setup
    if build is None:
        raise RuntimeError("call the Kefir or MoE entry point with an explicit model setup")
    model_config, trainer_config, engine, session, coordinator = build(args)
    if getattr(args, "source_commit", None):
        coordinator.executor_metadata = {
            "version": 1, "source_commit": args.source_commit,
            "run_source_commit": getattr(args, "run_source_commit", None) or args.source_commit,
        }
    observer = None
    if getattr(args, "experiment_dir", None) is not None:
        from .observation import RunObservation

        observer = RunObservation(
            args, engine, model_config, trainer_config,
            validation_reader_fn or validation_reader,
        )
    is_primary_process = _is_primary_process()
    distributed = getattr(engine, "distributed", None)

    def emit(*values, **kwargs) -> None:
        if is_primary_process:
            print(*values, **kwargs)

    remote = configure_remote_publication(args) if is_primary_process else None
    best_remote_metric = _existing_remote_best_metric(remote)
    best_model_repo = getattr(args, "best_model_repo", None) if is_primary_process else None
    best_model_metric: float | None = None
    best_model_token: str | None = None
    if best_model_repo:
        token_env = str(getattr(args, "remote_token_env", "HF_TOKEN"))
        best_model_token = os.environ.get(token_env)
        run_id = getattr(args, "wandb_run_id", None)
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("dedicated best-model publication requires a stable --wandb-run-id")
        persisted_best_loss = getattr(engine, "best_validation_loss", None)
        persisted_best_metric: float | None = None
        if persisted_best_loss is not None:
            if (
                isinstance(persisted_best_loss, bool)
                or not isinstance(persisted_best_loss, (int, float))
                or not math.isfinite(float(persisted_best_loss))
                or float(persisted_best_loss) < 0
            ):
                raise RuntimeError("trainer state has an invalid persisted best validation loss")
            persisted_best_metric = -float(persisted_best_loss)
        repository_best_metric = get_dedicated_best_metric(
            repo_id=str(best_model_repo),
            run_id=run_id,
            token=best_model_token,
        )
        if (
            repository_best_metric is None
            and persisted_best_metric is not None
            and getattr(args, "resume", None)
        ):
            resume_checkpoint_id = str(args.resume)
            resume_checkpoint = Path(args.checkpoint_dir) / resume_checkpoint_id
            resume_loss = checkpoint_validation_loss(resume_checkpoint)
            if -resume_loss == persisted_best_metric:
                repair = publish_dedicated_best_model(
                    repo_id=str(best_model_repo),
                    run_id=run_id,
                    checkpoint_dir=resume_checkpoint,
                    checkpoint_id=resume_checkpoint_id,
                    metric=persisted_best_metric,
                    validation_loss=resume_loss,
                    token=best_model_token,
                    recreate=bool(getattr(args, "best_model_recreate", False)),
                )
                repository_best_metric = persisted_best_metric
                emit(
                    json.dumps(
                        {"best_model_resume_repair": dict(repair)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        thresholds = [
            value
            for value in (persisted_best_metric, repository_best_metric)
            if value is not None
        ]
        best_model_metric = max(thresholds) if thresholds else None
    telemetry = configure_wandb(
        args,
        model_config=model_config,
        trainer_config=trainer_config,
        engine=engine,
    ) if is_primary_process else None
    uploader = None
    validation: dict[str, object] | None = None
    saved: set[str] = set()
    saved_paths: dict[str, Path] = {}
    saved_validation: dict[str, dict[str, object] | None] = {}
    remotely_published: set[str] = set()
    completed = 0
    last_validation_step = None
    keep_last_checkpoints = int(getattr(args, "keep_last_checkpoints", 0) or 0)
    milestone_every_steps = int(getattr(args, "milestone_every_steps", 0) or 0)
    max_wall_seconds = float(getattr(args, "max_wall_seconds", 0.0) or 0.0)
    best_local_checkpoint_id: str | None = None
    best_checkpoint_path = Path(args.checkpoint_dir) / "best_checkpoint.json"
    if keep_last_checkpoints > 0 and is_primary_process and best_checkpoint_path.exists():
        best_pointer = read_json(best_checkpoint_path)
        best_local_checkpoint_id = best_pointer["checkpoint_id"]
        if (not isinstance(best_local_checkpoint_id, str)
                or CHECKPOINT_ID.fullmatch(best_local_checkpoint_id) is None):
            raise ValueError("best_checkpoint.json has an invalid checkpoint ID")
        local_metric = best_pointer.get("metric")
        if (isinstance(local_metric, bool) or not isinstance(local_metric, (int, float))
                or not math.isfinite(local_metric) or local_metric > 0):
            raise ValueError("best_checkpoint.json has an invalid validation metric")
        best_model_metric = max(best_model_metric, local_metric) if best_model_metric is not None else local_metric

    def primary_validation() -> dict[str, object]:
        nonlocal last_validation_step
        if distributed is not None and not is_primary_process:
            last_validation_step = engine.global_step
            return {"loss": 0.0, "perplexity": 1.0, "target_tokens": 0, "blocks": 0}
        started = time.perf_counter()
        reader = (validation_reader_fn or validation_reader)(args, model_config)
        result = engine.evaluate(
            reader.iter_from_start(args.validation_blocks),
            maximum_batches=args.validation_blocks,
            **({"microbatch_size": args.validation_microbatch_size}
               if getattr(args, "validation_microbatch_size", 1) != 1 else {}),
        )
        result = dict(result)
        panels_directory = str(getattr(args, "thematic_panels", "") or "")
        if panels_directory:
            # Nested beside the scalar validation loss: `log_validation` flattens it
            # into `validation/thematic/<domain>/loss`, and best-checkpoint selection
            # keeps reading the top-level held-out `loss`.
            from .thematic_validation import evaluate_thematic, load_thematic_panels

            result["thematic"] = evaluate_thematic(
                getattr(engine, "raw_model", engine.model),
                device=engine.device,
                microbatch=int(getattr(args, "thematic_microbatch", 32)),
                panels=load_thematic_panels(panels_directory),
                precision=engine.config.precision,
            )
        elapsed = time.perf_counter() - started
        last_validation_step = engine.global_step
        emit(
            json.dumps(
                {"validation": result, "elapsed_seconds": elapsed},
                sort_keys=True,
            ),
            flush=True,
        )
        if telemetry is not None:
            telemetry.log_validation(
                step=engine.global_step,
                metrics=result,
                elapsed_seconds=elapsed,
            )
        return result

    def run_validation() -> dict[str, object]:
        error: BaseException | None = None
        result: dict[str, object] = {}
        try:
            result = primary_validation()
        except BaseException as caught:
            error = caught
        failed = (distributed.vote(error is not None, engine.device)
                  if distributed is not None else error is not None)
        if failed:
            if error is not None:
                raise error
            raise RuntimeError("distributed validation failed on another rank")
        return result

    def current_validation() -> dict[str, object] | None:
        return validation if last_validation_step == engine.global_step else None

    def ensure_local_checkpoint(checkpoint_id: str) -> Path:
        if distributed is not None and not is_primary_process:
            return Path(args.checkpoint_dir) / checkpoint_id
        # Finish the previous immutable snapshot before any save/prune can remove it.
        if uploader is not None:
            uploader.poll(wait=True)
        if checkpoint_id in saved:
            path = saved_paths.get(checkpoint_id)
            if path is not None:
                return path
            return Path(args.checkpoint_dir) / checkpoint_id
        if completed == 0 and checkpoint_id in {
            args.resume, getattr(args, "fork_bootstrap_checkpoint", None)
        }:
            path = Path(args.checkpoint_dir) / checkpoint_id
            saved.add(checkpoint_id)
            saved_paths[checkpoint_id] = path
            saved_validation[checkpoint_id] = read_json(path / "checkpoint.json").get("validation_metrics") or None
            if observer is not None:
                observer.checkpoint(checkpoint_id, elapsed_seconds=0., byte_size=None)
            return path
        started = time.perf_counter()
        checkpoint = session.save_checkpoint(
            coordinator,
            checkpoint_id,
            validation_metrics=current_validation(),
        )
        elapsed = time.perf_counter() - started
        if checkpoint is None:
            # Distributed execution shims may no-op checkpoint writes on
            # non-primary ranks while rank zero persists and publishes the
            # scientific checkpoint. Keep the generic trainer loop advancing
            # without inventing a local byte size for that non-written path.
            path = Path(args.checkpoint_dir) / checkpoint_id
            byte_size = None
        else:
            path = Path(checkpoint)
            byte_size = _tree_byte_size(checkpoint)
        saved.add(checkpoint_id)
        saved_paths[checkpoint_id] = path
        saved_validation[checkpoint_id] = current_validation()
        event = {
            "checkpoint_id": checkpoint_id,
            "elapsed_seconds": elapsed,
            "byte_size": byte_size,
        }
        emit(json.dumps({"local_checkpoint": event}, sort_keys=True), flush=True)
        if telemetry is not None:
            telemetry.log_checkpoint(
                step=engine.global_step,
                checkpoint_id=checkpoint_id,
                elapsed_seconds=elapsed,
                byte_size=event["byte_size"],
            )
        if observer is not None:
            observer.checkpoint(checkpoint_id, elapsed_seconds=elapsed, byte_size=byte_size)
        prune_local_checkpoints()
        return path

    def prune_local_checkpoints() -> None:
        if keep_last_checkpoints <= 0 or not is_primary_process:
            return
        protected = {
            value for value in (best_local_checkpoint_id, getattr(args, "resume", None))
            if isinstance(value, str) and value
        }
        removed = prune_checkpoints(
            Path(args.checkpoint_dir),
            keep_last=keep_last_checkpoints,
            protected=protected,
            milestone_every_steps=milestone_every_steps,
        )
        if not removed:
            return
        event = {
            "removed": sorted(removed),
            "keep_last": keep_last_checkpoints,
            "milestone_every_steps": milestone_every_steps,
            "protected": sorted(protected),
        }
        emit(json.dumps({"checkpoint_retention": event}, sort_keys=True), flush=True)

    def publish_best_model_if_improved(checkpoint_id: str) -> None:
        nonlocal best_model_metric, best_local_checkpoint_id
        if not is_primary_process:
            return
        metric = _validation_metric(saved_validation.get(checkpoint_id, current_validation()))
        if metric is None or (best_model_metric is not None and metric <= best_model_metric):
            return
        if keep_last_checkpoints > 0:
            ensure_local_checkpoint(checkpoint_id)
        if best_model_repo:
            run_id = getattr(args, "wandb_run_id", None)
            if not isinstance(run_id, str) or not run_id:
                raise RuntimeError("dedicated best-model publication lost its stable run ID")
            checkpoint_path = ensure_local_checkpoint(checkpoint_id)
            started = time.perf_counter()
            result = publish_dedicated_best_model(
                repo_id=str(best_model_repo),
                run_id=run_id,
                checkpoint_dir=checkpoint_path,
                checkpoint_id=checkpoint_id,
                metric=metric,
                validation_loss=-metric,
                token=best_model_token,
                recreate=bool(getattr(args, "best_model_recreate", False)),
            )
            event = {**dict(result), "elapsed_seconds": time.perf_counter() - started}
            emit(json.dumps({"best_model_publication": event}, sort_keys=True), flush=True)
        # The local best pointer protects the checkpoint from retention whether or not a
        # dedicated remote repository exists.
        if keep_last_checkpoints > 0 and checkpoint_id in saved:
            write_json_atomic(best_checkpoint_path, {"checkpoint_id": checkpoint_id, "metric": metric})
            best_local_checkpoint_id = checkpoint_id
        best_model_metric = metric

    def perform_remote_publication(checkpoint_id: str, *, final: bool, metric, step: int,
                                   verified_latest: Mapping[str, object] | None = None):
        nonlocal best_remote_metric
        started = time.perf_counter()
        result = {"latest": verified_latest} if verified_latest is not None else coordinator.publish(
            remote.publisher,
            checkpoint_id=checkpoint_id,
            drive_manifest=remote.drive_manifest,
            metric=None,
            best_metric=None,
        )
        best_updated = False
        if (
            not bool(getattr(remote, "rolling_latest_only", False))
            and metric is not None
            and (best_remote_metric is None or metric > best_remote_metric)
        ):
            latest = result.get("latest")
            if not isinstance(latest, Mapping):
                raise RuntimeError("remote publisher returned no verified latest pointer")
            run_id = remote.drive_manifest.get("run_id")
            store = getattr(remote.publisher, "store", None)
            write_json = getattr(store, "write_json", None)
            if not isinstance(run_id, str) or not run_id or not callable(write_json):
                raise RuntimeError("remote publication cannot write the best pointer")
            promote_best_pointer(store, run_id=run_id, latest=latest,
                                 metric=metric, best_metric=best_remote_metric)
            best_remote_metric = metric
            best_updated = True
        cleanup = cleanup_remote_publication(remote, checkpoint_id=checkpoint_id)
        elapsed = time.perf_counter() - started
        event = {
            "checkpoint_id": checkpoint_id,
            "step": step,
            "elapsed_seconds": elapsed,
            "final": final,
            "best_updated": best_updated,
            "validation_loss": None if metric is None else -metric,
            "rolling_cleanup": cleanup,
        }
        return event

    def observe_publication(event):
        emit(json.dumps({"remote_publication": event}, sort_keys=True), flush=True)
        if telemetry is not None:
            telemetry.log_remote_publication(
                step=event["step"], checkpoint_id=event["checkpoint_id"],
                elapsed_seconds=event["elapsed_seconds"], final=event["final"],
            )
        remotely_published.add(event["checkpoint_id"])

    def publish_remote_checkpoint(checkpoint_id: str, *, final: bool,
                                  verified_latest=None, synchronous=False):
        if remote is None:
            return
        if uploader is not None:
            uploader.poll(wait=True)
        if checkpoint_id in remotely_published:
            return
        ensure_local_checkpoint(checkpoint_id)
        # Capture metadata now. The worker only reads the persisted snapshot,
        # never live tensors, the current validation result, or W&B state.
        kwargs = dict(final=final, metric=_validation_metric(saved_validation[checkpoint_id]),
                      step=engine.global_step, verified_latest=verified_latest)
        if uploader is not None and not synchronous:
            uploader.submit(perform_remote_publication, checkpoint_id, **kwargs)
        else:
            observe_publication(perform_remote_publication(checkpoint_id, **kwargs))

    if remote is not None and getattr(args, "async_checkpoint_upload", False):
        from .async_publication import AsyncPublication
        uploader = AsyncPublication(observe_publication)

    try:
        # A provider can die after a local save but before its remote upload.
        # Repair that boundary before computing more updates on this provider.
        if remote is not None and getattr(remote, "keep_latest_and_best", False) and args.resume:
            validation = read_json(Path(args.checkpoint_dir) / args.resume / "checkpoint.json").get("validation_metrics") or None
            if validation is not None:
                last_validation_step = engine.global_step
            latest = remote.publisher.store.read_json(f"run/{remote.drive_manifest['run_id']}/latest.json")
            publish_remote_checkpoint(args.resume, final=False, synchronous=True,
                verified_latest=latest if latest is not None and latest.get("checkpoint_id") == args.resume else None)
        checkpoint_steps = set(getattr(args, "checkpoint_at_steps", ()))
        if engine.global_step in checkpoint_steps:
            if args.validation_blocks:
                validation = run_validation()
            ensure_local_checkpoint(f"step-{engine.global_step:08d}")
        drained_reason: str | None = None
        for _ in range(args.steps):
            target_tokens = getattr(args, "target_tokens", None)
            if target_tokens is not None and engine.consumed_tokens >= target_tokens:
                break
            if uploader is not None:
                uploader.poll()
            step_started = time.monotonic()
            try:
                with observer.profile(engine.global_step + 1) if observer else nullcontext():
                    metrics = session.step()
            except StopIteration:
                break
            step_seconds = time.monotonic() - step_started
            completed += 1
            if observer is not None:
                observer.training(metrics)
            emit(json.dumps(metrics.as_dict(), sort_keys=True), flush=True)
            if telemetry is not None:
                telemetry.log_training(metrics)
            local_drain = bool(
                max_wall_seconds
                and time.monotonic() - started_wall + step_seconds >= max_wall_seconds
            )
            if distributed is not None:
                local_drain = distributed.vote(local_drain, engine.device)
            source = getattr(session, "source", None)
            block_count = getattr(source, "block_count", None)
            cursor = getattr(source, "last_acknowledged_block_id", None)
            exhausted = (isinstance(block_count, int) and isinstance(cursor, int)
                         and cursor + 1 >= block_count)
            terminal = (completed >= args.steps or exhausted or local_drain
                        or (target_tokens is not None and engine.consumed_tokens >= target_tokens))
            if (
                args.validation_blocks
                and (terminal or engine.global_step in checkpoint_steps
                     or (trainer_config.evaluation_every_steps
                         and engine.global_step % trainer_config.evaluation_every_steps == 0))
            ):
                validation = run_validation()
            checkpoint_id = f"step-{engine.global_step:08d}"
            step_error: BaseException | None = None
            try:
                if (
                    engine.global_step in checkpoint_steps
                    or (trainer_config.checkpoint_every_steps
                        and engine.global_step % trainer_config.checkpoint_every_steps == 0)
                ):
                    ensure_local_checkpoint(checkpoint_id)
                publish_best_model_if_improved(checkpoint_id)
                if remote is not None and engine.global_step % remote.every_steps == 0:
                    publish_remote_checkpoint(checkpoint_id, final=False)
            except BaseException as error:
                step_error = error
            failed_step = (
                distributed.vote(step_error is not None, engine.device)
                if distributed is not None else step_error is not None
            )
            if failed_step:
                if step_error is not None:
                    raise step_error
                raise RuntimeError("distributed checkpoint or publication failed on another rank")
            if local_drain:
                drained_reason = "max_wall_seconds"
            if terminal:
                break

        final_error: BaseException | None = None
        try:
            if args.validation_blocks and last_validation_step != engine.global_step:
                validation = run_validation()
            checkpoint_id = f"step-{engine.global_step:08d}"
            ensure_local_checkpoint(checkpoint_id)
            publish_best_model_if_improved(checkpoint_id)
            publish_remote_checkpoint(checkpoint_id, final=True)
            if uploader is not None:
                uploader.close()
        except BaseException as error:
            final_error = error
        failed = (
            distributed.vote(final_error is not None, engine.device)
            if distributed is not None else final_error is not None
        )
        if failed:
            if final_error is not None:
                raise final_error
            raise RuntimeError("distributed finalization failed on another rank")
        emit(json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True), flush=True)
        if drained_reason is not None:
            drained = {
                "checkpoint_id": checkpoint_id,
                "reason": drained_reason,
                "completed_steps": completed,
                "global_step": engine.global_step,
                "elapsed_seconds": time.monotonic() - started_wall,
                "max_wall_seconds": max_wall_seconds,
            }
            emit(json.dumps({"drained": drained}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except BaseException:
        if uploader is not None:
            try:
                uploader.close()
            except Exception as publication_error:
                sys.stderr.write(f"Checkpoint finalization also failed: {publication_error}\n")
        if observer is not None:
            try:
                observer.finish(failed=True)
            except Exception as observation_error:
                sys.stderr.write(f"Observation finalization also failed: {observation_error}\n")
        if telemetry is not None:
            try:
                telemetry.finish(exit_code=1)
            except Exception as telemetry_error:  # noqa: BLE001 - preserve the primary failure
                sys.stderr.write(
                    "W&B finalization also failed while handling the training error: "
                    f"{type(telemetry_error).__name__}: {telemetry_error}\n"
                )
        raise

    if telemetry is not None:
        telemetry.finish(exit_code=0)
    if observer is not None:
        observer.finish(failed=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
