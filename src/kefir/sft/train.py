"""Supervised fine-tuning of Kefir 1: one loop for fresh starts, exact resumes and forks.

Run in one process, or under ``torchrun --nproc_per_node=2`` (two data-parallel ranks,
each given the same global batch; the core step splits it). The loop is the one the
Kefir 1 SFT runs executed: whole conversations, EOD-padded to 2048, loss = global mean
over the batch's Assistant targets, LR on the committed-target clock, a checkpoint and a
held-out NLL at each observed chat count and at the end of the stream.

Start modes, decided by one classification of the checkpoint against the recipe:
``fresh`` (no checkpoint), ``resume`` (same trainer configuration up to execution
fields: ``load_state_dict``) and ``fork`` (learning-rate recipe differs: the core
``load_fork_state_dict``, which rejects any other change and any LR discontinuity).
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

from kefir.artifact import read_weights, sha256_file, use_pairwise_backend
from kefir.trainer.config import MOE_RESUME_EXECUTION_FIELDS

from .data import Stream, batch_tensors, canonical_sha256, read_rows
from .evaluation import ModelBackend, validation_loss
from .recipe import Execution, Recipe

RECEIPT = "receipt.json"
STATE = "trainer_state.pt"
RESUME_VALIDATION_TOLERANCE = 5e-5


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def is_primary() -> bool:
    """Global rank 0 writes outputs and prints; ``LOCAL_RANK`` only picks the GPU."""
    return int(os.environ.get("RANK", "0")) == 0


def _checked(base: Path, spec: dict) -> Path:
    path = (base / spec["path"]).resolve()
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"input hash mismatch: {spec['path']}: expected {spec['sha256']}, got {actual}")
    return path


def load_inputs(recipe: Recipe, base: Path) -> tuple[Stream, list]:
    """Verify every pinned input and build the one training stream."""
    segments = []
    for segment in recipe.segments:
        rows = read_rows(_checked(base, segment))
        order = (json.loads(_checked(base, segment["order"]).read_text())
                 if segment.get("order") else None)
        segments.append((rows, order))
    stream = Stream.build(segments, batch_size=recipe.global_batch_size, seed=recipe.seed)
    validation = read_rows(_checked(base, recipe.validation))
    if not validation or ({r["conversation_id"] for r in stream.rows}
                          & {r["conversation_id"] for r in validation}):
        raise ValueError("empty or overlapping validation split")
    if recipe.observe_at_chats and recipe.observe_at_chats[-1] > len(stream.rows):
        raise ValueError("observe_at_chats beyond the end of the stream")
    return stream, validation


def classify_start(saved_config: dict, target_config: dict) -> str:
    """``resume`` when only execution fields differ, else ``fork``."""
    changed = {key for key in set(saved_config) | set(target_config)
               if saved_config.get(key) != target_config.get(key)}
    return "fork" if changed - MOE_RESUME_EXECUTION_FIELDS else "resume"


def resolve_device(device: str, local_rank: int) -> str:
    """``cuda`` means this rank's GPU (``cuda:LOCAL_RANK``); any other value is used as given."""
    return f"cuda:{local_rank}" if device == "cuda" else device


def train(recipe: Recipe, base: str | Path, execution: Execution, *,
          resume: str | Path | None = None) -> dict:
    import torch
    import torch.distributed as dist
    from kefir.model.distributed import prepare_distributed_engine
    from kefir.model.engine import MoETrainerEngine
    from kefir.model.evaluation import normalize_moe_model_config
    from kefir.model.model import MoESmallLLM
    from kefir.trainer.distributed import initialize_distributed, shutdown_distributed
    from kefir.trainer.state import load_trainer_state_file

    # Every check that can fail runs identically on every rank before the first collective.
    base = Path(base)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2):
        raise ValueError("Kefir 1 SFT runs on one or two ranks")
    cuda = execution.device.startswith("cuda")
    if world == 2 and cuda and execution.device != "cuda":
        raise ValueError("under torchrun each rank uses cuda:LOCAL_RANK; set execution device to 'cuda'")
    if recipe.precision == "fp16" and not cuda:
        raise ValueError("an fp16 recipe trains on CUDA only; use an fp32 recipe on CPU")
    output = Path(execution.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    stream, validation = load_inputs(recipe, base)
    config = recipe.trainer_config(stream, microbatch_size=execution.microbatch_size)
    device = resolve_device(execution.device, int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_num_threads(execution.threads)
    if cuda:
        torch.cuda.set_device(torch.device(device))
    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    context = initialize_distributed(device) if world == 2 else None
    primary = is_primary()
    barrier = dist.barrier if context is not None else (lambda: None)
    if primary:
        output.mkdir(parents=True)
    barrier()
    start = time.perf_counter()
    cursor = {"chats": 0, "targets": 0, "step": 0}
    recipe_sha = canonical_sha256(recipe.as_dict())

    def emit(kind, **values):
        if primary:
            event = dict(kind=kind, run_id=recipe.run_id, chats_seen=cursor["chats"],
                         assistant_tokens_seen=cursor["targets"], step=cursor["step"],
                         elapsed_seconds=time.perf_counter() - start, **values)
            with (output / "events.jsonl").open("a") as stream_out:
                stream_out.write(json.dumps(event, allow_nan=False) + "\n")

    try:
        parent = _checked(base, recipe.parent)
        raw_config, tensors, _ = read_weights(parent, trusted=True)
        model = MoESmallLLM(normalize_moe_model_config(raw_config))
        model.load_state_dict(tensors, strict=True)
        del tensors
        use_pairwise_backend(model)
        engine = MoETrainerEngine(model, config, device=device, optimizer_telemetry=True)
        if config.precision == "fp16":
            engine.scaler = torch.amp.GradScaler("cuda", init_scale=recipe.initial_loss_scale)
        if context is not None:
            prepare_distributed_engine(engine, context)
        # Held-out NLL runs in FP16 for an fp16 recipe and in FP32 otherwise (bf16 included).
        backend = ModelBackend(model, device=device,
                               precision="fp16" if config.precision == "fp16" else "fp32")

        def held_out():
            """Validation NLL on rank 0 with RNG, buffers and train mode restored."""
            python_state, cpu_state = random.getstate(), torch.get_rng_state()
            gpu_state = torch.cuda.get_rng_state(torch.device(device)) if cuda else None
            buffers = {name: value.clone() for name, value in model.named_buffers()}
            training = model.training
            try:
                model.eval()
                return validation_loss(backend, validation)
            finally:
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
                model.train(training)
                random.setstate(python_state)
                torch.set_rng_state(cpu_state)
                if gpu_state is not None:
                    torch.cuda.set_rng_state(gpu_state, torch.device(device))

        mode, parent_checkpoint = "fresh", None
        if resume is not None:
            folder = Path(resume)
            receipt = json.loads((folder / RECEIPT).read_text())
            digest = sha256_file(folder / STATE)
            if digest != receipt["checkpoint_sha256"]:
                raise ValueError("checkpoint differs from its receipt")
            state = load_trainer_state_file(folder / STATE, map_location="cpu")
            steps = state["global_step"]
            if (steps != receipt["step"] or state["consumed_tokens"] != receipt["assistant_tokens_seen"]
                    or stream.prefix_sha256(steps) != receipt["stream_prefix_sha256"]
                    or stream.targets_after(steps) != state["consumed_tokens"]
                    or receipt["parent_sha256"] != recipe.parent["sha256"]):
                raise ValueError("checkpoint was not produced by a prefix of this stream and parent")
            mode = classify_start(state["config"], config.as_dict())
            if mode == "resume":
                engine.load_state_dict(state)
            else:
                engine.load_fork_state_dict(state)
            del state
            model.train()
            cursor.update(chats=stream.chats_after(steps), targets=engine.consumed_tokens, step=steps)
            if engine.global_step != steps:
                raise RuntimeError("resume cursor mismatch")
            parent_checkpoint = digest
            emit(f"{mode}_verified", checkpoint_sha256=digest, config=engine.config.as_dict(),
                 scheduler=engine.scheduler.state_dict())
            if primary and receipt.get("validation"):
                value = held_out()
                delta = abs(value["assistant_nll"] - receipt["validation"]["assistant_nll"])
                if delta > RESUME_VALIDATION_TOLERANCE:
                    raise ValueError("resumed validation differs from the checkpoint receipt")
                emit("resume_validation_verified", absolute_delta=delta)
        barrier()
        emit("setup", mode=mode, recipe_sha256=recipe_sha, parent_sha256=recipe.parent["sha256"],
             world_size=world, device=device, validation_precision=backend.precision,
             torch_version=torch.__version__,
             config=config.as_dict(), batches=len(stream.batches), chats=len(stream.rows))

        def observe(save: bool) -> None:
            barrier()
            if primary:
                before = time.perf_counter()
                value = held_out()
                if save:
                    folder = output / f"exposure-{cursor['chats']:05d}"
                    folder.mkdir()
                    engine.save_checkpoint_state(folder / STATE)
                    digest = sha256_file(folder / STATE)
                    _atomic_json(folder / RECEIPT, {
                        "schema": "kefir-sft-checkpoint/1", "run_id": recipe.run_id,
                        "recipe_sha256": recipe_sha, "parent_sha256": recipe.parent["sha256"],
                        "start": {"mode": mode, "checkpoint_sha256": parent_checkpoint},
                        "chats_seen": cursor["chats"], "assistant_tokens_seen": cursor["targets"],
                        "step": cursor["step"], "stream_prefix_sha256": stream.prefix_sha256(cursor["step"]),
                        "checkpoint_sha256": digest, "trainer_config": engine.config.as_dict(),
                        "validation": value})
                    emit("checkpoint", checkpoint=str((folder / STATE).relative_to(output)), sha256=digest)
                emit("validation", validation=value, seconds=time.perf_counter() - before)
            barrier()

        if mode == "fresh":
            observe(save=False)
        milestones = sorted(set(recipe.observe_at_chats) | {len(stream.rows)})
        for batch in stream.batches[cursor["step"]:]:
            metrics = engine.train_batch(batch_tensors([stream.rows[i] for i in batch], cursor["step"]))
            if cuda:
                torch.cuda.synchronize()
            previous = cursor["chats"]
            cursor["chats"] += len(batch)
            cursor["targets"] += metrics.target_tokens
            cursor["step"] += 1
            if engine.consumed_tokens != cursor["targets"] or engine.global_step != cursor["step"]:
                raise RuntimeError("engine token/step cursor mismatch")
            emit("train", metrics=asdict(metrics), batch_chats=len(batch))
            if any(previous < n <= cursor["chats"] for n in milestones):
                observe(save=True)
        if cursor["chats"] != len(stream.rows) or not math.isfinite(engine.scheduler.last_lr):
            raise RuntimeError("incomplete stream")
        result = {"status": "completed", "mode": mode, "steps": cursor["step"],
                  "chats": cursor["chats"], "assistant_targets": cursor["targets"],
                  "final_learning_rate": engine.scheduler.last_lr,
                  "checkpoint": str(output / f"exposure-{cursor['chats']:05d}")}
        emit("complete", **result)
        return result
    except BaseException as error:
        emit("failed", error_type=type(error).__name__, error=str(error))
        if context is not None:
            import sys
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A single failed rank must not enter collective teardown; torchrun ends its peer.
            os._exit(1)
        raise
    finally:
        if context is not None:
            shutdown_distributed()


__all__ = ["classify_start", "is_primary", "load_inputs", "train"]
