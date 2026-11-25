"""Exact logical MoE updates across data-parallel workers."""

from __future__ import annotations

from contextlib import nullcontext
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from kefir.layers.fused_loss import chunked_cross_entropy_sum
from kefir.trainer.precision import autocast_context
from kefir.trainer.step import (
    _backoff_scaler_after_nonfinite_gradient,
    _clear_optimizer_step_statistics,
    _fp16_overflow_retry_limit,
    _microbatch_to_device,
    _optimizer_gradient_norms,
    _optimizer_step_statistics,
    _ordered_batch_tensors,
)
from kefir.trainer.types import IGNORE_INDEX, TokenBatch

from .metrics import MoEStepMetrics
from .step import _accumulate_telemetry, _empty_telemetry, _finalize_telemetry


class _LossModel(nn.Module):
    def __init__(self, raw_model: nn.Module) -> None:
        super().__init__()
        self.raw = raw_model

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        hidden, aux = self.raw.hidden_states_with_aux(inputs)
        if hidden.shape[:2] != labels.shape:
            raise RuntimeError("distributed MoE hidden states and labels disagree")
        ce_sum = chunked_cross_entropy_sum(
            hidden, self.raw.token_embedding.weight, labels,
            semantic_vocab_size=self.raw.config.semantic_vocab_size,
            ignore_index=IGNORE_INDEX,
        )
        return ce_sum, aux.z_loss, aux.layers


def prepare_distributed_engine(engine: object, context: object) -> None:
    raw = engine.raw_model
    if raw.config.dense.dropout != 0:
        raise RuntimeError("distributed RNG migration requires dropout-free training")
    if context.world_size != 2:
        raise ValueError("Kefir v1 distributed execution requires exactly two ranks")
    engine.distributed = context
    engine.model = DistributedDataParallel(
        _LossModel(raw),
        device_ids=[context.local_rank] if engine.device.type == "cuda" else None,
        output_device=context.local_rank if engine.device.type == "cuda" else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def _merge_router_frontiers(raw: object, telemetry: list[dict[str, object]], world: int) -> None:
    for block, item in zip(raw.blocks, telemetry, strict=True):
        router = block.ffn.router
        counts = item["counts"]
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        for key in ("selected_probability_sum", "entropy_sum"):
            dist.all_reduce(item[key], op=dist.ReduceOp.SUM)
        token_count = torch.tensor(int(item["token_count"]), device=counts.device)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        item["token_count"] = int(token_count.item())
        if router.balancing == "quantile":
            frontier, target = router._score_frontier, router._quantile_target
            if frontier is None or target is None or frontier.shape[1] < target:
                raise RuntimeError("distributed router has no complete score frontier")
            gathered = [torch.empty_like(frontier) for _ in range(world)]
            dist.all_gather(gathered, frontier.contiguous())
            router._score_frontier = torch.cat(gathered, dim=1).topk(target, dim=1).values
        router.commit_load(counts)


def distributed_moe_train_step(engine: object, batch: TokenBatch) -> MoEStepMetrics:
    context = engine.distributed
    raw, wrapped = engine.raw_model, engine.model
    context.agree(
        (batch.block_id, batch.sequence_count, batch.target_token_count,
         int(batch.split == "train"), engine.config.microbatch_size), engine.device
    )
    if batch.split != "train" or batch.sequence_count <= 0:
        raise ValueError("distributed training requires a nonempty train block")
    started = time.perf_counter()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    input_ids, labels = _ordered_batch_tensors(batch)
    total_positions = int(input_ids.numel())
    if total_positions <= 0:
        raise RuntimeError("distributed block has no input positions")
    # Replicate an uneven terminal block: DDP's mean preserves its gradient and
    # both ranks execute the same collective sequence, even for a single row.
    replicated = bool(batch.sequence_count % context.world_size)
    rows_per_rank = batch.sequence_count if replicated else batch.sequence_count // context.world_size
    first = 0 if replicated else context.rank * rows_per_rank
    last = first + rows_per_rank
    size = engine.config.microbatch_size
    if size <= 0:
        raise ValueError("microbatch size must be positive")
    retries = 0
    overflow_retry_limit = (
        _fp16_overflow_retry_limit(engine.scaler, engine.config.max_overflow_retries)
        if engine.scaler.is_enabled() else engine.config.max_overflow_retries
    )
    scaler_scale = float(engine.scaler.get_scale())
    z_coefficient = float(raw.config.router_z_loss_coefficient)
    while True:
        engine.optimizer.zero_grad(set_to_none=True)
        next_tokens = engine.consumed_tokens + batch.target_token_count
        lr = engine.scheduler.prepare_step(next_tokens)
        local_ce = torch.zeros((), dtype=torch.float32, device=engine.device)
        local_z = torch.zeros((), dtype=torch.float32, device=engine.device)
        telemetry = _empty_telemetry(engine)
        for block in raw.blocks:
            block.ffn.router.begin_step(total_positions)
        for start in range(first, last, size):
            micro_inputs, micro_labels = _microbatch_to_device(
                input_ids, labels, start=start, stop=min(start + size, last), device=engine.device
            )
            sync = wrapped.no_sync() if start + size < last else nullcontext()
            with sync:
                with autocast_context(engine.config.precision, engine.device):
                    ce_sum, z_loss, layers = wrapped(micro_inputs, micro_labels)
                    fraction = float(micro_inputs.numel()) / float(total_positions)
                    objective = ce_sum / batch.target_token_count + z_coefficient * z_loss * fraction
                local_bad = not bool(torch.isfinite(ce_sum) & torch.isfinite(z_loss) & torch.isfinite(objective))
                if context.vote(local_bad, engine.device):
                    engine.scheduler.cancel_step()
                    engine.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("non-finite distributed MoE forward; block unacknowledged")
                engine.scaler.scale(objective).backward()
            local_ce += ce_sum.detach().float()
            local_z += z_loss.detach().float() * fraction
            _accumulate_telemetry(telemetry, layers)

        engine.scaler.unscale_(engine.optimizer)
        if not replicated:
            # Normalize after the FP16 backward so its scale is topology-independent.
            for parameter in raw.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(context.world_size)
        role_gradient_norms = _optimizer_gradient_norms(engine.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            raw.parameters(), engine.config.max_grad_norm
        )
        grad_value = float(gradient_norm.detach())
        if context.vote(not bool(torch.isfinite(gradient_norm)), engine.device):
            engine.scheduler.cancel_step()
            _clear_optimizer_step_statistics(engine.optimizer)
            engine.optimizer.zero_grad(set_to_none=True)
            if not engine.scaler.is_enabled():
                raise FloatingPointError("non-finite distributed MoE gradient; block unacknowledged")
            scaler_scale = _backoff_scaler_after_nonfinite_gradient(engine.scaler)
            retries += 1
            engine.overflow_events += 1
            if retries > overflow_retry_limit:
                raise FloatingPointError("distributed FP16 gradient remained non-finite")
            continue

        gradient_clipped = grad_value > float(engine.config.max_grad_norm)
        scale_before = float(engine.scaler.get_scale())
        _clear_optimizer_step_statistics(engine.optimizer)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        scaler_scale = float(engine.scaler.get_scale())
        if context.vote(scaler_scale < scale_before, engine.device):
            engine.scheduler.cancel_step()
            retries += 1
            engine.overflow_events += 1
            if retries > overflow_retry_limit:
                raise FloatingPointError("distributed FP16 optimizer overflowed")
            continue
        update_statistics = _optimizer_step_statistics(engine.optimizer)
        engine.consumed_tokens = next_tokens
        engine.global_step += 1
        engine.scheduler.commit(next_tokens)
        if replicated:
            for block, item in zip(raw.blocks, telemetry, strict=True):
                block.ffn.router.commit_load(item["counts"])
        else:
            _merge_router_frontiers(raw, telemetry, context.world_size)
            dist.all_reduce(local_ce, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_z, op=dist.ReduceOp.SUM)
        moe = _finalize_telemetry(engine, telemetry) if context.primary else {}
        break

    ce = float(local_ce / batch.target_token_count)
    z = float(local_z)
    elapsed = max(time.perf_counter() - started, 1e-12)
    peak = torch.cuda.max_memory_allocated(engine.device) if engine.device.type == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved(engine.device) if engine.device.type == "cuda" else 0
    memory = torch.tensor([peak, reserved], dtype=torch.int64, device=engine.device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    peak, reserved = int(memory[0].item()), int(memory[1].item())
    return MoEStepMetrics(
        step=engine.global_step, block_id=batch.block_id, loss=ce,
        objective_loss=ce + z_coefficient * z, router_z_loss=z,
        router_z_loss_coefficient=z_coefficient, learning_rate=lr,
        gradient_norm=grad_value, sequences=batch.sequence_count,
        target_tokens=batch.target_token_count, consumed_tokens=engine.consumed_tokens,
        elapsed_seconds=elapsed, tokens_per_second=batch.target_token_count / elapsed,
        overflow_retries=retries, peak_memory_bytes=peak,
        grad_scaler_scale=scaler_scale, gradient_clipped=gradient_clipped,
        overflow_events_total=engine.overflow_events, peak_reserved_memory_bytes=reserved,
        optimizer_gradient_norms=role_gradient_norms,
        optimizer_update_statistics=update_statistics, moe=moe,
    )


__all__ = ["distributed_moe_train_step", "prepare_distributed_engine"]
