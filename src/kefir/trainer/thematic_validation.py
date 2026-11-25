"""Frozen thematic panels scored beside the periodic held-out validation.

The panels are 12 immutable ``<domain>.npz`` buckets of right-padded 2,048-token
windows, one window per document, with no cross-document context. The semantics
reproduced here are the ones of the read-only reference evaluator kept with the
panels (``thematic-validation-v1/evaluate.py``):

* a bucket loss is ``sum(NLL) / sum(valid targets)`` — never a mean of window
  means, because the windows carry very different target counts;
* only ``mask == 1`` positions are scored, so right padding (token id 0) never
  enters the sum;
* the logits are scored over the 8,000-entry semantic vocabulary, not over the
  padded embedding width.

The reference scores one cropped document per forward; this module scores whole
right-padded windows in batches, which is identical for a causal model because
no scored position can attend to a later padding position.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .precision import autocast_context

MANIFEST_FILENAME = "manifest.json"
SEMANTIC_VOCAB_SIZE = 8_000
PADDING_TOKEN_ID = 0
FIRST_SPECIAL_TOKEN_ID = 7_992
RESERVED_RESULT_KEYS = frozenset({"weighted_loss", "elapsed_seconds"})

_PANEL_CACHE: dict[Path, "ThematicPanels"] = {}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_length(inputs: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> int:
    """Return the scored prefix length, rejecting any window the panel contract forbids."""

    if inputs.ndim != 1 or inputs.shape != targets.shape or inputs.shape != mask.shape:
        raise ValueError("thematic window shapes disagree")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("thematic window mask is not binary")
    length = int(mask.sum())
    if length < 1 or not np.all(mask[:length] == 1) or np.any(mask[length:]):
        raise ValueError("thematic window mask is not a non-empty prefix")
    if np.any(inputs[:length] >= FIRST_SPECIAL_TOKEN_ID) or np.any(
        targets[:length] >= FIRST_SPECIAL_TOKEN_ID
    ):
        raise ValueError("thematic window contains a reserved token id")
    if not np.array_equal(inputs[1:length], targets[: length - 1]):
        raise ValueError("thematic window is not a one-position causal shift")
    return length


@dataclass(frozen=True, slots=True)
class ThematicBucket:
    """One immutable domain panel, already verified against the manifest."""

    name: str
    input_ids: np.ndarray
    target_ids: np.ndarray
    mask: np.ndarray
    target_tokens: int

    @property
    def window_count(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass(frozen=True, slots=True)
class ThematicPanels:
    """The frozen panel set, loaded and hash-verified exactly once per directory."""

    directory: Path
    version: str
    context: int
    tokenizer_sha256: str
    buckets: tuple[ThematicBucket, ...]

    @property
    def target_tokens(self) -> int:
        return sum(bucket.target_tokens for bucket in self.buckets)

    @property
    def window_count(self) -> int:
        return sum(bucket.window_count for bucket in self.buckets)


def load_thematic_panels(directory: str | Path) -> ThematicPanels:
    """Load and verify the frozen panels once; drift against the manifest fails closed."""

    root = Path(directory).resolve()
    cached = _PANEL_CACHE.get(root)
    if cached is not None:
        return cached

    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"thematic panels have no {MANIFEST_FILENAME}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("thematic panel manifest must be a JSON object")
    bucket_manifests = manifest.get("buckets")
    if not isinstance(bucket_manifests, Mapping) or not bucket_manifests:
        raise ValueError("thematic panel manifest has no buckets")
    context = manifest.get("context")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("thematic panel manifest has no positive context length")

    buckets: list[ThematicBucket] = []
    for name in sorted(bucket_manifests):
        entry = bucket_manifests[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"thematic bucket {name} has no manifest object")
        if name in RESERVED_RESULT_KEYS:
            raise ValueError(f"thematic bucket name {name!r} collides with a reserved result key")
        path = root / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"thematic bucket file is missing: {path}")
        expected_digest = entry.get("sha256")
        if not isinstance(expected_digest, str) or not expected_digest:
            raise ValueError(f"thematic bucket {name} has no manifest sha256")
        actual_digest = _file_digest(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"thematic bucket {name} drifted from the manifest: "
                f"{actual_digest} != {expected_digest}"
            )
        with np.load(path) as arrays:
            input_ids = np.array(arrays["input_ids"])
            target_ids = np.array(arrays["target_ids"])
            mask = np.array(arrays["mask"])
        if input_ids.ndim != 2 or input_ids.shape[1] != context:
            raise ValueError(f"thematic bucket {name} does not hold {context}-token windows")
        expected_targets = entry.get("target_tokens")
        if isinstance(expected_targets, bool) or not isinstance(expected_targets, int):
            raise ValueError(f"thematic bucket {name} has no manifest target_tokens")
        observed = 0
        for window, targets, window_mask in zip(input_ids, target_ids, mask, strict=True):
            observed += _valid_length(window, targets, window_mask)
        if observed != expected_targets:
            raise ValueError(
                f"thematic bucket {name} has {observed} valid targets, "
                f"manifest declares {expected_targets}"
            )
        buckets.append(
            ThematicBucket(
                name=name,
                input_ids=input_ids,
                target_ids=target_ids,
                mask=mask,
                target_tokens=observed,
            )
        )

    panels = ThematicPanels(
        directory=root,
        version=str(manifest.get("version", "")),
        context=context,
        tokenizer_sha256=str(manifest.get("tokenizer_sha256", "")),
        buckets=tuple(buckets),
    )
    _PANEL_CACHE[root] = panels
    return panels


@torch.inference_mode()
def evaluate_thematic(
    model: torch.nn.Module,
    *,
    device: torch.device,
    microbatch: int,
    panels: ThematicPanels,
    precision: str = "fp32",
    semantic_vocab_size: int = SEMANTIC_VOCAB_SIZE,
) -> dict[str, object]:
    """Return one token-weighted NLL per frozen domain panel, plus their pooled loss.

    The model is left exactly as it was found: no optimizer touch, no parameter
    write, and the original ``training`` flag restored.
    """

    if isinstance(microbatch, bool) or not isinstance(microbatch, int) or microbatch <= 0:
        raise ValueError("thematic microbatch must be a positive integer")
    if not panels.buckets:
        raise ValueError("thematic panels hold no buckets")

    started = time.perf_counter()
    was_training = model.training
    model.eval()
    result: dict[str, object] = {}
    total_nll, total_targets = 0.0, 0
    try:
        for bucket in panels.buckets:
            bucket_nll, bucket_targets = 0.0, 0
            for start in range(0, bucket.window_count, microbatch):
                stop = min(bucket.window_count, start + microbatch)
                inputs = torch.from_numpy(
                    bucket.input_ids[start:stop].astype(np.int64)
                ).to(device=device)
                targets = torch.from_numpy(
                    bucket.target_ids[start:stop].astype(np.int64)
                ).to(device=device)
                scored = torch.from_numpy(
                    bucket.mask[start:stop].astype(np.bool_)
                ).to(device=device)
                with autocast_context(precision, device):
                    logits = model(inputs)
                if logits.shape[-1] < semantic_vocab_size:
                    raise ValueError("model logits are narrower than the semantic vocabulary")
                if logits.shape[-1] > semantic_vocab_size:
                    logits = logits.narrow(-1, 0, semantic_vocab_size)
                losses = F.cross_entropy(
                    logits.reshape(-1, semantic_vocab_size).float(),
                    targets.reshape(-1),
                    reduction="none",
                )
                masked = losses.double() * scored.reshape(-1).double()
                chunk_nll = float(masked.sum())
                if not math.isfinite(chunk_nll) or chunk_nll < 0:
                    raise FloatingPointError("non-finite thematic panel NLL")
                bucket_nll += chunk_nll
                bucket_targets += int(scored.sum().item())
                del logits, losses, masked, inputs, targets, scored
            if bucket_targets != bucket.target_tokens:
                raise RuntimeError(f"thematic bucket {bucket.name} lost targets during evaluation")
            result[bucket.name] = {
                "loss": bucket_nll / bucket_targets,
                "target_tokens": bucket_targets,
            }
            total_nll += bucket_nll
            total_targets += bucket_targets
    finally:
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if total_targets == 0:
        raise RuntimeError("thematic panels yielded no scored targets")
    result["weighted_loss"] = total_nll / total_targets
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


__all__ = [
    "SEMANTIC_VOCAB_SIZE",
    "ThematicBucket",
    "ThematicPanels",
    "evaluate_thematic",
    "load_thematic_panels",
]
