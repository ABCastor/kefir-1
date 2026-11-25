"""The one inference weights form: a single ``.safetensors`` file.

The file holds the model tensors and, in its header metadata, the model configuration
as JSON (``model_config``). This is the form of the released Kefir 1 weights. Every
inference entry (generation, chat, evaluation, SFT parent) normalizes through
``read_weights``; a pickled trainer state is read only when the caller states trust.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kefir.layers.gdn2_pairwise import PairwiseGDN2Backend
from kefir.model.evaluation import normalize_moe_model_config
from kefir.model.model import MoESmallLLM

from .tokenizer import SEMANTIC_VOCAB_SIZE, TOKENIZER_SHA256


SUFFIX = ".safetensors"
# The GDN-2 recurrence used by every Kefir 1 SFT run, its evaluation and its chat.
PAIRWISE_CHUNK = 32


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def use_pairwise_backend(model: MoESmallLLM) -> MoESmallLLM:
    """Install the pairwise FP32 GDN-2 backend (chunk 32) on every recurrent mixer."""
    for block in model.blocks:
        if hasattr(block.mixer, "backend"):
            block.mixer.backend = PairwiseGDN2Backend(PAIRWISE_CHUNK)
    return model


def save_weights(model: MoESmallLLM, destination: str | Path, *,
                 metadata: dict[str, str] | None = None) -> Path:
    """Write tensors plus ``model_config`` metadata to a new file, atomically."""
    if model.config.semantic_vocab_size != SEMANTIC_VOCAB_SIZE:
        raise ValueError("Kefir weights require the pinned 8000-token vocabulary")
    path = Path(destination)
    if path.suffix != SUFFIX:
        raise ValueError(f"weights file must end with {SUFFIX}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    extra = dict(metadata or {})
    reserved = {"format", "model_config", "tokenizer_sha256"} & set(extra)
    if reserved or any(not isinstance(k, str) or not isinstance(v, str) for k, v in extra.items()):
        raise ValueError("metadata must map strings to strings and not redefine reserved keys")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().cpu().contiguous().clone()
               for name, tensor in model.state_dict().items()}
    header = {"format": "pt",
              "model_config": json.dumps(model.config.as_dict(), sort_keys=True,
                                         separators=(",", ":")),
              "tokenizer_sha256": TOKENIZER_SHA256, **extra}
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        save_file(tensors, temporary, metadata=header)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def read_weights(path: str | Path, *, expected_sha256: str | None = None,
                 trusted: bool = False) -> tuple[dict, dict, dict[str, str]]:
    """Return ``(model_config, tensors, info)`` from weights or a trusted trainer state.

    A trainer state (``.pt``/``.pkl``) is a pickle that can execute code: it is read
    only with ``trusted=True``. ``info`` carries the file SHA-256 and any metadata.
    """
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"expected a Kefir weights file ({SUFFIX}); got {file}")
    digest = sha256_file(file)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("weights checksum mismatch")
    if file.suffix == SUFFIX:
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            metadata = dict(reader.metadata() or {})
        raw = metadata.get("model_config")
        if not raw:
            raise ValueError("weights file has no model_config metadata")
        tokenizer = metadata.get("tokenizer_sha256")
        if tokenizer is not None and tokenizer != TOKENIZER_SHA256:
            raise ValueError("weights file uses a different tokenizer")
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ValueError("model_config metadata must be a JSON object")
        return config, load_file(str(file), device="cpu"), {**metadata, "sha256": digest}
    if not trusted:
        raise ValueError(
            f"expected a Kefir weights file ({SUFFIX}); a trainer state is a pickle and is "
            "read only when trusted (export trusted checkpoints first)")
    from kefir.trainer.state import load_trainer_state_file
    state = load_trainer_state_file(file, map_location="cpu")
    return state["model_config"], state["model"], {"sha256": digest,
                                                   "global_step": str(state.get("global_step"))}


def load_weights(path: str | Path, *, device: torch.device | str = "cpu",
                 expected_sha256: str | None = None, trusted: bool = False):
    """Build the model from weights, strict, pairwise backend, evaluation mode."""
    raw, tensors, info = read_weights(path, expected_sha256=expected_sha256, trusted=trusted)
    config = normalize_moe_model_config(raw)
    if config.semantic_vocab_size != SEMANTIC_VOCAB_SIZE:
        raise ValueError("weights vocabulary disagrees with the pinned tokenizer")
    model = MoESmallLLM(config)
    model.load_state_dict(tensors, strict=True)
    del tensors
    use_pairwise_backend(model).to(torch.device(device)).eval()
    return model, config, info


__all__ = ["PAIRWISE_CHUNK", "load_weights", "read_weights", "save_weights", "sha256_file",
           "use_pairwise_backend"]
