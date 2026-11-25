"""Deterministic synthetic SFT inputs and a tiny Kefir-shaped MoE for CPU tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from kefir.artifact import save_weights
from kefir.model.config import MoEModelConfig
from kefir.model.model import MoESmallLLM
from kefir.sft.data import render_rows
from kefir.sft.renderer import ChatRenderer

WORDS = ("river stone lamp quiet orange window garden seven bright cloud paper "
         "violin morning tea north letter bridge silver forest number").split()


def tiny_config() -> MoEModelConfig:
    """Full 2048 context and 8000 vocabulary (the SFT geometry), tiny width."""
    return MoEModelConfig.accepted(
        max_seq_len=2048, d_model=16, n_layers=4, d_ff=32, n_heads=2, head_dim=8,
        gdn_num_key_heads=2, gdn_num_value_heads=2, gdn_key_dim=8, gdn_value_dim=8,
        gdn_chunk_size=64, num_experts=4, expert_d_ff=16,
    )


def conversation(index: int) -> dict:
    """A whole chat: optional System, 1-3 exchanges, varying lengths."""
    def text(offset, length):
        return " ".join(WORDS[(index * 7 + offset + k) % len(WORDS)] for k in range(length))
    messages = []
    if index % 3 == 0:
        messages.append({"role": "system", "content": "Answer briefly and kindly."})
    for turn in range(1 + index % 3):
        messages.append({"role": "user", "content": f"Question {index}.{turn}: {text(turn, 3 + index % 5)}?"})
        messages.append({"role": "assistant", "content": text(turn + 11, 2 + (index + turn) % 7) + "."})
    return {"conversation_id": f"chat-{index:03d}", "messages": messages}


def write_rows(path: Path, indexes) -> str:
    render_rows([conversation(i) for i in indexes], path, renderer=ChatRenderer.load())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_parent(path: Path, seed: int = 5) -> Path:
    torch.manual_seed(seed)
    return save_weights(MoESmallLLM(tiny_config()), path)


def write_inputs(root: Path, *, segments=((0, 7), (7, 12)), validation=(12, 14),
                 frozen_order: bool = True) -> dict:
    """Write parent, segment rows (+ optional frozen orders) and validation under root."""
    root.mkdir(parents=True, exist_ok=True)
    parent = write_parent(root / "parent.safetensors")
    specs = []
    for number, (first, last) in enumerate(segments):
        rows = root / f"segment-{number}.jsonl"
        write_rows(rows, range(first, last))
        spec = pinned(rows, root)
        if frozen_order:
            order = list(reversed(range(last - first)))
            order = order[1::2] + order[0::2]
            path = root / f"order-{number}.json"
            path.write_text(json.dumps(order))
            spec["order"] = pinned(path, root)
        specs.append(spec)
    held_out = root / "validation.jsonl"
    write_rows(held_out, range(*validation))
    return {"parent": pinned(parent, root), "segments": specs, "validation": pinned(held_out, root)}


def recipe(inputs: dict, *, schedule=None, batch=3, seed=17, observe=None, lr=2e-3) -> dict:
    return {"run_id": "tiny-sft", **inputs, "global_batch_size": batch, "seed": seed,
            "precision": "fp32",
            "optimizer": {"optimizer": "hybrid_muon_adamw", "learning_rate": lr,
                          "weight_decay": 0.1, "max_grad_norm": 1.0},
            "schedule": schedule or {"kind": "constant"},
            "observe_at_chats": observe or []}


def write_recipe(root: Path, value: dict, name: str = "recipe.json") -> Path:
    path = root / name
    path.write_text(json.dumps(value, indent=2))
    return path
