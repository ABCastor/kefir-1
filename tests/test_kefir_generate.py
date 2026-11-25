from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from kefir.generate import generate_text


class _EodModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*input_ids.shape, 8000), device=input_ids.device)
        logits[..., 7992] = 10.0
        logits[..., 7993:] = 100.0
        return logits


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_public_generation_uses_v2_text_tokens_and_stops_at_eod(temperature) -> None:
    model = _EodModel()
    with patch("kefir.generate.load_weights",
               return_value=(model, SimpleNamespace(max_seq_len=2048), {})):
        result = generate_text(Path("unused-checkpoint"), ["Hello <think>", "Ciao"],
                               max_new_tokens=4, device="cpu", precision="fp32",
                               temperature=temperature, top_k=1)
    assert result == ["", ""]
