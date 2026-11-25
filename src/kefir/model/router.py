"""FP32 routing with selection-only bias and exact per-step quantile balancing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TopKRouting:
    logits: Tensor
    expert_indices: Tensor          # [tokens, k]
    selected_probabilities: Tensor  # [tokens, k], mixture weights
    z_loss: Tensor
    entropy_sum: Tensor
    token_count: int
    expert_counts: Tensor           # [num_experts], assignments not tokens



class SwitchTopKRouter(nn.Module):
    """Bias-free FP32 router with differentiable selected affinity weights."""

    def __init__(
        self, d_model: int, num_experts: int, *, init_std: float,
        top_k: int = 1, scoring: str = "softmax",
        balancing: str = "none",
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_experts <= 1 or init_std <= 0:
            raise ValueError("invalid router geometry")
        if scoring not in {"softmax", "sqrt_softplus"}:
            raise ValueError("unsupported router scoring")
        if scoring == "sqrt_softplus" and int(top_k) < 2:
            raise ValueError("sqrt_softplus affinities require top_k >= 2")
        self.scoring = scoring
        if balancing not in {"none", "quantile"}:
            raise ValueError("unsupported balancing controller")
        self.balancing = balancing
        # Transient within one logical optimizer step; never checkpoint state.
        self._quantile_target: int | None = None
        self._score_frontier: Tensor | None = None
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError("top_k must be between 1 and num_experts")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.init_std = float(init_std)
        self.projection = nn.Linear(d_model, num_experts, bias=False)
        self.register_buffer("selection_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("token_exposure", torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer("expert_updates", torch.zeros(num_experts, dtype=torch.long))
        self.reset_router_parameters()

    def begin_step(self, expected_tokens: int) -> None:
        """Open one logical optimizer step for Quantile Balancing.

        The controller needs the exact per-expert score quantile at rate K/E over the
        whole step, so the target rank is fixed up front from the step's token count and
        one frontier of the top-ranked scores is carried across gradient-accumulation
        microbatches. Keeping exactly that many scores makes the merge exact: an element
        of the final top-M of the union survives every intermediate top-M.
        """

        if self.balancing != "quantile":
            return
        if expected_tokens <= 0:
            raise ValueError("a logical step must route at least one token")
        target = round(expected_tokens * self.top_k / self.num_experts)
        self._quantile_target = max(1, min(int(expected_tokens), int(target)))
        self._score_frontier = None

    @torch.no_grad()
    def _observe_scores(self, scores: Tensor) -> None:
        if self.balancing != "quantile" or self._quantile_target is None or not self.training:
            return
        incoming = scores.detach().float().transpose(0, 1)
        merged = (
            incoming if self._score_frontier is None
            else torch.cat((self._score_frontier, incoming), dim=1)
        )
        keep = min(self._quantile_target, int(merged.shape[1]))
        self._score_frontier = merged.topk(keep, dim=1).values

    @torch.no_grad()
    def commit_load(self, counts: Tensor) -> None:
        """Commit one successful full batch; never called by forward/evaluation."""
        if counts.shape != self.token_exposure.shape or counts.dtype != torch.long:
            raise ValueError("load must contain one int64 count per expert")
        self.token_exposure.add_(counts)
        self.expert_updates.add_(counts > 0)
        if self.balancing == "quantile":
            # Exact Quantile Balancing: for each expert take the quantile of its own
            # scores at the target selection rate K/E over the whole step, and set the
            # bias to the mean-centred negative of that threshold. An expert whose scores
            # sit high is pushed down until it is selected at the target rate. The bias is
            # recomputed from absolute scores each step, never accumulated.
            frontier = self._score_frontier
            if frontier is None or self._quantile_target is None:
                raise RuntimeError("quantile balancing committed without observing a step")
            rank = min(self._quantile_target, int(frontier.shape[1]))
            threshold = frontier[:, rank - 1]
            self.selection_bias.copy_(threshold.mean() - threshold)
            self._quantile_target = None
            self._score_frontier = None
            return

    def reset_router_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.projection.weight, mean=0.0, std=self.init_std)
            self.selection_bias.zero_()
            self.token_exposure.zero_()
            self.expert_updates.zero_()

    def forward(self, x: Tensor) -> TopKRouting:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError("router expects [tokens, d_model]")
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = F.linear(x.float(), self.projection.weight.float())
            if self.scoring == "softmax":
                # Softmax control: its z-loss constrains the partition function.
                scores = torch.softmax(logits, dim=-1)
                z_loss = torch.logsumexp(logits, dim=-1).square().mean()
                distribution = scores
            else:
                # Accepted contract: elementwise affinity, no cross-expert normalisation
                # before selection. There is no softmax partition function here, so the
                # classical z-loss does not apply and is returned as an exact zero.
                scores = torch.sqrt(F.softplus(logits))
                z_loss = logits.new_zeros(())
                distribution = scores / scores.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(scores.dtype).tiny
                )
            # The balancing bias steers selection only; weights come from unbiased scores.
            expert_indices = (scores + self.selection_bias).topk(self.top_k, dim=-1).indices
            selected = scores.gather(-1, expert_indices)
            if self.top_k > 1:
                selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(selected.dtype).tiny
                )
            entropy_sum = -(
                distribution * distribution.clamp_min(1e-12).log()
            ).sum(dim=-1).sum()
            expert_counts = torch.bincount(
                expert_indices.reshape(-1), minlength=self.num_experts
            )
            self._observe_scores(scores)
        return TopKRouting(
            logits=logits.detach(),
            expert_indices=expert_indices,
            selected_probabilities=selected,
            z_loss=z_loss,
            entropy_sum=entropy_sum,
            token_count=int(x.shape[0]),
            expert_counts=expert_counts,
        )


__all__ = ["SwitchTopKRouter", "TopKRouting"]
