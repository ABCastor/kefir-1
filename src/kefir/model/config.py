"""Checkpoint-visible configuration for Kefir v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from kefir.layers.config import ModelConfig

RouterScoring = Literal["softmax", "sqrt_softplus"]
RouterCombine = Literal["selected_softmax_probability", "normalised_topk_softmax"]
DispatchKind = Literal["dropless_grouped_by_expert", "dropless_padded_batched_gemm"]
BalancingKind = Literal["none", "quantile"]


@dataclass(frozen=True, slots=True)
class MoEModelConfig:
    """Complete checkpoint-visible identity for Kefir v1."""

    dense: ModelConfig
    num_experts: int = 64
    top_k: int = 2
    expert_d_ff: int = 352
    expert_type: str = "swiglu"
    moe_layer_indices: tuple[int, ...] = tuple(range(8))
    router_scoring: RouterScoring = "softmax"
    router_combine: RouterCombine = "normalised_topk_softmax"
    router_bias: bool = False
    router_fp32: bool = True
    router_init_std: float = 0.01
    router_jitter: float = 0.0
    router_z_loss_coefficient: float = 1e-4
    load_balancing: BalancingKind = "none"
    balancing_step_size: float = 0.0
    dispatch: DispatchKind = "dropless_padded_batched_gemm"
    capacity_factor: None = None
    dropped_tokens_allowed: bool = False
    shared_expert: bool = False
    version: int = 3

    def __post_init__(self) -> None:
        if self.version != 3:
            raise ValueError("unsupported MoE configuration version; expected 3")
        if self.router_scoring not in {"softmax", "sqrt_softplus"}:
            raise ValueError("unsupported MoE router scoring")
        if self.expert_type != "swiglu":
            raise ValueError("unsupported MoE expert type")
        if self.dispatch not in {"dropless_grouped_by_expert", "dropless_padded_batched_gemm"}:
            raise ValueError("unsupported MoE dispatch")
        if self.num_experts < 2:
            raise ValueError("a routed MoE needs at least two experts")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        expected_combine = (
            "selected_softmax_probability" if self.top_k == 1 else "normalised_topk_softmax"
        )
        if self.router_combine != expected_combine:
            raise ValueError(f"top_k={self.top_k} requires router_combine={expected_combine!r}")
        if self.expert_d_ff <= 0:
            raise ValueError("expert_d_ff must be positive")
        if self.router_scoring == "sqrt_softplus":
            # Elementwise affinities carry no cross-expert normalisation of their own.
            # At k = 1 the mixture weight would renormalise to exactly 1 and the router
            # would receive no gradient from the language-model loss, so the contract
            # that selected these scores also requires k >= 2.
            if self.top_k < 2:
                raise ValueError("sqrt_softplus affinities require top_k >= 2")
            if self.router_z_loss_coefficient != 0.0:
                raise ValueError(
                    "the classical softmax z-loss does not apply to sqrt_softplus "
                    "affinities; the accepted contract sets its coefficient to zero"
                )
        if self.moe_layer_indices != tuple(range(self.dense.n_layers)):
            raise ValueError("Kefir replaces the FFN in every decoder layer")
        if self.router_bias:
            raise ValueError("Kefir router is bias-free")
        if not self.router_fp32:
            raise ValueError("Kefir router must compute logits/probabilities in FP32")
        if self.router_jitter != 0.0:
            raise ValueError("Kefir forbids routing jitter")
        if self.load_balancing not in {"none", "quantile"}:
            raise ValueError("unsupported load-balancing controller")
        if self.balancing_step_size != 0.0:
            raise ValueError("Kefir v1 does not use a balancing step size")
        if self.capacity_factor is not None or self.dropped_tokens_allowed:
            raise ValueError("Kefir is strictly dropless")
        if self.shared_expert:
            raise ValueError("Kefir has no shared expert")
        if not math.isfinite(self.router_init_std) or self.router_init_std <= 0:
            raise ValueError("router_init_std must be positive")
        if not math.isfinite(self.router_z_loss_coefficient) or self.router_z_loss_coefficient < 0:
            raise ValueError("router_z_loss_coefficient must be finite and non-negative")

    @classmethod
    def accepted(cls, **overrides: object) -> "MoEModelConfig":
        """Accepted production geometry: 64 experts, Top-2, no shared expert.

        Eight decoder layers in the frozen (gdn, gdn, gdn, mha) pattern give six GDN-2
        blocks and two gated-MHA blocks; width 256 with four 64-wide heads; SwiGLU experts
        of width 352. The frozen tokenizer exposes exactly 8,000 semantic IDs while the
        tied embedding tensor is physically padded to 8,192 rows for alignment. The extra
        rows are storage only and must never become corpus IDs or output classes.
        """

        dense_fields = {
            "semantic_vocab_size": 8_000, "padded_vocab_size": 8_192, "max_seq_len": 2_048,
            "d_model": 256, "n_layers": 8, "d_ff": 704, "n_heads": 4, "head_dim": 64,
            "gdn_num_key_heads": 4, "gdn_num_value_heads": 4,
            "gdn_key_dim": 64, "gdn_value_dim": 64,
            "gdn_conv_kernel_size": 4, "gdn_chunk_size": 32,
        }
        dense_fields.update(
            {k: v for k, v in overrides.items() if k in ModelConfig.__dataclass_fields__}
        )
        moe_fields: dict[str, object] = {
            "num_experts": 64, "top_k": 2, "expert_d_ff": 352, "expert_type": "swiglu",
            "router_combine": "normalised_topk_softmax",
            "dispatch": "dropless_padded_batched_gemm",
            "moe_layer_indices": tuple(range(int(dense_fields["n_layers"]))),
            # Accepted router contract: sqrt-softplus affinities, no classical softmax
            # z-loss, router init 0.02, and Quantile Balancing for selection only.
            "router_scoring": "sqrt_softplus",
            "router_init_std": 0.02,
            "router_z_loss_coefficient": 0.0,
            "load_balancing": "quantile",
            "version": 3,
        }
        moe_fields.update(
            {k: v for k, v in overrides.items() if k not in ModelConfig.__dataclass_fields__}
        )
        return cls(dense=ModelConfig(**dense_fields), **moe_fields)  # type: ignore[arg-type]

    @property
    def semantic_vocab_size(self) -> int:
        return self.dense.semantic_vocab_size

    @property
    def padded_vocab_size(self) -> int:
        return self.dense.padded_vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.dense.max_seq_len

    @property
    def d_model(self) -> int:
        return self.dense.d_model

    @property
    def n_layers(self) -> int:
        return self.dense.n_layers

    @property
    def d_ff(self) -> int:
        return self.dense.d_ff

    @property
    def architecture(self) -> str:
        return self.dense.architecture

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
