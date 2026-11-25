"""Small explicit geometries shared by CPU MoE contracts."""

from dataclasses import replace

from kefir.layers.config import ModelConfig
from kefir.model.config import MoEModelConfig


def tiny_moe_config(num_experts: int = 6, top_k: int = 2, d_model: int = 16, d_ff: int = 10) -> MoEModelConfig:
    dense = ModelConfig(
        semantic_vocab_size=32, padded_vocab_size=32, max_seq_len=8,
        d_model=d_model, n_layers=4, d_ff=d_ff, n_heads=2, head_dim=d_model // 2,
        gdn_num_key_heads=2, gdn_num_value_heads=2,
        gdn_key_dim=d_model // 2, gdn_value_dim=d_model // 2,
        gdn_conv_kernel_size=4, gdn_chunk_size=4,
    )
    return MoEModelConfig(
        dense=dense, num_experts=num_experts, top_k=top_k, expert_d_ff=d_ff,
        expert_type="swiglu",
        router_combine="normalised_topk_softmax" if top_k > 1 else "selected_softmax_probability",
        dispatch="dropless_padded_batched_gemm",
        moe_layer_indices=tuple(range(4)), version=3,
    )



def tiny_training_config() -> MoEModelConfig:
    return replace(tiny_moe_config(), router_scoring="sqrt_softplus",
                   load_balancing="quantile", router_z_loss_coefficient=0.0)
