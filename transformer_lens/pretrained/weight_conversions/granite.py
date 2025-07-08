# In transformer_lens/pretrained/weight_conversions/granite.py

from typing import cast
import einops
import torch
from transformers import GraniteForCausalLM
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_granite_weights(
    hf_model: GraniteForCausalLM, cfg: HookedTransformerConfig
) -> dict[str, torch.Tensor]:
    """
    Converts the weights of a Hugging Face GraniteForCausalLM model to the format
    used by HookedTransformer, correctly handling Grouped-Query Attention (GQA)
    and weight transpositions.
    """
    state_dict = {}

    # Token Embeddings
    state_dict["embed.W_E"] = hf_model.model.embed_tokens.weight

    # Safely get the number of key-value heads for GQA
    # This is the number of heads for the Key and Value projections
    n_kv_heads = cast(int, cfg.n_key_value_heads)

    for l in range(cfg.n_layers):
        # LayerNorm 1 (before attention)
        state_dict[f"blocks.{l}.ln1.w"] = hf_model.model.layers[l].input_layernorm.weight

        # Attention weights
        W_Q = hf_model.model.layers[l].self_attn.q_proj.weight
        W_K = hf_model.model.layers[l].self_attn.k_proj.weight
        W_V = hf_model.model.layers[l].self_attn.v_proj.weight
        W_O = hf_model.model.layers[l].self_attn.o_proj.weight

        # Reshape weights for TransformerLens internal format.
        
        # W_Q uses the main number of heads (n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            W_Q, "(n h) m -> n m h", n=cfg.n_heads, h=cfg.d_head
        )
        
        # W_K and W_V use the smaller number of heads for GQA (n_kv_heads)
        # This is the line that fixes the bug.
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            W_K, "(n h) m -> n m h", n=n_kv_heads, h=cfg.d_head
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            W_V, "(n h) m -> n m h", n=n_kv_heads, h=cfg.d_head
        )
        
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            W_O, "m (n h) -> n h m", n=cfg.n_heads, h=cfg.d_head
        )

        # LayerNorm 2 (before MLP)
        state_dict[f"blocks.{l}.ln2.w"] = hf_model.model.layers[l].post_attention_layernorm.weight

        # MLP weights (transpose is necessary)
        state_dict[f"blocks.{l}.mlp.W_gate"] = hf_model.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_in"] = hf_model.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = hf_model.model.layers[l].mlp.down_proj.weight.T

    # Final LayerNorm
    state_dict["ln_final.w"] = hf_model.model.norm.weight
    
    # Unembedding weights (transpose is necessary)
    state_dict["unembed.W_U"] = hf_model.lm_head.weight.T

    return state_dict