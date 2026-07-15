from typing import List, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from nn.linear import Linear, LinearKMaskedBias
from transformers import Optional

# RoPE
def rope_ratate_half(x: jax.Array):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def rope_apply(x: jax.Array, sin: jax.Array, cos: jax.Array):
    return (x * cos) + (rope_ratate_half(x) * sin)

class SelfAttention(nnx.Module):
    def __init__(
        self,
        features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = features // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else Linear
        self.qkv = linear_class(features, features * 3, use_bias=qkv_bias, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = Linear(features, features, use_bias=proj_bias, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)
    
    def apply_rope(self, q: jax.Array, k: jax.Array, rope: jax.Array | Tuple[jax.Array, jax.Array]):
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.astype(rope_dtype)
        k = k.astype(rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = jnp.concatenate((q_prefix, q), axis=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k, k[:, :, prefix:, :], sin, cos)
        k = jnp.concatenate((k_prefix, k), axis=-2)
        q = q.astype(q_dtype)
        k = k.astype(k_dtype)
        return q, k
    
    def __call__(self, x: jax.Array, attn_bias=None, rope: jax.Array = None):
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x
    
    def compute_attention(self, qkv: jax.Array, attn_bias: Optional[jax.Array] = None, rope: Optional[jax.Array] = None):
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2] 
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x: jax.Array = nnx.dot_product_attention(q, k, v) / self.scale
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])
    
class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_attn_std: None | float = None,
        init_proj_std: None | float = None,
        *,
        rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.features = features
        self.num_heads = num_heads
        head_dim = features // num_heads
        self.scale = head_dim**-0.5

        init_attn_std, init_proj_std = self.init_weights(init_attn_std, init_proj_std)

        self.qkv = Linear(
            features, 
            features * 3, 
            use_bias=qkv_bias, 
            kernel_initializer=nnx.initializers.normal(stddev=init_attn_std), 
            rngs=rngs
        )
        self.attn_drop = attn_drop
        self.proj = Linear(
            features, 
            features, 
            use_bias=proj_bias, 
            kernel_initializer=nnx.initializers.normal(stddev=init_proj_std), 
            rngs=rngs
        )
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)


    def init_weights(
        self, 
        init_attn_std: float | None = None, 
        init_proj_std: float | None = None,
        factor: float = 1.0,
        *,
        rngs: nnx.Rngs
    ) -> None:
        init_attn_std = init_attn_std or (self.features**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        return init_attn_std, init_proj_std

    def __call__(self, x: jax.Array, is_causal: bool = True, *, rngs: nnx.Rngs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:,:,0,:], qkv[:,:,1,:], qkv[:,:,2,:]
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = nnx.dot_product_attention(
            q, k, v, 
            dropout_rate=self.attn_drop if self.training else 0, 
            is_causal=is_causal, 
            dropout_rng=rngs()
        )
        x = jnp.copy(x.transpose(1, 2)).reshape(B, N, C)
        x = self.proj_drop(self.proj(x), rngs=rngs)
