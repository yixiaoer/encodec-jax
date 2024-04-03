from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecVectorQuantization

from .array_conversion import jax2pt, pt2jax

class VectorQuantizationParams(NamedTuple):
    embed: Array

def convert_vq_params(vq: EncodecVectorQuantization) -> VectorQuantizationParams:
    return VectorQuantizationParams(pt2jax(vq.codebook.embed).T)

def quantize_vq(params: VectorQuantizationParams, hidden_states: Array) -> Array:
    embed = params.embed
    distances = jnp.sum(hidden_states ** 2, axis=1)[:,None] - 2 * hidden_states @ embed + jnp.sum(embed ** 2, axis=0)[None,:]
    embed_indices = jnp.argmin(distances, axis=-1)
    return embed_indices

def encode_vq(params: VectorQuantizationParams, hidden_states: Array) -> Array:
    batch_size, dim, len_ = hidden_states.shape  # b, d, l
    hidden_states = hidden_states.transpose(0, 2, 1).reshape((-1, dim))
    embed_indices = quantize_vq(params, hidden_states)
    return embed_indices.reshape(batch_size, len_)

def decode_vq(params: VectorQuantizationParams, embed_indices: Array) -> Array:
    embed = params.embed
    quantize = embed.T[embed_indices]
    return quantize.transpose(0, 2, 1)

def test_methods_vq(model: EncodecModel) -> None:
    batch_size, codebook_dim, len_ = 10, 128, 20
    vq = model.quantizer.layers[10]

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, codebook_dim, len_))
    x_pt = jax2pt(x)

    out_encode_pt = vq.encode(x_pt)
    out_decode_pt = vq.decode(out_encode_pt)

    vq_params = convert_vq_params(vq)
    out_encode = encode_vq(vq_params, x)
    out_decode = decode_vq(vq_params, out_encode)

    assert jnp.allclose(out_encode, pt2jax(out_encode_pt), atol=1e-5)
    assert jnp.allclose(out_decode, pt2jax(out_decode_pt), atol=1e-5)
