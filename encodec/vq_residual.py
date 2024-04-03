import math

from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecResidualVectorQuantizer

from .array_conversion import jax2pt, pt2jax
from .vq import VectorQuantizationParams, convert_vq_params, encode_vq, decode_vq

# TODO: eliminate this
codebook_size = 1024  # model.config.codebook_size
frame_rate = 75  # model.config.frame_rate
num_quantizers_ = 32  # 32, model.config.num_quantizers

ResidualVectorQuantizerParams = list[VectorQuantizationParams]

def get_num_quantizers(bandwidth: float | None = None) -> int:
    if bandwidth is None:
        return num_quantizers_
    num_quantizers = math.floor(bandwidth * 1000 / (math.log2(codebook_size) * frame_rate))
    num_quantizers = max(1, num_quantizers)
    return int(num_quantizers)

def convert_rvq_params(rvq: EncodecResidualVectorQuantizer, bandwidth: float| None = None) -> ResidualVectorQuantizerParams:
    num_quantizers = get_num_quantizers(bandwidth)
    return [convert_vq_params(layer) for layer in rvq.layers[: num_quantizers]]

def encode_rvq(params: ResidualVectorQuantizerParams, input_: Array) -> Array:
    embed_indices = []

    for param in params:
        if not isinstance(param, VectorQuantizationParams):
            raise NotImplementedError('Not support type for layers in ResidualVectorQuantizerParams')

        param_indices = encode_vq(param, input_)
        quantized = decode_vq(param, param_indices)
        input_ -= quantized
        embed_indices.append(param_indices)

    return jnp.array(embed_indices)

def decode_rvq(params: ResidualVectorQuantizerParams, embed_indices: Array) -> Array:
    quantized = jnp.array([0.])
    for i, indices in enumerate(embed_indices):
        quantized += decode_vq(params[i], indices)
    return quantized

def test_methods_rvq(model: EncodecModel) -> None:
    batch_size, codebook_dim, len_ = 10, 128, 20
    rvq = model.quantizer

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, codebook_dim, len_))
    x_pt = jax2pt(x)
    bandwidth = 6.

    out_encode_pt = rvq.encode(x_pt, bandwidth)
    out_decode_pt = rvq.decode(out_encode_pt)

    rvq_params = convert_rvq_params(rvq, bandwidth)
    out_encode = encode_rvq(rvq_params, x)
    out_decode = decode_rvq(rvq_params, out_encode)

    assert jnp.allclose(out_encode, pt2jax(out_encode_pt), atol=1e-5)
    assert jnp.allclose(out_decode, pt2jax(out_decode_pt), atol=1e-5)
