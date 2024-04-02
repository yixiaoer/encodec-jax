import math
from typing import Any, NamedTuple

from jax import Array, lax
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecConvTranspose1d

from encodec.array_conversion import jax2pt, pt2jax

# TODO: eliminate this
trim_right_ratio = 1.0  # model.config: "trim_right_ratio": 1.0

class ConvTranspose1dParams(NamedTuple):
    weight: Array
    bias: Array
    dilation: tuple[Any]
    stride: tuple[Any]
    padding_len: int

def convert_convtranspose1d_params(convtranspose1d: EncodecConvTranspose1d) -> ConvTranspose1dParams:
    padding_len = convtranspose1d.conv.kernel_size[0] - convtranspose1d.conv.stride[0]
    return ConvTranspose1dParams(pt2jax(convtranspose1d.conv.weight.data), pt2jax(convtranspose1d.conv.bias.data), convtranspose1d.conv.dilation, convtranspose1d.conv.stride, padding_len)

def forward_convtranspose1d(params: ConvTranspose1dParams, input_: Array) -> Array:
    weight, bias, dilation, stride, padding_len = params
    padding_r = math.ceil(padding_len * trim_right_ratio)
    padding_l = padding_len - padding_r

    result = lax.conv_transpose(
        input_.transpose([0, 2, 1]),
        weight.transpose([2, 1, 0]),
        strides=stride,
        rhs_dilation=dilation,
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),  # input_, weight, output
        transpose_kernel=True,
        precision='highest',
    )
    result = (result + bias).transpose([0, 2, 1])

    return result[:, :, padding_l:result.shape[-1] - padding_r] 

def test_forward_convtranspose1d(model: EncodecModel) -> None:
    batch_size, input_channels, len_ = 4, 512, 10
    convtranspose1d_pt = model.decoder.layers[3]

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, input_channels, len_))
    x_pt = jax2pt(x)

    out_pt = convtranspose1d_pt(x_pt)  # unpad in EncodecConvTranspose1d after applying convtranspose1d
    convtranspose1d_param = convert_convtranspose1d_params(convtranspose1d_pt)
    out = forward_convtranspose1d(convtranspose1d_param, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
