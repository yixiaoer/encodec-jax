from typing import Any

from jax import Array, lax
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecConv1d

from encodec.array_conversion import jax2pt, pt2jax

Conv1dParams = tuple[Array, Array, tuple[Any], tuple[Any], str]

def convert_conv1d_parms(conv1d: EncodecConv1d) -> Conv1dParams:
    padding_len = conv1d.conv.kernel_size[0] - conv1d.conv.dilation[0]
    return pt2jax(conv1d.conv.weight.data), pt2jax(conv1d.conv.bias.data), conv1d.conv.dilation, conv1d.conv.stride, conv1d.pad_mode, padding_len

def forward_conv1d(params: Conv1dParams, input_: Array) -> Array:
    weight, bias, dilation, stride, pad_mode, padding_len = params
    input_ = jnp.pad(input_, pad_width=((0, 0), (0, 0), (padding_len, 0)), mode=pad_mode)
    
    result = lax.conv_general_dilated(
        input_.transpose([0, 2, 1]),
        weight.transpose([2, 1, 0]),  # kernerl_size, in, out
        window_strides=stride,
        rhs_dilation=dilation,
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),  # input_, weight, output
        precision='highest',
    )
    return (result + bias).transpose([0, 2, 1])

def test_forward_conv1d(model: EncodecModel) -> None:
    batch_size, input_channels, len_ = 4, 1, 10
    conv1d_pt = model.encoder.layers[0]

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, input_channels, len_))
    x_pt = jax2pt(x)

    out_pt = conv1d_pt(x_pt)  # padding in EncodecConv1d before applying conv1d

    conv1d_param = convert_conv1d_parms(conv1d_pt)
    out = forward_conv1d(conv1d_param, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
