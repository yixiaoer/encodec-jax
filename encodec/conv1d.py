from jax import Array, lax
import jax.numpy as jnp
import jax.random as jrand
import torch
from torch.nn import Conv1d as TorchConv1d
from transformers import EncodecModel

from encodec.array_conversion import jax2pt, pt2jax

Conv1dParams = tuple[Array, Array]

def convert_conv1d_parms(conv1d: TorchConv1d) -> Conv1dParams:
    return pt2jax(conv1d.weight.data), pt2jax(conv1d.bias.data)

def forward_conv1d(params: Conv1dParams, input_: Array, stride=(1,), padding='VALID') -> Array:
    # weight_g, weight_v = params
    weight, bias = params
    result = lax.conv_general_dilated(
        input_.transpose([0, 2, 1]),
        weight.transpose([2, 1, 0]),  # kernerl_size, in, out
        window_strides=stride,
        padding=padding,
        dimension_numbers=('NWC', 'WIO', 'NWC'),  # input_, weight, output
        precision='highest',
    )
    return (result + bias).transpose([0, 2, 1])

def test_forward_conv1d(model: EncodecModel) -> None:
    batch_size, input_channels, len_ = 4, 1, 100
    conv1d_pt = model.encoder.layers[0].conv

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, input_channels, len_))
    x_pt = jax2pt(x)

    out_pt = conv1d_pt(x_pt)  # with shape [4, 32, 94], batch_size, out_channels, len_out( can be calculated with equation refer to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html, (100 + 0 - 1 * (7 - 1) - 1) / 1 + 1) = 94 )

    conv1d_param = convert_conv1d_parms(conv1d_pt)
    out = forward_conv1d(conv1d_param, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
