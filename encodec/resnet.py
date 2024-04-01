from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecResnetBlock

from encodec.array_conversion import jax2pt, pt2jax
from encodec.conv1d import Conv1dParams, convert_conv1d_parms, forward_conv1d

class ResnetParams(NamedTuple):
    conv1d_1: Conv1dParams
    conv1d_3: Conv1dParams
    conv1d_shortcut: Conv1dParams

def convert_resnet_parms(resnet_block: EncodecResnetBlock) -> ResnetParams:
    return ResnetParams(convert_conv1d_parms(resnet_block.block[1]), convert_conv1d_parms(resnet_block.block[3]), convert_conv1d_parms(resnet_block.shortcut))

def forward_resnet(params: ResnetParams, input_: Array) -> Array:
    conv1d_0_param, conv1d_1_param, conv1d_shortcut_param = params
    shortcut_input = forward_conv1d(conv1d_shortcut_param, input_)
    for conv1d in (conv1d_0_param, conv1d_1_param):
        input_ = jax.nn.elu(input_)
        input_ =  forward_conv1d(conv1d, input_)
    return shortcut_input + input_

def test_forward_resnet(model: EncodecModel) -> None:
    # batch_size, input_channels, len_ = 4, 32, 20
    # resnet_block = model.encoder.layers[1]
    batch_size, input_channels, len_ = 4, 128, 2
    resnet_block = model.encoder.layers[7]

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, input_channels, len_))  # batch_size, input_channels, len
    x_pt = jax2pt(x)

    out_pt = resnet_block(x_pt)  # torch.Size([4, 32, 20])

    resnet_param = convert_resnet_parms(resnet_block)
    out = forward_resnet(resnet_param, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
