import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecConv1d, EncodecConvTranspose1d, EncodecDecoder, EncodecLSTM, EncodecResnetBlock
from torch.nn import ELU

from .array_conversion import jax2pt, pt2jax
from .conv1d import Conv1dParams, convert_conv1d_params, forward_conv1d
from .convtranspose1d import ConvTranspose1dParams, convert_convtranspose1d_params, forward_convtranspose1d
from .lstm import LSTMParams, convert_lstm_params, forward_lstm
from .resnet import ResnetParams, convert_resnet_params, forward_resnet

DecoderParams = list[Conv1dParams | ResnetParams | str | LSTMParams | ConvTranspose1dParams]

def convert_decoder_params(decoder: EncodecDecoder) -> DecoderParams:
    params: DecoderParams = []
    for layer in decoder.layers:
        if isinstance(layer, EncodecConv1d):
            params.append(convert_conv1d_params(layer))
        elif isinstance(layer, EncodecConvTranspose1d):
            params.append(convert_convtranspose1d_params(layer))
        elif isinstance(layer, EncodecResnetBlock):
            params.append(convert_resnet_params(layer))
        elif isinstance(layer, EncodecLSTM):
            params.append(convert_lstm_params(layer))
        elif isinstance(layer, ELU):
            params.append('elu')
        else:
            raise NotImplementedError('Not support type for layers in EncodecDecoder')
    return params

def forward_decoder(params: DecoderParams, hidden_states: Array) -> Array:
    for param in params:
        if isinstance(param, Conv1dParams):
            hidden_states = forward_conv1d(param, hidden_states)
        elif isinstance(param, ResnetParams):
            hidden_states = forward_resnet(param, hidden_states)
        elif isinstance(param, ConvTranspose1dParams):
            hidden_states = forward_convtranspose1d(param, hidden_states)
        elif isinstance(param, LSTMParams):
            hidden_states = forward_lstm(param, hidden_states)
        elif param == 'elu':
            hidden_states = jax.nn.elu(hidden_states)
        else:
            raise NotImplementedError('Not support type for layers in DecoderParams')
    return hidden_states

def test_forward_decoder(model: EncodecModel) -> None:
    batch_size, input_channels, len_ = 4, 128, 10
    decoder = model.decoder

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, input_channels, len_))
    x_pt = jax2pt(x)

    out_pt = decoder(x_pt)

    decoder_params = convert_decoder_params(decoder)
    out = forward_decoder(decoder_params, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
