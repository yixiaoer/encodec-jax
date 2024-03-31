import einops as op
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecLSTM

from encodec.array_conversion import jax2pt, pt2jax

LSTMParams = tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array], int, int]

def convert_lstm_parms(lstm: EncodecLSTM) -> LSTMParams:
    return ((pt2jax(lstm.lstm.weight_hh_l0), pt2jax(lstm.lstm.weight_ih_l0), pt2jax(lstm.lstm.bias_hh_l0), pt2jax(lstm.lstm.bias_ih_l0)), 
    (pt2jax(lstm.lstm.weight_hh_l1), pt2jax(lstm.lstm.weight_ih_l1), pt2jax(lstm.lstm.bias_hh_l1), pt2jax(lstm.lstm.bias_ih_l1)), 
    lstm.lstm.num_layers, 
    lstm.lstm.hidden_size,)

def forward_lstm(params: LSTMParams, inputs: Array) -> Array:
    params_layer0, params_layer1, num_layers, hidden_size = params
    batch_size, len_ = inputs.shape[0], inputs.shape[2]
    out = []
    h_n = [jnp.zeros((batch_size, hidden_size))] * 2
    c_n = [jnp.zeros((batch_size, hidden_size))] * 2
    for idx in range(len_):
        input_ = inputs[:,:,idx]
        for layer_n, param in enumerate((params_layer0, params_layer1)):
            w_hh, w_ih, b_hh, b_ih = param
            h, c = h_n[layer_n], c_n[layer_n]
            w_input = op.einsum(input_, w_ih, 'b i, h_4 i-> b h_4')
            w_h = op.einsum(h, w_hh, 'b h, h_4 h-> b h_4')
            i = jax.nn.sigmoid(w_input[:, :hidden_size] + b_ih[:hidden_size] + w_h[:, :hidden_size] + b_hh[:hidden_size])
            f = jax.nn.sigmoid(w_input[:, hidden_size: hidden_size * 2] + b_ih[hidden_size: hidden_size * 2] + w_h[:, hidden_size: hidden_size * 2] + b_hh[hidden_size: hidden_size * 2])
            g = jnp.tanh(w_input[:, hidden_size * 2: hidden_size * 3] + b_ih[hidden_size * 2: hidden_size * 3] + w_h[:, hidden_size * 2: hidden_size * 3] + b_hh[hidden_size * 2: hidden_size * 3])
            o = jax.nn.sigmoid(w_input[:, hidden_size * 3:] + b_ih[hidden_size * 3:] + w_h[:, hidden_size * 3:] + b_hh[hidden_size * 3:])
            c = f * c + i * g
            h = o * jnp.tanh(c)

            h_n[layer_n], c_n[layer_n] = h, c
            input_ = h
        out.append(h_n[1])
    return jnp.array(out).transpose([1,2,0]) + inputs

def test_forward_lstm(model: EncodecModel) -> None:
    batch_size, input_size, len_ = 10, 512, 20
    lstm = model.encoder.layers[13]

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, input_size, len_))
    x_pt = jax2pt(x)

    out_pt = lstm(x_pt)

    lstm_params = convert_lstm_parms(lstm)
    out = forward_lstm(lstm_params, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
