from datasets import Audio, load_dataset
from jax import Array
import jax.numpy as jnp
from transformers import AutoProcessor, EncodecModel

from .array_conversion import pt2jax
from .decoder import DecoderParams, convert_decoder_params, forward_decoder
from .encoder import EncoderParams, convert_encoder_params, forward_encoder
from .vq_residual import ResidualVectorQuantizerParams, convert_rvq_params, decode_rvq, encode_rvq

EncodecModelParams = tuple[EncoderParams, ResidualVectorQuantizerParams, DecoderParams]

def convert_encodec_model_params(model: EncodecModel, bandwidth: float | None = None) -> EncodecModelParams:
    encoder_params = convert_encoder_params(model.encoder)
    quantizer_params = convert_rvq_params(model.quantizer, bandwidth)
    decoder_params = convert_decoder_params(model.decoder)
    return encoder_params, quantizer_params, decoder_params

def encode_encodec(params: EncodecModelParams, input_: Array) -> Array:
    encoder_params, quantizer_params, _ = params
    len_ = stride = chunk_length = input_.shape[-1]
    encoded_chunks = []
    step = chunk_length - stride

    for offset in range(0, len_ - step, stride):
        chunk = input_[:, :, offset : offset + chunk_length]
        embeddings = forward_encoder(encoder_params, chunk)
        encoded_chunk = encode_rvq(quantizer_params, embeddings)
        encoded_chunks.append(encoded_chunk)
    return jnp.array(encoded_chunks).transpose(0, 2, 1, 3)

def decode_encodec(params: EncodecModelParams, codes: Array, padding_mask: Array) -> Array:
    _, quantizer_params, decoder_params = params
    embeddings = decode_rvq(quantizer_params, codes[0].transpose(1, 0, 2))
    audio_values = forward_decoder(decoder_params, embeddings)

    if padding_mask.shape[-1] < audio_values.shape[-1]:
        audio_values = audio_values[..., : padding_mask.shape[-1]]
    return audio_values

def foward_encodec(params: EncodecModelParams, input_: Array, padding_mask: Array, bandwidth: float | None = float) -> Array:
    out_encode = encode_encodec(params, input_)
    out_decode = decode_encodec(params, out_encode, padding_mask)
    return out_decode

def test_methods_encodec(model: EncodecModel) -> None:
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
    audio_sample = librispeech_dummy[-1]["audio"]["array"]
    bandwidth = 6.

    inputs_pt = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
    encoder_outputs_pt = model.encode(inputs_pt["input_values"], inputs_pt["padding_mask"], bandwidth=bandwidth)
    out_encode_pt = encoder_outputs_pt.audio_codes
    out_decode_pt = model.decode(encoder_outputs_pt.audio_codes, encoder_outputs_pt.audio_scales, inputs_pt["padding_mask"])[0]
    out_forward_pt = model(inputs_pt["input_values"], inputs_pt["padding_mask"], bandwidth).audio_values

    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="jax")
    encodec_params = convert_encodec_model_params(model, bandwidth)
    out_encode = encode_encodec(encodec_params, inputs["input_values"])
    out_decode = decode_encodec(encodec_params, out_encode, inputs["padding_mask"])
    out_forward = foward_encodec(encodec_params, inputs["input_values"], inputs["padding_mask"], bandwidth)

    assert jnp.allclose(out_encode, pt2jax(out_encode_pt), atol=1e-5)
    assert jnp.allclose(out_decode, pt2jax(out_decode_pt), atol=1e-5)
    assert jnp.allclose(out_forward, pt2jax(out_forward_pt), atol=1e-5)
