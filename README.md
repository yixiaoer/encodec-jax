# EnCodec JAX

This project is the JAX implementation of [EnCodec](https://github.com/facebookresearch/encodec), a deep learning based audio codec.

It is supported by Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Roadmap

- [ ] Model architecture
    - [x] EnCodec (encodec_24khz) 🤔
    - [ ] EnCodec (encodec_48khz)

## Usage

```python
from datasets import Audio, load_dataset
import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
import jax.numpy as jnp
from transformers import AutoProcessor, EncodecModel

from encodec.array_conversion import pt2jax
from encodec.encodec_model import convert_encodec_model_params, decode_encodec, encode_encodec, foward_encodec

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[-1]["audio"]["array"]
bandwidth = 6

inputs_pt = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
encoder_outputs_pt = model.encode(inputs_pt["input_values"], inputs_pt["padding_mask"], bandwidth=bandwidth)
out_encode_pt = encoder_outputs_pt.audio_codes
out_decode_pt = model.decode(encoder_outputs_pt.audio_codes, encoder_outputs_pt.audio_scales, inputs_pt["padding_mask"])[0]
out_forward_pt = model(inputs_pt["input_values"], inputs_pt["padding_mask"], bandwidth).audio_values

cpu_device = jax.devices('cpu')[0]
# to make JAX calculation precision the same with PyTorch
with jax.default_device(cpu_device):
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="jax")
    encodec_params = convert_encodec_model_params(model, bandwidth)
    encoder_params, quantizer_params, decoder = encodec_params
    out_encode = encode_encodec(encodec_params, inputs["input_values"])
    out_decode = decode_encodec(encodec_params, out_encode, inputs["padding_mask"])
    out_forward = foward_encodec(encodec_params, inputs["input_values"], inputs["padding_mask"], bandwidth)

assert jnp.allclose(out_encode, pt2jax(out_encode_pt), atol=1e-5)
assert jnp.allclose(out_decode, pt2jax(out_decode_pt), atol=1e-5)
assert jnp.allclose(out_forward, pt2jax(out_forward_pt), atol=1e-5)
print(out_decode)
```

## Install

This project requires Python 3.11, JAX 0.4.25.

Create venv:

```sh
python3.11 -m venv venv
```

install dependencies:

TPU VM:

```sh
. venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
```

## Model Architecture

To learn more details about EnCodec, go to the [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) paper.

The EnCodec model consists of three main components.

First, an **encoder** network is input an audio extract and outputs a latent representation:

```sh
EncodecEncoder(
  (layers): ModuleList(
    (0): EncodecConv1d(
      (conv): Conv1d(1, 32, kernel_size=(7,), stride=(1,))
    )
    (1): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(32, 16, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(16, 32, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      )
    )
    (2): ELU(alpha=1.0)
    (3): EncodecConv1d(
      (conv): Conv1d(32, 64, kernel_size=(4,), stride=(2,))
    )
    (4): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(64, 32, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
      )
    )
    (5): ELU(alpha=1.0)
    (6): EncodecConv1d(
      (conv): Conv1d(64, 128, kernel_size=(8,), stride=(4,))
    )
    (7): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(128, 64, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (8): ELU(alpha=1.0)
    (9): EncodecConv1d(
      (conv): Conv1d(128, 256, kernel_size=(10,), stride=(5,))
    )
    (10): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(256, 128, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(128, 256, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      )
    )
    (11): ELU(alpha=1.0)
    (12): EncodecConv1d(
      (conv): Conv1d(256, 512, kernel_size=(16,), stride=(8,))
    )
    (13): EncodecLSTM(
      (lstm): LSTM(512, 512, num_layers=2)
    )
    (14): ELU(alpha=1.0)
    (15): EncodecConv1d(
      (conv): Conv1d(512, 128, kernel_size=(7,), stride=(1,))
    )
  )
)
```

Next, a **quantization** layer produces a compressed representation, using vector quantization:

```sh
EncodecResidualVectorQuantizer(
  (layers): ModuleList(
    (0-31): 32 x EncodecVectorQuantization(
      (codebook): EncodecEuclideanCodebook()
    )
  )
)
```

Last, a **decoder** network $G$ reconstructs the time-domain signal from compressed latent representation:

```sh
EncodecDecoder(
  (layers): ModuleList(
    (0): EncodecConv1d(
      (conv): Conv1d(128, 512, kernel_size=(7,), stride=(1,))
    )
    (1): EncodecLSTM(
      (lstm): LSTM(512, 512, num_layers=2)
    )
    (2): ELU(alpha=1.0)
    (3): EncodecConvTranspose1d(
      (conv): ConvTranspose1d(512, 256, kernel_size=(16,), stride=(8,))
    )
    (4): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(256, 128, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(128, 256, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      )
    )
    (5): ELU(alpha=1.0)
    (6): EncodecConvTranspose1d(
      (conv): ConvTranspose1d(256, 128, kernel_size=(10,), stride=(5,))
    )
    (7): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(128, 64, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (8): ELU(alpha=1.0)
    (9): EncodecConvTranspose1d(
      (conv): ConvTranspose1d(128, 64, kernel_size=(8,), stride=(4,))
    )
    (10): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(64, 32, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
      )
    )
    (11): ELU(alpha=1.0)
    (12): EncodecConvTranspose1d(
      (conv): ConvTranspose1d(64, 32, kernel_size=(4,), stride=(2,))
    )
    (13): EncodecResnetBlock(
      (block): ModuleList(
        (0): ELU(alpha=1.0)
        (1): EncodecConv1d(
          (conv): Conv1d(32, 16, kernel_size=(3,), stride=(1,))
        )
        (2): ELU(alpha=1.0)
        (3): EncodecConv1d(
          (conv): Conv1d(16, 32, kernel_size=(1,), stride=(1,))
        )
      )
      (shortcut): EncodecConv1d(
        (conv): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      )
    )
    (14): ELU(alpha=1.0)
    (15): EncodecConv1d(
      (conv): Conv1d(32, 1, kernel_size=(7,), stride=(1,))
    )
  )
)
```
