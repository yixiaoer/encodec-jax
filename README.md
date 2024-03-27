# EnCodec JAX

This project is the JAX implementation of [EnCodec](https://github.com/facebookresearch/encodec).

It is supported by Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Roadmap

- [ ] Model architecture
    - [🤔] EnCodec (encodec_24khz)
    - [ ] EnCodec (encodec_48khz)

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
