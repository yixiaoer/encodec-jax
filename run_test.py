import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
from transformers import EncodecModel

from encodec.conv1d import test_forward_conv1d

def main():
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    test_forward_conv1d(model)


    print('✅ All tests passed!')

if __name__ == '__main__':
    main()
