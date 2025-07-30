import argparse
import pathlib
import shutil
import sys

import numpy as np

from pyrtlnet.inference_util import (
    display_image,
    display_outputs,
    quantized_model_prefix,
)
from pyrtlnet.numpy_inference import NumPyInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="numpy_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    mnist_test_data_file = pathlib.Path(".") / "mnist_test_data.npz"
    if not mnist_test_data_file.exists():
        sys.exit("mnist_test_data.npz not found. Run tensorflow_training.py first.")

    # Load MNIST test data.
    mnist_test_data = np.load(str(mnist_test_data_file))
    test_images = mnist_test_data.get("test_images")
    test_labels = mnist_test_data.get("test_labels")

    tensor_file = pathlib.Path(".") / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(
            f"{quantized_model_prefix}.npz not found. Run tensorflow_training.py first."
        )
    # Collect weights, biases, and quantization metadata.
    numpy_inference = NumPyInference(quantized_model_prefix=quantized_model_prefix)

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        # Run inference on test_index.
        test_image = test_images[test_index]

        print(f"NumPy network input (#{test_index}):")
        display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        layer0_output, layer1_output, actual = numpy_inference.run(test_image)
        print(
            "NumPy layer0 output (transposed)", layer0_output.shape, layer0_output.dtype
        )
        print(layer0_output.transpose(), "\n")

        print(
            "NumPy layer1 output (transposed)", layer1_output.shape, layer1_output.dtype
        )
        print(layer1_output.transpose(), "\n")

        print(f"NumPy network output (#{test_index}):")
        expected = test_labels[test_index]
        display_outputs(layer1_output, expected=expected, actual=actual)

        if actual == expected:
            correct += 1

        if test_index < args.num_images - 1:
            print()

    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
