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
from pyrtlnet.pyrtl_inference import PyRTLInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--verilog", action="store_true", default=False)
    args = parser.parse_args()

    if args.verilog and args.num_images != 1:
        sys.exit("--verilog can only be used with one image (--num_images=1)")

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

    # Create PyRTL inference hardware.
    input_bitwidth = 8
    accumulator_bitwidth = 32
    pyrtl_inference = PyRTLInference(
        quantized_model_prefix=quantized_model_prefix,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        # Print the test image.
        test_image = test_images[test_index]
        print(f"PyRTL network input (#{test_index}):")
        display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        # Run PyRTL inference on the test image.
        layer0_output, layer1_output, actual = pyrtl_inference.simulate(
            test_image, args.verilog
        )

        # Print results.
        print(
            "PyRTL layer0 output (transposed)", layer0_output.shape, layer0_output.dtype
        )
        print(layer0_output.transpose(), "\n")

        print(
            "PyRTL layer1 output (transposed)", layer1_output.shape, layer1_output.dtype
        )
        print(layer1_output.transpose(), "\n")

        print(f"PyRTL network output (#{test_index}):")
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
