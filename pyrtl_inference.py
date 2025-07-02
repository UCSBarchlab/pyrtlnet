import argparse
import shutil

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from pyrtlnet.inference_util import display_image, display_outputs, tflite_file_name
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.pyrtl_inference import PyRTLInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST data set.
    _, (test_images, test_labels) = load_mnist_images()

    # Load quantized model.
    interpreter = Interpreter(model_path=tflite_file_name)

    # Create PyRTL inference hardware.
    input_bitwidth = 8
    accumulator_bitwidth = 32
    pyrtl_inference = PyRTLInference(interpreter, input_bitwidth, accumulator_bitwidth)

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        # Print the test image.
        test_image = test_images[test_index]
        print(f"PyRTL network input (#{test_index}):")
        display_image(test_image)

        # Run PyRTL inference on the test image.
        layer0_output, layer1_output, actual = pyrtl_inference.simulate(test_image)

        # Print results.
        print(
            "PyRTL layer0 output (transposed)", layer0_output.shape, layer0_output.dtype
        )
        print(layer0_output.transpose(), "\n")

        print(
            "PyRTL layer1 output (transposed)", layer1_output.shape, layer1_output.dtype
        )
        print(layer1_output.transpose(), "\n")

        print(f"\nPyRTL network output (#{test_index}):")
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
