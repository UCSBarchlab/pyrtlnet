import argparse
import shutil

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from pyrtlnet.inference_util import display_image, display_outputs, tflite_file_name
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.numpy_inference import NumPyInference


def main():
    parser = argparse.ArgumentParser(prog="numpy_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load quantized model.
    interpreter = Interpreter(model_path=tflite_file_name)

    # Colllect weights, biases, and quantization metadata.
    numpy_inference = NumPyInference(interpreter)

    # Load MNIST data set.
    _, (test_images, test_labels) = load_mnist_images()

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
