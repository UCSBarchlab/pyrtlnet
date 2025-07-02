import argparse
import shutil

import numpy as np

from pyrtlnet.inference_util import display_image, display_outputs, tflite_file_name
from pyrtlnet.litert_inference import load_tflite_model, run_tflite_model
from pyrtlnet.mnist_util import load_mnist_images


def main() -> None:
    parser = argparse.ArgumentParser(prog="litert_inference.py")
    parser.add_argument(
        "--start_image",
        type=int,
        default=0,
        help="Starting image index in the MNIST test dataset",
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="Number of images to run inference on"
    )
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST dataset.
    _, (test_images, test_labels) = load_mnist_images()

    interpreter = load_tflite_model(model_file_name=tflite_file_name)

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        test_image = test_images[test_index]
        print(f"LiteRT network input (#{test_index}):")
        display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        layer0_output, layer1_output, actual = run_tflite_model(
            interpreter=interpreter, test_image=test_image
        )

        print(f"LiteRT layer 0 output {layer0_output.shape} {layer0_output.dtype}")
        print(f"{layer0_output}\n")
        print(f"LiteRT layer 1 output {layer1_output.shape} {layer1_output.dtype}")
        print(f"{layer1_output}\n")

        expected = test_labels[test_index]
        print(f"LiteRT network output (#{test_index}):")
        display_outputs(layer1_output.flatten(), expected, actual)
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
