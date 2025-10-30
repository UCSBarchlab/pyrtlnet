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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tensor_path", type=str, default=".")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    mnist_test_data_file = pathlib.Path(args.tensor_path) / "mnist_test_data.npz"
    if not mnist_test_data_file.exists():
        sys.exit(f"{mnist_test_data_file} not found. Run tensorflow_training.py first.")

    # Load MNIST test data.
    mnist_test_data = np.load(str(mnist_test_data_file))
    test_images = mnist_test_data.get("test_images")
    test_labels = mnist_test_data.get("test_labels")

    # Correct arguments if needed.
    if args.batch_size > args.num_images:
        args.num_images = args.batch_size

    if args.num_images > len(test_images):
        args.num_images = len(test_images)

    if args.num_images == 1:
        args.verbose = True

    tensor_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(f"{tensor_file} not found. Run tensorflow_training.py first.")
    # Collect weights, biases, and quantization metadata.
    numpy_inference = NumPyInference(quantized_model_name=tensor_file)

    correct = 0
    actual_tested_image_count = 0
    for index in range(
        args.start_image, args.start_image + args.num_images, args.batch_size
    ):
        # Run inference on batches

        batch_index = min(index, len(test_images) - 1)
        batch_end_index = min(batch_index + args.batch_size, len(test_images))
        vanilla_batch = test_images[batch_index:batch_end_index]

        test_batch = np.array(vanilla_batch)
        layer0_outputs, layer1_outputs, actuals = numpy_inference.run(test_batch)

        layer0_outputs = layer0_outputs.transpose()
        layer1_outputs = layer1_outputs.transpose()

        for test_index in range(len(vanilla_batch)):
            test_image = vanilla_batch[test_index]
            expected = test_labels[batch_index + test_index]
            if args.verbose:
                print(f"NumPy network input (#{test_index}):")
                display_image(test_image)
                print("test_image", test_image.shape, test_image.dtype, "\n")

                print(
                    "NumPy layer0 output (transposed)",
                    layer0_outputs[test_index].shape,
                    layer0_outputs[test_index].dtype,
                )
                print(layer0_outputs[test_index], "\n")
                print(layer1_outputs.shape)
                print(
                    "NumPy layer1 output (transposed)",
                    layer1_outputs[test_index].shape,
                    layer1_outputs[test_index].dtype,
                )
                print(layer1_outputs[test_index], "\n")
                print(f"NumPy network output (#{test_index}):")
                display_outputs(
                    layer1_outputs[test_index],
                    expected=expected,
                    actual=actuals[test_index],
                )
                if test_index < args.num_images - 1:
                    print()

            if actuals[test_index] == expected:
                correct += 1
            actual_tested_image_count += 1

    if args.num_images > 1:
        print(
            f"{correct}/{actual_tested_image_count} correct predictions, "
            f"{100.0 * correct / actual_tested_image_count:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
