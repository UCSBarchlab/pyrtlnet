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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to process per batch. "
        "If batch_size > num_images, the model will process up to num_images. "
        "If num_images mod batch_size != 0, the last batch will be of size"
        " num_images mod batch_size.",
    )
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

    # Validate arguments.
    if args.batch_size <= 0:
        sys.exit("batch_size must be greater than 0.")
    if args.num_images + args.start_image > len(test_images):
        print(
<<<<<<< HEAD
            f"Test data set contains {len(test_images)} images. Can't start at image "
            f"{args.start_image} and run {args.num_images} images."
        )
=======
            f"Couldn't start at image {args.start_image} \
              and run {args.num_images} tests."
        )
        print(f"Test data set is {len(test_images)} images long.")
>>>>>>> 9325800b4db10bc8b676adf4529c22e66baa3041
        args.num_images = len(test_images) - args.start_image
        print(f"Running {args.num_images} images instead.")

    if args.num_images == 1:
        args.verbose = True

    tensor_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(f"{tensor_file} not found. Run tensorflow_training.py first.")
    # Collect weights, biases, and quantization metadata.
    numpy_inference = NumPyInference(quantized_model_name=tensor_file)

    correct = 0
    for batch_start_index in range(
        args.start_image, args.start_image + args.num_images, args.batch_size
    ):
        # Run inference on batches
        batch_end_index = min(
            batch_start_index + args.batch_size,
            len(test_images),
            args.start_image + args.num_images,
        )

        test_batch = test_images[batch_start_index:batch_end_index]

        layer0_outputs, layer1_outputs, actuals = numpy_inference.run(test_batch)

        layer0_outputs = layer0_outputs.transpose()
        layer1_outputs = layer1_outputs.transpose()

        for batch_index in range(len(test_batch)):
            test_image = test_batch[batch_index]
            expected = test_labels[batch_start_index + batch_index]

            if batch_index > 0:
                print()

            print(
                f"NumPy network input (#{batch_start_index + batch_index}, ",
<<<<<<< HEAD
                f"batch {(batch_start_index - args.start_image) // args.batch_size}, ",
=======
                f"batch {int(np.floor(batch_start_index / args.batch_size))}, ",
>>>>>>> 9325800b4db10bc8b676adf4529c22e66baa3041
                f"batch_index {batch_index})",
            )

            if args.verbose:
                display_image(test_image)
                print("test_image", test_image.shape, test_image.dtype, "\n")

                print(
                    "NumPy layer0 output (transposed)",
                    layer0_outputs[batch_index].shape,
                    layer0_outputs[batch_index].dtype,
                )
                print(layer0_outputs[batch_index], "\n")
                print(
                    "NumPy layer1 output (transposed)",
                    layer1_outputs[batch_index].shape,
                    layer1_outputs[batch_index].dtype,
                )
                print(layer1_outputs[batch_index], "\n")
                print(f"NumPy network output (#{batch_start_index + batch_index}):")
                display_outputs(
                    layer1_outputs[batch_index],
                    expected=expected,
                    actual=actuals[batch_index],
                )
            else:
                print(f"Expected: {expected} | Actual: {actuals[batch_index]}")

            if actuals[batch_index] == expected:
                correct += 1

        print()
    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
