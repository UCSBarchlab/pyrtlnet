import argparse
import shutil

import numpy as np

from pyrtlnet.cli_util import (
    Accuracy,
    display_images,
    display_outputs,
    force_verbose,
    trim_batch,
)
from pyrtlnet.inference_util import (
    add_common_arguments,
    batched_images,
    load_mnist_data,
)
from pyrtlnet.litert_inference import load_tflite_model, run_tflite_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="litert_inference.py")
    add_common_arguments(parser)
    args = parser.parse_args()

    terminal_width = shutil.get_terminal_size().columns
    if force_verbose(args.num_images, args.batch_size, terminal_width):
        args.verbose = True

    np.set_printoptions(linewidth=terminal_width)

    test_images, test_labels = load_mnist_data(args.tensor_path)

    interpreter = load_tflite_model(tensor_path=args.tensor_path)

    accuracy = Accuracy()
    for batch_number, (batch_start_index, test_batch) in enumerate(
        batched_images(test_images, args.start_image, args.num_images, args.batch_size)
    ):
        # Run LiteRT inference on the test batch.
        layer0_outputs, layer1_outputs, actual = run_tflite_model(
            interpreter=interpreter, test_batch=test_batch
        )

        # `batched_images` may have padded the batch with null images. Filter out any
        # results corresponding to these null images.
        image_indices, layer0_outputs, layer1_outputs, actual, expected = trim_batch(
            batch_start_index,
            batch_number,
            args.batch_size,
            args.num_images,
            layer0_outputs,
            layer1_outputs,
            actual,
            test_labels,
        )

        # Display the batch of test images.
        display_images(
            script_name="LiteRT Inference",
            images=test_batch,
            image_indices=image_indices,
            batch_number=batch_number,
            verbose=args.verbose,
        )

        # Print the batch inference results.
        display_outputs(
            script_name="LiteRT Inference",
            layer0_output=layer0_outputs,
            layer1_output=layer1_outputs,
            expected=expected,
            actual=actual,
            verbose=args.verbose,
        )
        accuracy.update(actual=actual, expected=expected)

        print()

    accuracy.display()


if __name__ == "__main__":
    main()
