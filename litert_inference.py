import argparse
import shutil

import numpy as np

from pyrtlnet.cli_util import Accuracy, display_image, display_outputs
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

    # Validate arguments.
    assert args.batch_size == 1

    if args.num_images == 1:
        args.verbose = True

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    test_images, test_labels = load_mnist_data(args.tensor_path)

    interpreter = load_tflite_model(tensor_path=args.tensor_path)

    accuracy = Accuracy()
    for batch_number, (batch_start_index, test_batch) in enumerate(
        batched_images(test_images, args.start_image, args.num_images, args.batch_size)
    ):
        # Display the test image.
        test_image = test_batch[0]
        display_image(
            image=test_image,
            script_name="LiteRT Inference",
            image_index=batch_start_index,
            batch_number=batch_number,
            batch_index=0,
            verbose=args.verbose,
        )

        # Run LiteRT inference on the test image.
        layer0_outputs, layer1_outputs, actual = run_tflite_model(
            interpreter=interpreter, test_image=test_image
        )

        # Print results.
        expected = test_labels[batch_start_index]
        display_outputs(
            script_name="LiteRT Inference",
            layer0_output=layer0_outputs[0],
            layer1_output=layer1_outputs[0],
            expected=expected,
            actual=actual,
            verbose=args.verbose,
            transposed_outputs=False,
        )

        accuracy.update(actual=actual, expected=expected)

        print()

    accuracy.display()


if __name__ == "__main__":
    main()
