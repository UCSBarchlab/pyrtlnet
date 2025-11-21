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
        # Run LiteRT inference on the test image
        layer0_outputs, layer1_outputs, actuals = run_tflite_model(
            interpreter=interpreter, test_batch=test_batch
        )
        # Print results.
        expected = test_labels[batch_start_index]
        for batch_index in range(len(test_batch)):
            # Display the test image
            display_image(
                script_name="LiteRT Inference",
                image=test_batch[batch_index],
                image_index=batch_start_index + batch_index,
                batch_number=batch_number,
                batch_index=batch_index,
                verbose=args.verbose,
            )

            # Display results.
            expected = test_labels[batch_start_index + batch_index]
            actual = actuals[batch_index]
            display_outputs(
                script_name="LiteRT Inference",
                layer0_output=layer0_outputs[batch_index],
                layer1_output=layer1_outputs[batch_index],
                expected=expected,
                actual=actual,
                verbose=args.verbose,
                transposed_outputs=True,
            )

        accuracy.update(actual=actual, expected=expected)

        print()

    accuracy.display()

if __name__ == "__main__":
    main()
