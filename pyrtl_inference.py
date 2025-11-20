import argparse
import shutil
import sys

import numpy as np

from pyrtlnet.cli_util import Accuracy, display_image, display_outputs
from pyrtlnet.inference_util import (
    add_common_arguments,
    batched_images,
    load_mnist_data,
)
from pyrtlnet.pyrtl_inference import PyRTLInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    add_common_arguments(parser)
    parser.add_argument("--verilog", action="store_true", default=False)
    parser.add_argument("--axi", action="store_true", default=False)
    parser.add_argument("--initial_delay_cycles", type=int, default=0)
    args = parser.parse_args()

    # Validate arguments.
    assert args.batch_size == 1

    if args.verilog and args.num_images != 1:
        sys.exit("--verilog can only be used with one image (--num_images=1)")

    if args.num_images == 1:
        args.verbose = True

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST test data.
    test_images, test_labels = load_mnist_data(args.tensor_path)

    # Create PyRTL inference hardware.
    input_bitwidth = 8
    accumulator_bitwidth = 32
    pyrtl_inference = PyRTLInference(
        tensor_path=args.tensor_path,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
        axi=args.axi,
        initial_delay_cycles=args.initial_delay_cycles,
    )

    accuracy = Accuracy()
    for batch_number, (batch_start_index, test_batch) in enumerate(
        batched_images(test_images, args.start_image, args.num_images, args.batch_size)
    ):
        # Display the test image.
        test_image = test_batch[0]
        display_image(
            image=test_image,
            script_name="PyRTL Inference",
            image_index=batch_start_index,
            batch_number=batch_number,
            batch_index=0,
            verbose=args.verbose,
        )

        # Run PyRTL inference on the test image.
        layer0_outputs, layer1_outputs, actual = pyrtl_inference.simulate(
            test_batch, args.verilog
        )

        layer0_outputs = layer0_outputs.transpose()
        layer1_outputs = layer1_outputs.transpose()

        # Display results.
        expected = test_labels[batch_start_index]
        display_outputs(
            script_name="PyRTL Inference",
            layer0_output=layer0_outputs[0],
            layer1_output=layer1_outputs[0],
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
