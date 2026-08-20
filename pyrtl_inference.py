import argparse
import shutil
import sys

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
from pyrtlnet.pyrtl_inference import PyRTLInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    add_common_arguments(parser)
    parser.add_argument(
        "--verilog",
        action="store_true",
        default=False,
        help="""If enabled, export the pyrtlnet hardware design to a Verilog file. The
             file will be named `pyrtl_inference.v` or `pyrtl_inference_axi.v`,
             depending on `--axi`. A Verilog testbench will also be written, named
             `pyrtl_inference_test.v` or `pyrtl_inference_axi_test.v`, which repeats the
             `pyrtl_inference.py` simulation in Verilog.""",
    )
    parser.add_argument(
        "--axi",
        action="store_true",
        default=False,
        help="""If enabled, transmit image data via AXI-Stream and receive layer outputs
             via AXI-Lite. This is synthesizable, but more complicated. If disabled,
             `Simulation` initializes memories with image data, and retrieves layer
             outputs by inspecting registers. This is not synthesizable, and is less
             complicated.""",
    )
    parser.add_argument(
        "--simulation",
        type=str,
        default="FastSimulation",
        help="""Name of the PyRTL `Simulation` class to instantiate. Valid values are
             `Simulation`, `FastSimulation`, and `CompiledSimulation`.""",
    )
    parser.add_argument(
        "--initial_delay_cycles",
        type=int,
        default=0,
        help="""A hack which should not be necessary. Currently required for correct
             FPGA synthesis.""",
    )
    args = parser.parse_args()

    # Validate arguments.
    if args.verilog and args.num_images != 1:
        sys.exit("--verilog can only be used with one image (--num_images=1)")

    terminal_width = shutil.get_terminal_size().columns
    if force_verbose(args.num_images, args.batch_size, terminal_width):
        args.verbose = True

    np.set_printoptions(linewidth=terminal_width)

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
        batch_size=args.batch_size,
    )

    accuracy = Accuracy()

    for batch_number, (batch_start_index, test_batch) in enumerate(
        batched_images(test_images, args.start_image, args.num_images, args.batch_size)
    ):
        # Run PyRTL inference on the test image.
        layer0_outputs, layer1_outputs, actual = pyrtl_inference.simulate(
            test_batch, args.verilog, args.verbose, args.simulation
        )

        # The last batch may not be full. Filter out results for any null images added
        # by `batched_images`.
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
            script_name="PyRTL Inference",
            images=test_batch,
            image_indices=image_indices,
            batch_number=batch_number,
            verbose=args.verbose,
        )

        # Print the batch inference results.
        display_outputs(
            script_name="PyRTL Inference",
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
