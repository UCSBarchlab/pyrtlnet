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

    if args.num_images == 1:
        args.verbose = True

    np.set_printoptions(linewidth=shutil.get_terminal_size((80, 24)).columns)

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

        # Display the test image.
        for test_batch_index in range(test_batch.shape[0]):
            test_image = test_batch[test_batch_index]
            display_image(
                image=test_image,
                script_name="PyRTL Inference",
                image_index=batch_start_index + test_batch_index,
                batch_number=batch_number,
                batch_index=test_batch_index,
                verbose=args.verbose,
            )

            # Display results.
            expected = test_labels[batch_start_index + test_batch_index]
            display_outputs(
                script_name="PyRTL Inference",
                layer0_output=layer0_outputs[test_batch_index],
                layer1_output=layer1_outputs[test_batch_index],
                expected=expected,
                actual=actual[test_batch_index],
                verbose=args.verbose,
            )

            accuracy.update(actual=actual[test_batch_index], expected=expected)

            print()

    accuracy.display()


if __name__ == "__main__":
    main()
