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
    # assert args.batch_size == 1

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
        batch_size = args.batch_size
    )
    #matrix gets predefined batch size. even when the batch size doesnt fit cleanly in the num-images, still tries to use batch size
    #since it was defined here. so fill in small batch with 0'd images

    accuracy = Accuracy()
    # If batch_size doesn't fit cleanly into num_images (i.e, num_images % batch_size != 0), use the compensation amount of np.zero images to fill out the batch for the hardware
    compensation = args.num_images % args.batch_size
    for batch_number, (batch_start_index, test_batch) in enumerate(
        batched_images(test_images, args.start_image, args.num_images, args.batch_size)
    ):
        # Run PyRTL inference on the test image.
        compensated = False
        if test_batch.shape[0] < args.batch_size:

            """
            different batch sizes and num_images are giving different results, need to investigate
            """

            filler = np.zeros((compensation,test_batch[0].shape[0],test_batch[0].shape[1]))
            test_batch = np.append(test_batch, filler, axis = 0)
            compensated = True

        layer0_outputs, layer1_outputs, actual = pyrtl_inference.simulate(
            test_batch, args.verilog
        )

        # Display results.
        expected = test_labels[batch_start_index]
        display_outputs(
            script_name="PyRTL Inference",
            layer0_output=layer0_outputs[0],
            layer1_output=layer1_outputs[0],
            expected=expected,
            actual=actual,
            verbose=args.verbose,
        )

        # Display the test image.
        for test_batch_index in range(current_batch_len):
            test_image = test_batch[test_batch_index]
            display_image(
                image=test_image,
                script_name="PyRTL Inference",
                image_index=batch_start_index,
                batch_number=batch_number,
                batch_index=test_batch_index,
                verbose=args.verbose,
            )

            # # Run PyRTL inference on the test image.
            # layer0_outputs, layer1_outputs, actual = pyrtl_inference.simulate(
            #     test_batch, args.verilog
            # )

            # layer0_outputs = layer0_outputs.transpose()
            # layer1_outputs = layer1_outputs.transpose()

            # Display results.
            expected = test_labels[batch_start_index + test_batch_index]
            display_outputs(
                script_name="PyRTL Inference",
                layer0_output=layer0_outputs[test_batch_index],
                layer1_output=layer1_outputs[test_batch_index],
                expected=expected,
                actual=actual[test_batch_index],
                verbose=args.verbose,
                transposed_outputs=True,
            )

            accuracy.update(actual=actual[test_batch_index], expected=expected)

            print()

    accuracy.display()


if __name__ == "__main__":
    main()
