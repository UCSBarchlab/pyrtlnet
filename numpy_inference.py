import argparse
import pathlib
import shutil
import sys
from datetime import datetime

import numpy as np

from pyrtlnet.inference_util import (
    display_image,
    display_outputs,
    quantized_model_prefix,
)
from pyrtlnet.numpy_inference import NumPyInference


def main() -> None:
    #TODO add logic for making sure numimages is >= batchsize, ensure num images <= training size, etc.
    parser = argparse.ArgumentParser(prog="numpy_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--tensor_path", type=str, default=".")
    parser.add_argument("--batch_size", type = int, default = 1)
    #fast? enable np multiplication instead of element wise in quant multiply
    parser.add_argument("--verbose", action = argparse.BooleanOptionalAction)
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

    #alvin: form a batch from test_images

    tensor_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(f"{tensor_file} not found. Run tensorflow_training.py first.")
    # Collect weights, biases, and quantization metadata.
    numpy_inference = NumPyInference(quantized_model_name=tensor_file)

    correct = 0
    #alvin: for batch in totalTest Size / batch_size
    #alvin: once batching is done, maybe add option for quick summary of stuff
    # for b in range(len(test_images) / batch_size):
    #     test_batch = np.zeros((test_images[0].shape[0] * test_images[0].shape[1], batch_size))
    #     test_batch = 
    #     # test_batch = [test_images[i] for i in range(batch_size)]

    #     layer0_output, layer1_output, actual = numpy_inference.run(test_batch)
    print(datetime.now())
    for batch_index in range(args.start_image, args.start_image + args.num_images, args.batch_size):
        # Run inference on batches
        vanilla_batch = [test_images[i] for i in range(batch_index,batch_index+args.batch_size)]
        test_batch = np.array(vanilla_batch)
        layer0_outputs, layer1_outputs, actuals = numpy_inference.run(test_batch)
        #TODO fix the indexing here
        for test_index in range(len(actuals)):
            test_image = vanilla_batch[test_index]
            expected = test_labels[batch_index+test_index]
            if args.verbose == True:
                print(f"NumPy network input (#{test_index}):")
                display_image(test_image)
                print("test_image", test_image.shape, test_image.dtype, "\n")

                print(
                    "NumPy layer0 output (transposed)", layer0_outputs[test_index].shape, layer0_outputs[test_index].dtype
                )
                print(layer0_outputs.transpose()[test_index], "\n")

                print(
                    "NumPy layer1 output (transposed)", layer1_outputs[test_index].shape, layer1_outputs[test_index].dtype
                )
                print(layer1_outputs.transpose()[test_index], "\n")

                print(f"NumPy network output (#{test_index}):")
                display_outputs(layer1_outputs.transpose()[test_index], expected=expected, actual=actuals[test_index])
                if test_index < args.num_images - 1:
                    print()

            if actuals[test_index] == expected:
                correct += 1

    print(datetime.now())
    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
