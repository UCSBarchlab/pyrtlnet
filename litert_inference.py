"""Run quantized LiteRT inference on test images from the MNIST dataset.

This implementation uses the reference LiteRT inference implementation. It prints
each layer's tensor output, which is useful for verifying correctness in
numpy_inference.py

This is a simple implementation with batch size 1. Multiple images can be processed by
setting num_images, but the images will run through the interpreter one at a time.

"""

import argparse
import shutil

from ai_edge_litert.interpreter import Interpreter
import inference_util
import mnist_util
import numpy as np


def normalize_input(interpreter, x):
    """Normalize input data to int8."""
    input_details = interpreter.get_input_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    # The MNIST image data contains pixel values in the range [0, 255]. The neural
    # network was trained by first converting these values to floating point, in the
    # range [0, 1.0]. Dividing by input_scale below undoes this conversion, converting
    # the range from [0, 1.0] back to [0, 255].
    #
    # We could avoid these back-and-forth conversions by modifying `load_mnist_images()`
    # to skip the first conversion, and returning `x + input_zero_point` below to skip
    # the second conversion, but we do them anyway to simplify the code and make it more
    # consistent with existing sample code like
    # https://ai.google.dev/edge/litert/models/post_training_integer_quant
    #
    # Adding input_zero_point (-128) effectively converts the uint8 image data to int8,
    # by shifting the range [0, 255] to [-128, 127].
    return x / input_scale + input_zero_point


# Helper function to run inference on a TFLite model
def run_tflite_model(interpreter, start_image: int, num_images: int):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0
    for test_index in range(start_image, start_image + num_images):
        interpreter.reset_all_variables()

        test_image = test_images[test_index]

        print(f"network input (#{test_index}):")
        inference_util.display_image(test_image)

        # If the input type is quantized, rescale input data.
        if input_details["dtype"] == np.int8:
            test_image = normalize_input(interpreter, test_image)

        # Add the batch dimension and convert from float32 to the actual input type.
        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        print("test_image", test_image.shape, test_image.dtype, "\n")
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()

        # Tensor metadata, from the Model Explorer
        # (https://github.com/google-ai-edge/model-explorer):
        #
        # tensor 0: input          int8[1, 12, 12]
        #
        # tensor 1: reshape shape  int32[2]
        # tensor 2: reshape output int8[1, 144]
        #
        # tensor 3: layer 0 weight int8[18, 144]
        # tensor 4: layer 0 bias   int32[18]
        # tensor 5: layer 0 output int8[1, 32]
        #
        # tensor 6: layer 1 weight int8[10, 18]
        # tensor 7: layer 1 bias   int32[10]
        # tensor 8: layer 1 output int8[1, 10]

        # Retrieve and display the first layer's output. This is not necessary for
        # correctness, but useful for debugging differences between inference
        # implementations.
        layer0_output_index = 5
        layer0_output = interpreter.get_tensor(layer0_output_index)
        print(
            f"layer 0 output (tensor index {layer0_output_index})",
            layer0_output.shape,
            layer0_output.dtype,
            "\n",
            layer0_output,
            "\n",
        )

        # Retrieve and display the second layer's output.
        layer1_output_index = 8
        layer1_output = interpreter.get_tensor(layer1_output_index)
        print(
            f"layer 1 output (tensor index {layer1_output_index})",
            layer1_output.shape,
            layer1_output.dtype,
            "\n",
            layer1_output,
            "\n",
        )

        output = interpreter.get_tensor(output_details["index"])[0]
        output_scale, output_zero_point = output_details["quantization"]

        # Second layer's output should be the model's output.
        assert np.logical_and.reduce(layer1_output == output, axis=None)

        actual = output.argmax()
        expected = test_labels[test_index]
        print(f"network output (#{test_index}):")
        inference_util.display_outputs(output, expected, actual)

        if actual == expected:
            correct += 1

        if test_index < num_images - 1:
            print()

    if num_images > 1:
        print(
            f"{correct}/{num_images} correct predictions, "
            f"{100.0 * correct / num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="litert_inference.py")
    parser.add_argument("--start_image", type=int, default=0,
                        help="Starting image index in the MNIST test dataset")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to run inference on")
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST dataset.
    _, (test_images, test_labels) = mnist_util.load_mnist_images()

    # Load the quantized model and initialize the LiteRT interpreter. Set
    # preserve_all_tensors so we can inspect intermediate tensor values. Intermediate
    # values help when debugging alternative quantized inference implementations like
    # numpy_inference.py
    tflite_file = "quantized.tflite"
    interpreter = Interpreter(
        model_path=tflite_file, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    run_tflite_model(
        interpreter, start_image=args.start_image, num_images=args.num_images)
