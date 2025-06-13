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
import tensorflow as tf


def load_tflite_model(model_file_name: str) -> Interpreter:
    """Load the quantized model and return an initialized LiteRT Interpreter."""
    # Set preserve_all_tensors so we can inspect intermediate tensor values.
    # Intermediate values help when debugging other quantized inference implementations.
    interpreter = Interpreter(
        model_path=model_file_name, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    return interpreter


def normalize_input(interpreter: Interpreter, input: tf.Tensor):
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
    return input / input_scale + input_zero_point


def run_tflite_model(
        interpreter: Interpreter, test_image: tf.Tensor) -> (tf.Tensor, tf.Tensor, int):
    """Run inference on a single image with a TFLite model.

    Returns (layer0_output, layer1_output, predicted_digit), where:

    * `layer0_output` is the first layer's raw Tensor output (shape (1, 18)).
    * `layer1_output` is the second layer's raw Tensor output (shape (1, 10)).
    * `predicted_digit` is the actual predicted digit. It is equivalent to
      `layer1_output.flatten().argmax()`.

    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.reset_all_variables()

    # If the input type is quantized, rescale input data.
    if input_details["dtype"] == np.int8:
        test_image = normalize_input(interpreter, test_image)

    # Add the batch dimension and convert from float32 to the actual input type.
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
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
    # tensor 5: layer 0 output int8[1, 18]
    #
    # tensor 6: layer 1 weight int8[10, 18]
    # tensor 7: layer 1 bias   int32[10]
    # tensor 8: layer 1 output int8[1, 10]

    # Retrieve and display the first layer's output. This is not necessary for
    # correctness, but useful for debugging differences between inference
    # implementations.
    layer0_output_index = 5
    layer0_output = interpreter.get_tensor(layer0_output_index)

    # Retrieve and display the second layer's output.
    layer1_output_index = 8
    layer1_output = interpreter.get_tensor(layer1_output_index)

    output = interpreter.get_tensor(output_details["index"])[0]
    output_scale, output_zero_point = output_details["quantization"]

    # Second layer's flattened output should be the model's output.
    assert np.logical_and.reduce(layer1_output.flatten() == output)

    return layer0_output, layer1_output, output.argmax()

def main():
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

    interpreter = load_tflite_model(model_file_name="quantized.tflite")

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        test_image = test_images[test_index]
        print(f"network input (#{test_index}):")
        inference_util.display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        layer0_output, layer1_output, actual = run_tflite_model(
            interpreter=interpreter, test_image=test_image)

        print(f"layer 0 output {layer0_output.shape} {layer0_output.dtype}")
        print(f"{layer0_output}\n")
        print(f"layer 1 output {layer1_output.shape} {layer1_output.dtype}")
        print(f"{layer1_output}\n")

        expected = test_labels[test_index]
        print(f"network output (#{test_index}):")
        inference_util.display_outputs(layer1_output.flatten(), expected, actual)
        if actual == expected:
            correct += 1

        if test_index < args.num_images - 1:
            print()

    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )

if __name__ == "__main__":
    main()
