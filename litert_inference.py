import argparse

import inference_util
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# Load MNIST dataset.
mnist = tf.keras.datasets.mnist
_, (test_images, test_labels) = mnist.load_data()

test_images = inference_util.preprocess_images(test_images)

# Load the quantized model
tflite_file = "quantized.tflite"

# Initialize the interpreter
interpreter = Interpreter(model_path=tflite_file, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
tensors = interpreter.get_tensor_details()

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

input_scale, input_zero_point = input_details["quantization"]


def normalize_input(x):
    return x + input_zero_point


# Helper function to run inference on a TFLite model
def run_tflite_model(start_image: int, num_images: int):
    correct = 0
    for test_index in range(start_image, start_image + num_images):
        interpreter.reset_all_variables()

        test_image = test_images[test_index]

        print(f"network input (#{test_index}):")
        inference_util.display_image(test_image)

        # If the input type is quantized, rescale input data.
        if input_details["dtype"] == np.int8:
            test_image = normalize_input(test_image)

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        print("test_image", test_image.shape, test_image.dtype, "\n")
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()

        layer0_output_index = 5
        layer0_output = interpreter.get_tensor(layer0_output_index)
        print(
            f"layer 0 output (tensor {layer0_output_index})",
            layer0_output.shape,
            layer0_output.dtype,
            "\n",
            layer0_output,
            "\n",
        )

        layer1_output_index = 8
        layer1_output = interpreter.get_tensor(layer1_output_index)
        print(
            f"layer 1 output (tensor {layer1_output_index})",
            layer1_output.shape,
            layer1_output.dtype,
            "\n",
            layer1_output,
            "\n",
        )

        output = interpreter.get_tensor(output_details["index"])[0]
        output_scale, output_zero_point = output_details["quantization"]
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


parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
parser.add_argument("--start_image", type=int, default=0)
parser.add_argument("--num_images", type=int, default=1)
args = parser.parse_args()

run_tflite_model(start_image=args.start_image, num_images=args.num_images)
