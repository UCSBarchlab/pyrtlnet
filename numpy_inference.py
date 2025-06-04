# Implement quantized inference, following the equations in
# https://arxiv.org/pdf/1712.05877.pdf . All Equation references in code comments refer
# to equations in this paper.
#
# The first layer is quantized per-axis, see
# https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

import argparse

from ai_edge_litert.interpreter import Interpreter
import inference_util
import numpy as np
import tensorflow as tf
from fxpmath import Fxp


def normalization_constants(s1: float, s2: float, s3: float) -> (Fxp, int):
    # Equation 5.
    m = s1 * s2 / s3

    # Equation 6: Express M as a bit-shifted fixed-point multiplier M0.
    # M == (2 ** -n) * M0, where M is in the interval [0.5, 1).
    m0 = []
    n = []
    for m_in in m:
        for n_out in range(0, 32):
            m0_out = m_in * (2 ** n_out)
            if m0_out >= 0.5 and m0_out < 1:
                m0.append(m0_out)
                n.append(n_out)
                break
    m0 = np.array(m0)
    n = np.array(n)
    assert len(m0) == len(m)
    assert len(n) == len(m)
    assert (m0 >= 0.5).all()
    assert (m0 < 1).all()

    m0 = Fxp(m0, signed=False, n_word=32, n_frac=32)
    return m0, n


def relu(x):
    return x * (x > 0)


def quantized_matmul(q1: np.ndarray, z1: int, q2: np.ndarray, z2: int) -> np.ndarray:
    # Equation 7 (the part in parentheses) and Equation 8. The part of equation 7 #
    # that's outside the parentheses (addition of Z3 and multiplication by M) are done
    # by normalize(), after adding the bias.
    #
    # All the math in this function is ordinary integer arithmetic. Fixed-point
    # calculations only occur in normalize().
    q1 = q1.astype(np.int32)
    q2 = q2.astype(np.int32)
    inner_dim = q1.shape[1]

    # Equation 8, left half. This sums each column of q2.
    a2 = np.zeros((q2.shape[1],), dtype=np.int32)
    for k in range(a2.shape[0]):
        a2[k] = sum(q2[j][k] for j in range(inner_dim))

    # Equation 8, right half. This sums each row of q1.
    a1_ = np.zeros((q1.shape[0],), dtype=np.int32)
    for i in range(a1_.shape[0]):
        a1_[i] = sum(q1[i][j] for j in range(inner_dim))

    output = np.zeros((q1.shape[0], q2.shape[1]), dtype=np.int32)
    assert q1.shape[1] == q2.shape[0]
    # z1 is always zero, which can simplify the math below.
    z1 = np.broadcast_to(z1, [inner_dim])
    z2 = np.broadcast_to(z2, [inner_dim])
    assert (z1 == 0).all()
    for i in range(q1.shape[0]):
        for k in range(q2.shape[1]):
            # Matrix multiplication with per-axis zero points, as described at:
            # https://ai.google.dev/edge/litert/models/quantization_spec#symmetric_vs_asymmetric
            output[i][k] = (
                sum(q1[i][j] * q2[j][k] for j in range(inner_dim))
                - sum(q1[i][j] * z2[j] for j in range(inner_dim))
                - sum(q2[j][k] * z1[j] for j in range(inner_dim))
                + sum(z1[j] * z2[j] for j in range(inner_dim))
            )
    return output

def normalize(product: np.ndarray, m0: Fxp, n: int, z3: int) -> np.ndarray:
    # Equation 7, the part outside the parentheses. This function adds Z3 and
    # multiplies by M, using fixed-point arithmetic. M is decomposed into (m0, n) by
    # normalization_constants(), using Equation 6.
    product = Fxp(product, signed=True, n_word=32, n_frac=0)
    if m0.size == product.size:
        m0 = np.reshape(m0, product.shape)
    else:
        m0 = np.broadcast_to(m0, product.shape)
    multiplied = m0 * product
    if n.size != multiplied.size:
        n = np.broadcast_to(n, (multiplied.size,))
    shifted = []
    for i, multiplier in enumerate(multiplied):
        shifted.append((multiplier >> int(n[i])).astype(np.int8))
    shifted = np.array(shifted)
    added = z3 + shifted
    return added


def get_tensor_scale_zero(interpreter, tensor_index):
    tensors = interpreter.get_tensor_details()
    quantization_parameters = tensors[tensor_index]["quantization_parameters"]
    return quantization_parameters["scales"], quantization_parameters["zero_points"]

class QuantizedLayer:
    def __init__(
        self, interpreter, input_scale, weight_index, bias_index, output_index
    ):
        tensors = interpreter.get_tensor_details()

        weight_scale, weight_zero = get_tensor_scale_zero(interpreter=interpreter, tensor_index=weight_index)
        assert (weight_zero == 0).all()
        self.scale, self.zero = get_tensor_scale_zero(interpreter=interpreter, tensor_index=output_index)
        # Equation 6.
        self.m0, self.n = normalization_constants(weight_scale, input_scale, self.scale)
        self.weight = interpreter.get_tensor(weight_index)
        self.bias = np.expand_dims(interpreter.get_tensor(bias_index), axis=1)

def main(start_image: int, num_images: int):
    # Load quantized model.
    tflite_file = "quantized.tflite"
    interpreter = Interpreter(model_path=tflite_file)
    tensors = interpreter.get_tensor_details()

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

    # Read model tensor quantization metadata.
    input_scale, input_zero = get_tensor_scale_zero(interpreter=interpreter, tensor_index=2)

    print("layer0")
    layer0 = QuantizedLayer(
        interpreter=interpreter,
        input_scale=input_scale,
        weight_index=3,
        bias_index=4,
        output_index=5,
    )
    print("layer1")
    layer1 = QuantizedLayer(
        interpreter=interpreter,
        input_scale=layer0.scale,
        weight_index=6,
        bias_index=7,
        output_index=8,
    )
    layer = [layer0, layer1]

    # Load MNIST data set.
    mnist = tf.keras.datasets.mnist
    _, (test_images, test_labels) = mnist.load_data()

    test_images = inference_util.preprocess_images(test_images)

    correct = 0
    for test_index in range(start_image, start_image + num_images):
        # Run inference on test_index.
        test_image = test_images[test_index]
        print(f"network input (#{test_index}):")
        inference_util.display_image(test_image)

        # Dividing by input_scale effectively cancels out the division by 255.0.
        #
        # Adding input_zero (-128) effectively converts the uint8 test_image data to
        # int8, by shifting the range [0, 255] to [-128, 127].
        flat_size = (test_image.shape[0] * test_image.shape[1], 1)
        flat_image = np.reshape(
            (test_image / 255.0 / input_scale) + input_zero, newshape=flat_size
        ).astype(np.int8)
        print("flat_image", flat_image.shape, flat_image.dtype, "\n")
        print("layer0 weight", layer[0].weight.shape, layer[0].weight.dtype)
        print("layer0 bias", layer[0].bias.shape, layer[0].bias.dtype)

        layer0_output = quantized_matmul(layer[0].weight, 0, flat_image, input_zero)
        layer0_output = relu(layer0_output + layer[0].bias)
        layer0_output = normalize(layer0_output, layer[0].m0, layer[0].n, layer[0].zero)
        layer0_output = layer0_output.astype(np.int8)
        # layer0_output should approximately equal the layer 0 output from
        # tflite_inference.py. The tflite interpreter has unusual rounding behavior to
        # match ARM intrinsics, see
        # https://github.com/tensorflow/tensorflow/issues/25087#issuecomment-634262762
        print("layer0 output", layer0_output.shape, layer0_output.dtype)
        print(layer0_output.T, "\n")

        print("layer1 weight", layer[1].weight.shape, layer[1].weight.dtype)
        print("layer1 bias", layer[1].bias.shape, layer[1].bias.dtype)

        layer1_output = quantized_matmul(layer[1].weight, 0, layer0_output, layer[0].zero)
        layer1_output = layer1_output + layer[1].bias
        layer1_output = normalize(
            layer1_output, layer[1].m0, layer[1].n, layer[1].zero)

        print("layer1 output", layer1_output.shape, layer1_output.dtype)
        print(layer1_output.T, "\n")
        layer1_output = layer1_output.reshape((10,))

        actual = layer1_output.argmax()
        print(f"network output (#{test_index}):")
        expected = test_labels[test_index]
        inference_util.display_outputs(layer1_output, expected=expected, actual=actual)

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
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()
    main(start_image=args.start_image, num_images=args.num_images)
