"""Implement quantized inference with numpy and fxpmath.

This does not invoke the LiteRT reference implementation, though it does instantiate an
Interpreter to extract weights, biases, and quantization metadata.

This implements the equations in https://arxiv.org/pdf/1712.05877.pdf . All Equation
references in code comments refer to equations in this paper.

The first layer is quantized per-axis, which is not described in the paper above. See
https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

"""

import argparse
import shutil

from ai_edge_litert.interpreter import Interpreter
import inference_util
import mnist_util
import numpy as np
from fxpmath import Fxp


def normalization_constants(
        s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> (Fxp, np.ndarray):
    """Normalize multiplier `m` to a fixed-point multiplier `m0` and a bit-shift `n`.

    See Section 2.2 in the paper. The multiplier `m` (Equation 5) is computed from:

    `s1`, the scale factors for the matrix multiplication's left input, which is the
        layer's weight matrix.
    `s2`, the scale factors for the matrix multiplication's right input, which is the
        layer's input matrix.
    `s3`, the scale factors for the matrix multiplication's output, which is the layer's
        output matrix.

    This multiplier `m` can then be expressed as a pair of `(m0, n)`, where `m0` is a
    fixed-point 32-bit multiplier. `m == (2 ** -n) * m0`, where `m0` must be in the
    interval [0.5, 1). So a floating-point multiplication by `m` is equivalent to a
    fixed-point multiplication by `m0`, followed by a bitwise right-shift by `n`. This
    fixed-point multiplication and bitwise shift are done by `normalize()`.

    A layer can have per-axis scale factors, so `s1`, `s2`, and `s3` are vectors of
    scale factors. This function returns a vector of fixed-point `m0` values and a
    vector of integer `n` values. See
    https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

    """
    # Equation 5.
    m = s1 * s2 / s3

    # Find the smallest value of `n` such that `m0 == m * (2^n)`, where
    # `m0 >= 0.5` and `m0 < 1`.
    m0 = []
    n = []
    for m_in in m:
        for n_out in range(0, 32):
            # Equation 6.
            m0_out = m_in * (2 ** n_out)
            if m0_out >= 0.5 and m0_out < 1:
                m0.append(m0_out)
                n.append(n_out)
                break
    m0 = np.array(m0)
    n = np.array(n)
    assert len(m0) == len(m)
    assert len(n) == len(m)

    m0 = Fxp(m0, signed=False, n_word=32, n_frac=32)
    return m0, n


def relu(x: np.ndarray):
    return np.maximum(0, x)


def quantized_matmul(q1: np.ndarray, z1: int, q2: np.ndarray, z2: int) -> np.ndarray:
    """Quantized matrix multiplication of `q1` and `q2`.

    `z1` is the zero point for `q1`, and `z2` is the zero point for `q2`.

    This function returns the *un-normalized* matrix multiplication output, which is
    int32. See Sections 2.3 and 2.4 in the paper. The layer's int32 bias can be added to
    this function's output, and the activation function applied. The output must be
    normalized back to int8 with `normalize()` before proceeding to the next layer.

    """
    # Equation 7 (the part in parentheses) and Equation 8. The part of equation 7 that's
    # outside the parentheses (addition of z3 and multiplication by m) are done by
    # normalize(), after adding the bias.
    #
    # All the math in this function is ordinary integer arithmetic. All fixed-point
    # calculations are done in normalize().

    # Accumulations are done with 32-bit integers, see Section 2.4 in the paper.
    q1 = q1.astype(np.int32)
    q2 = q2.astype(np.int32)
    inner_dim = q1.shape[1]

    output = np.zeros((q1.shape[0], q2.shape[1]), dtype=np.int32)
    assert q1.shape[1] == q2.shape[0]
    z1 = np.broadcast_to(z1, [inner_dim])
    z2 = np.broadcast_to(z2, [inner_dim])
    # `z1` is always zero, which can simplify the math below.
    assert (z1 == 0).all()
    for i in range(q1.shape[0]):
        for k in range(q2.shape[1]):
            # Matrix multiplication with per-axis zero points. This is the equation in
            # the "Symmetric vs asymmetric" section at:
            # https://ai.google.dev/edge/litert/models/quantization_spec#symmetric_vs_asymmetric
            #
            # This calculation can be simplified, but we leave it as-is so it's easier
            # to see how it maps to the equation in the LiteRT quantization spec.
            output[i][k] = (
                sum(q1[i][j] * q2[j][k] for j in range(inner_dim))
                - sum(q1[i][j] * z2[j] for j in range(inner_dim))
                - sum(q2[j][k] * z1[j] for j in range(inner_dim))
                + sum(z1[j] * z2[j] for j in range(inner_dim))
            )
    return output


def normalize(
        product: np.ndarray, m0: Fxp, n: np.ndarray, z3: np.ndarray) -> np.ndarray:
    """Convert a 32-bit layer output to a normalized 8-bit output.

    This function effectively multiplies the layer's output by its scale factor `m` and
    adds its zero point `z3`.

    `m` is a floating-point number, which can also be represented by a 32-bit
    fixed-point multiplier `m0` and bitwise right shift `n`, see
    `normalization_constants()`. So instead of doing a floating-point multiplication, we
    do a fixed-point multiplication, followed by a bitwise right shift. This
    multiplication and shift reduces 32-bit `product` values into 8-bit outputs,
    utilizing the 8-bit output range as effectively as possible.

    Layers can have per-axis scale factors, so `m0` and `n` will be vectors of scale
    factors and shift amounts. See
    https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

    """
    # Implement Equation 7, the part outside the parentheses. This function adds `z3`
    # and multiplies by `m`, using fixed-point arithmetic. `m` is decomposed into `(m0,
    # n)` by `normalization_constants()`, using Equation 6.
    product = Fxp(product, signed=True, n_word=32, n_frac=0)
    if m0.size == product.size:
        # Per-axis quantization, so there is one value of `m0` for each dimension in the
        # product. Make the shapes match so we can elementwise-multiply `m0` and
        # `product`.
        m0 = np.reshape(m0, product.shape)
    else:
        # Per-tensor quantization, so there is just one shared value of `m0` for the
        # whole tensor. Broadcast to make copies of `m0` so we can elementwise-multiply
        # `m0` and `product`.
        m0 = np.broadcast_to(m0, product.shape)
    # Multiply by `m0`. The `*` on the next line performs elementwise 32-bit fixed-point
    # multiplication.
    multiplied = m0 * product

    # Fxp only supports shifting by a scalar integer. `n` is a tensor of shift amounts,
    # so we implement a bitwise right shift by `n` as division by the appropriate power
    # of two.
    shift_powers = 2 ** n
    if shift_powers.size == multiplied.size:
        shift_powers = np.reshape(shift_powers, multiplied.shape)
    else:
        shift_powers = np.broadcast_to(shift_powers, multiplied.shape)
    shifted = multiplied / shift_powers

    # Add `z3` and convert to int8.
    added = z3 + shifted

    # Rounding right shift to drop all fractional bits, then convert to 8-bit signed.
    # Fractions are rounded to the nearest integer:
    #   100.4 -> 100
    #   100.5 -> 101
    #   -10.4 -> -10
    #   -10.5 -> -11
    #
    # round_up is the value of the most significant fractional bit (0.5). round_up
    # indicates if the fractional part is greater than or equal to 0.5 for positive
    # numbers. The value is two's complement encoded, so if the value is negative, this
    # bit will be inverted and indicate if the fractional part is less than 0.5.
    #
    # See https://github.com/tensorflow/tensorflow/issues/25087#issuecomment-634262762
    # for more details.
    round_up = (added.val >> (added.n_frac - 1)) & 1
    # overflow="wrap" makes values larger than 127 or smaller than -128 wrap around (128
    # -> -128).
    added_int8 = Fxp((added.val >> added.n_frac) + round_up,
                     signed=True, n_word=8, n_frac=0, overflow="wrap").astype(np.int8)
    return added_int8


def get_tensor_scale_zero(interpreter: Interpreter, tensor_index: int):
    """Retrieve a tensor's scales and zero points from the LiteRT interpreter.

    These scales and zero points may be per-axis or per-tensor.

    """
    tensors = interpreter.get_tensor_details()
    quantization_parameters = tensors[tensor_index]["quantization_parameters"]
    return quantization_parameters["scales"], quantization_parameters["zero_points"]


class QuantizedLayer:
    """Retrieve and store a layer's quantization metadata.

    This retrieves quantization metadata from the LiteRT Interpreter, and performs some
    additional computations. For example, the layer's floating-point scale factor `m` is
    converted to a fixed-point scale factor `m0` and a bitwise right-shift `n`.

    QuantizedLayer also holds the layer's quantized weights and biases.

    """
    def __init__(
        self, interpreter, input_scale, weight_index, bias_index, output_index
    ):
        tensors = interpreter.get_tensor_details()

        weight_scale, weight_zero = get_tensor_scale_zero(
            interpreter=interpreter, tensor_index=weight_index)
        assert (weight_zero == 0).all()
        self.scale, self.zero = get_tensor_scale_zero(
            interpreter=interpreter, tensor_index=output_index)
        # Equation 6.
        self.m0, self.n = normalization_constants(weight_scale, input_scale, self.scale)
        self.weight = interpreter.get_tensor(weight_index)
        self.bias = np.expand_dims(interpreter.get_tensor(bias_index), axis=1)


class NumpyInference:
    """Run quantized inference on an input image."""
    def __init__(self, interpreter: Interpreter):
        """Collect weights, biases, and quantization metadata from a LiteRT Interpreter.

        """
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
        self.input_scale, self.input_zero = get_tensor_scale_zero(
            interpreter=interpreter, tensor_index=2)

        layer0 = QuantizedLayer(
            interpreter=interpreter,
            input_scale=self.input_scale,
            weight_index=3,
            bias_index=4,
            output_index=5,
        )
        layer1 = QuantizedLayer(
            interpreter=interpreter,
            input_scale=layer0.scale,
            weight_index=6,
            bias_index=7,
            output_index=8,
        )
        self.layer = [layer0, layer1]


    def run(self, test_image: np.ndarray) -> (np.ndarray, np.ndarray, int):
        """Run quantized inference on a single image.

        All calculations are done with numpy and fxpmath.

        Returns (layer0_output, layer1_output, predicted_digit), where:

        * `layer0_output` is the first layer's raw Tensor output (shape (1, 18)).
        * `layer1_output` is the second layer's raw Tensor output (shape (1, 10)).
        * `predicted_digit` is the actual predicted digit. It is equivalent to
          `layer1_output.flatten().argmax()`.

        """
        # Flatten the image and add the batch dimension.
        flat_shape = (test_image.shape[0] * test_image.shape[1], 1)

        # The MNIST image data contains pixel values in the range [0, 255]. The neural
        # network was trained by first converting these values to floating point, in the
        # range [0, 1.0]. Dividing by input_scale below undoes this conversion,
        # converting the range from [0, 1.0] back to [0, 255].
        #
        # We could avoid these back-and-forth conversions by modifying
        # `load_mnist_images()` to skip the first conversion, and returning `x +
        # input_zero_point` below to skip the second conversion, but we do them anyway
        # to simplify the code and make it more consistent with existing sample code
        # like https://ai.google.dev/edge/litert/models/post_training_integer_quant
        #
        # Adding input_zero_point (-128) effectively converts the uint8 image data to
        # int8, by shifting the range [0, 255] to [-128, 127].
        flat_image = np.reshape(
            test_image / self.input_scale + self.input_zero, newshape=flat_shape
        ).astype(np.int8)

        layer0_output = quantized_matmul(
            self.layer[0].weight, 0, flat_image, self.input_zero)
        layer0_output = relu(layer0_output + self.layer[0].bias)
        layer0_output = normalize(
            layer0_output, self.layer[0].m0, self.layer[0].n, self.layer[0].zero)
        layer0_output = layer0_output.astype(np.int8)

        layer1_output = quantized_matmul(
            self.layer[1].weight, 0, layer0_output, self.layer[0].zero)
        layer1_output = layer1_output + self.layer[1].bias
        layer1_output = normalize(
            layer1_output, self.layer[1].m0, self.layer[1].n, self.layer[1].zero)

        layer1_output = layer1_output.reshape((10,))

        actual = layer1_output.argmax()

        return layer0_output, layer1_output, actual


def main():
    parser = argparse.ArgumentParser(prog="numpy_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load quantized model.
    tflite_file = "quantized.tflite"
    interpreter = Interpreter(model_path=tflite_file)

    # Colllect weights, biases, and quantization metadata.
    numpy_inference = NumpyInference(interpreter)

    # Load MNIST data set.
    _, (test_images, test_labels) = mnist_util.load_mnist_images()

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        # Run inference on test_index.
        test_image = test_images[test_index]

        print(f"network input (#{test_index}):")
        inference_util.display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        layer0_output, layer1_output, actual = numpy_inference.run(test_image)
        print("layer0 output (transposed)", layer0_output.shape, layer0_output.dtype)
        print(layer0_output.T, "\n")

        print("layer1 output (transposed)", layer1_output.shape, layer1_output.dtype)
        print(layer1_output.T, "\n")

        print(f"network output (#{test_index}):")
        expected = test_labels[test_index]
        inference_util.display_outputs(layer1_output, expected=expected, actual=actual)

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
