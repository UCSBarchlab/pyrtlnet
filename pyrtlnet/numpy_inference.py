"""
Implement quantized inference with `NumPy`_ and `fxpmath`_.

This does not invoke the :ref:`litert_inference` reference implementation, though it
does instantiate an ``Interpreter`` to extract weights, biases, and quantization
metadata.

This implements the equations in
`Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference`_.
All `Equation` references in documentation and code comments refer to equations in this
paper.

.. _Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference: https://arxiv.org/pdf/1712.05877.pdf

The first layer is quantized per-axis, which is not described in the paper above. See
`per-axis quantization`_ for details.

.. _per-axis quantization: https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

The `numpy_inference demo`_ uses :class:`NumPyInference` to implement quantized
inference with `NumPy`_.

.. _numpy_inference demo: https://github.com/UCSBarchlab/pyrtlnet/blob/main/numpy_inference.py
"""  # noqa: E501

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fxpmath import Fxp


def normalization_constants(
    s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> tuple[Fxp, np.ndarray]:
    """Normalize multiplier ``m`` to fixed-point ``m0`` and bit-shift ``n``.

    See Section 2.2 in
    `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference`_.
    The multiplier ``m`` (`Equation 5`) is computed from three scale factors
    ``s1, s2, s3``.

    This multiplier ``m`` can then be expressed as a pair of ``(m0, n)``, where ``m0``
    is a fixed-point 32-bit multiplier and ``n`` is a bitwise right-shift amount. A
    floating-point multiplication by ``m`` is equivalent to a fixed-point multiplication
    by ``m0``, followed by a bitwise right-shift by ``n``. This fixed-point
    multiplication and bitwise shift are done by :func:`normalize`.

    In other words, ``m == (2 ** -n) * m0``, where ``m0`` must be in the interval
    ``[0.5, 1)``.

    A layer can have per-axis scale factors, so ``s1``, ``s2``, and ``s3`` are vectors
    of scale factors. This function returns a vector of fixed-point ``m0`` values and a
    vector of integer ``n`` values. See `per-axis quantization`_ for details.

    :param s1: Scale factors for the matrix multiplication's left input, which is the
        layer's weight matrix.
    :param s2: Scale factors for the matrix multiplication's right input, which is the
        layer's input matrix.
    :param s3: Scale factors for the matrix multiplication's output, which is the
        layer's output matrix.

    :returns: ``(m0, n)``, where ``m0`` is a fixed-point multiplier in the interval
              ``[0.5, 1)``, ``n`` is a bitwise right-shift amount, and ``m == (2 ** -n)
              * m0``.
    """  # noqa: E501
    # Equation 5.
    m = s1 * s2 / s3

    # Find the smallest value of ``n`` such that ``m0 == m * (2^n)``, where
    # ``m0 >= 0.5`` and ``m0 < 1``.
    m0 = []
    n = []
    for m_in in m:
        for n_out in range(0, 32):
            # Equation 6.
            m0_out = m_in * (2**n_out)
            if m0_out >= 0.5 and m0_out < 1:
                m0.append(m0_out)
                n.append(n_out)
                break
    m0 = np.array(m0)
    n = np.array(n)
    assert len(m0) == len(m)
    assert len(n) == len(m)

    # ``m0`` must be in the interval ``[0.5, 1)`` so it can be unsigned and we only need
    # fractional bits.
    m0 = Fxp(m0, signed=False, n_word=32, n_frac=32)
    return m0, n


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function, which converts negative values to zero.

    :param x: Input to the activation function.
    :returns: Activation function's output, where each element will be non-negative.

    """
    return np.maximum(0, x)


def quantized_matmul(q1: np.ndarray, z1: int, q2: np.ndarray, z2: int) -> np.ndarray:
    """Quantized matrix multiplication of ``q1`` and ``q2``.

    This function returns the *un-normalized* matrix multiplication output, which has
    ``dtype int32``. See Sections 2.3 and 2.4 in
    `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference`_.
    The layer's ``int32`` bias can be added to this function's output, and the
    :func:`relu` activation function applied, if necessary. The output must then be
    normalized back to ``int8`` with :func:`normalize` before proceeding to the next
    layer.

    :param q1: Left input to the matrix multiplication.
    :param z1: Zero point for ``q1``.
    :param q2: Right input to the matrix multiplication.
    :param z2: Zero point for ``q2``.
    :returns: Un-normalized matrix multiplication output, with ``dtype int32``.

    """  # noqa: E501
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
    product: np.ndarray, m0: Fxp, n: np.ndarray, z3: np.ndarray
) -> np.ndarray:
    """Convert a 32-bit layer output to a normalized 8-bit output.

    This function effectively multiplies the layer's output by its scale factor ``m``
    and adds its zero point ``z3``.

    ``m`` is a floating-point number, which can also be represented by a 32-bit
    fixed-point multiplier ``m0`` and bitwise right shift ``n``, see
    :func:`normalization_constants`. So instead of doing a floating-point
    multiplication, we do a fixed-point multiplication, followed by a bitwise right
    shift. This multiplication and shift reduces 32-bit ``product`` values into 8-bit
    outputs, utilizing the 8-bit output range as effectively as possible.

    Layers can have per-axis scale factors, so ``m0`` and ``n`` will be vectors of scale
    factors and shift amounts. See `per-axis quantization`_ for details.

    :param product: Matrix to normalize, with ``dtype int32``.
    :param m0: Vector of per-row fixed-point multipliers.
    :param n: Vector of per-row shift amounts.
    :param z3: Vector of per-row zero-point adjustments.

    :returns: ``z3 + (product * m0) >> n``, where ``*`` is elementwise fixed-point
              multiplication, and ``>>`` is a rounding right shift. The return value has
              the same shape as ``product`` and ``dtype int8``.
    """
    assert product.dtype == np.int32

    # Implement Equation 7, the part outside the parentheses. This function adds `z3`
    # and multiplies by `m`, using fixed-point arithmetic. `m` is decomposed into `(m0,
    # n)` by `normalization_constants()`, using Equation 6.
    #
    # ``m0` and ``n`` may be quantized on axis 0 (see ``quantized_dimension``). All the
    # operations in this function are elementwise, so we can make NumPy broadcasting
    # work for us by transposing the input, performing all operations, then transposing
    # the output.
    product = Fxp(product.transpose(), signed=True, n_word=32, n_frac=0)

    # Multiply by `m0`. The `*` on the next line performs elementwise 32-bit fixed-point
    # multiplication.
    multiplied = m0 * product

    # Fxp only supports shifting by a scalar integer. `n` is a tensor of shift amounts,
    # so we implement a bitwise right shift by `n` as division by the appropriate power
    # of two.
    shift_powers = 2**n
    shifted = multiplied / shift_powers

    # Rounding right shift to drop all fractional bits. Fractions are rounded to the
    # nearest integer:
    #   100.4 -> 100
    #   100.5 -> 101
    #   -10.4 -> -10
    #   -10.5 -> -11
    #
    # `round_up` is the value of the most significant fractional bit (0.5). `round_up`
    # indicates if the fractional part is greater than or equal to 0.5 for positive
    # numbers. The value is two's complement encoded, so if the value is negative, this
    # bit will be inverted and indicate if the fractional part is less than 0.5.
    #
    # See https://github.com/tensorflow/tensorflow/issues/25087#issuecomment-634262762
    # for more details.
    round_up = (shifted.val >> (shifted.n_frac - 1)) & 1
    shifted = (shifted.val >> shifted.n_frac) + round_up

    # Add `z3` and convert to int8. overflow="wrap" makes values larger than 127 or
    # smaller than -128 wrap around (128 -> -128).
    added = z3 + shifted
    return Fxp(
        added.transpose(), signed=True, n_word=8, n_frac=0, overflow="wrap"
    ).astype(np.int8)


def get_tensor_scale_zero(
    interpreter: Interpreter, tensor_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve a tensor's scale and zero point from the LiteRT ``Interpreter``.

    These scales and zero points may be per-axis or per-tensor.

    :param interpreter: LiteRT ``Interpreter`` to retrieve tensor metadata from.
    :param tensor_index: Index of the tensor to retrieve. These indices can be extracted
        from the `Model Explorer`_.
    :returns: ``(scale, zero_point)`` for the tensor. These are one-dimensional tensors
        with length 1 for per-tensor quantization, and length > 1 for per-axis
        quantization.

    .. _Model Explorer: https://github.com/google-ai-edge/model-explorer


    """
    tensors = interpreter.get_tensor_details()
    quantization_parameters = tensors[tensor_index]["quantization_parameters"]
    scales = quantization_parameters["scales"]
    # TODO: Support other quantized dimensions.
    if len(scales) > 1:
        assert quantization_parameters["quantized_dimension"] == 0
    return scales, quantization_parameters["zero_points"]


class QuantizedLayer:
    """Retrieve and store a layer's quantization metadata from a LiteRT ``Interpreter``.

    This class retrieves weights, biases, and quantization metadata from a LiteRT
    ``Interpreter``, and performs some additional pre-processing. For example, the
    layer's floating-point scale factor ``m`` is converted to a fixed-point scale factor
    ``m0`` and a bitwise right-shift ``n`` with :func:`normalization_constants`.

    """

    def __init__(
        self,
        interpreter: Interpreter,
        input_scale: np.ndarray,
        weight_index: int,
        bias_index: int,
        output_index: int,
    ):
        """Retrieve weights, biases, and quantization metadata from an ``Interpreter``.

        Tensor indices can be found by uploading the model to the `Model Explorer`_.

        :param interpreter: LiteRT ``Interpreter`` to retrieve weights, biases, and
            quantization metadata from.
        :param input_scale: Scale factor for the layer's input. The first layer's input
            comes from the model's input tensor, and that tensor's scale can be
            retrieved with :func:`get_tensor_scale_zero`. The input for subsequent
            layers comes from the preceding layer, so subsequent layer inputs use the
            preceding layer's scale factor. Each layer's scale factor can be retrieved
            with ``QuantizedLayer.scale``.
        :param weight_index: Index of this layer's weight tensor in the model. This
            index can be found in the `Model Explorer`_.
        :param bias_index: Index of this layer's bias tensor in the model. This index
            can be found in the `Model Explorer`_.
        :param output_index: Index of this layer's output tensor in the model. This
            index can be found in the `Model Explorer`_.
        """
        # TODO: Find a way to automatically determine these tensor indices, without
        # having to run them through the Model Explorer.
        weight_scale, weight_zero = get_tensor_scale_zero(
            interpreter=interpreter, tensor_index=weight_index
        )
        assert (weight_zero == 0).all()
        self.scale, self.zero = get_tensor_scale_zero(
            interpreter=interpreter, tensor_index=output_index
        )
        # Equation 6.
        self.m0, self.n = normalization_constants(weight_scale, input_scale, self.scale)
        self.weight = interpreter.get_tensor(weight_index)
        self.bias = np.expand_dims(interpreter.get_tensor(bias_index), axis=1)


class NumPyInference:
    """Run quantized inference on an input image with NumPy and fxpmath."""

    def __init__(self, interpreter: Interpreter):
        """Collect weights, biases, and quantization metadata from an ``Interpreter``.

        :param interpreter: LiteRT ``Interpreter`` to retrieve the neural network's
            weights, biases, and quantization metadata from. The ``Interpreter`` can be
            constructed with :func:`.load_tflite_model`.

        """
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
            interpreter=interpreter, tensor_index=2
        )

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

    def _run_layer(
        self,
        layer_num: int,
        layer_input: np.ndarray,
        layer_input_zero: np.ndarray,
        run_relu: bool,
    ) -> np.ndarray:
        layer_output = quantized_matmul(
            self.layer[layer_num].weight, 0, layer_input, layer_input_zero
        )
        layer_output = layer_output + self.layer[layer_num].bias
        if run_relu:
            layer_output = relu(layer_output)
        layer_output = normalize(
            layer_output,
            self.layer[layer_num].m0,
            self.layer[layer_num].n,
            self.layer[layer_num].zero,
        )
        return layer_output.astype(np.int8)

    def run(self, test_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Run quantized inference on a single image.

        All calculations are done with NumPy and fxpmath.

        :param test_image: An image to run through the NumPy inference implementation.

        :returns: ``(layer0_output, layer1_output, predicted_digit)``, where
                  ``layer0_output`` is the first layer's raw tensor output, with shape
                  ``(18, 1)``. ``layer1_output`` is the second layer's raw tensor
                  output, with shape ``(10, 1)``. Note that these layer outputs are
                  transposed compared to :func:`.run_tflite_model`. ``predicted_digit``
                  is the actual predicted digit. ``predicted_digit`` is equivalent to
                  ``layer1_output.flatten().argmax()``.
        """
        # Flatten the image and add the batch dimension.
        batch_size = 1
        flat_shape = (test_image.shape[0] * test_image.shape[1], batch_size)

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

        layer0_output = self._run_layer(0, flat_image, self.input_zero, run_relu=True)
        layer1_output = self._run_layer(
            1, layer0_output, self.layer[0].zero, run_relu=False
        )

        actual = layer1_output.argmax()

        return layer0_output, layer1_output, actual
