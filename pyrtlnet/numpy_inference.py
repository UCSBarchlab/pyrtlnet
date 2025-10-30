"""
Implement quantized inference with `NumPy`_ and `fxpmath`_.

This does not use the :ref:`litert_inference` reference implementation.

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
from fxpmath import Fxp

from pyrtlnet.inference_util import SavedTensors


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
    :func:`.normalization_constants`. So instead of doing a floating-point
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


class NumPyInference:
    """Run quantized inference on an input batch with NumPy and fxpmath."""

    def __init__(self, quantized_model_name: str) -> None:
        """Collect weights, biases, and quantization metadata from a ``.npz`` file
        created by ``tensorflow_training.py``.

        :param quantized_model_name: Name of the ``.npz`` file created by
            ``tensorflow_training.py``.
        """
        saved_tensors = SavedTensors(quantized_model_name)
        self.input_scale = saved_tensors.input_scale
        self.input_zero = saved_tensors.input_zero
        self.layer = saved_tensors.layer

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

    def preprocess_image(self, test_image: np.ndarray) -> np.ndarray:
        """Preprocess the raw image data in the batch.
        This is required by the quantized neural network.

        This adjusts the batch image data by ``input_scale`` and ``input_zero``.
        Then, it flattens the 2D image(s) into 1D and adds the batch dimension.

        :param test_batch: Batch data to preprocess. This data should have already been
            normalized to [0.0, 1.0] and resized to 12×12, usually by
            :func:`~pyrtlnet.mnist_util.load_mnist_images`.

        :returns: Flattened batch data, adjusted by the quantized neural network's
                  ``input_scale`` and ``input_zero``.
        """
        # Adjust batch image data to range [-128,127]
        test_image = (test_image / self.input_scale + self.input_zero).astype(np.int8)

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

        #Flatten 2D -> 1D
        return test_image.reshape(test_image.shape[0], -1).T

    def run(self, test_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Run quantized inference on a batch.

        All calculations are done with NumPy and fxpmath.

        :param test_batch: A batch to run through the NumPy inference implementation.

        :returns: ``(layer0_outputs, layer1_outputs, actuals)``, where
                  ``layer0_outputs`` is the first layer's raw tensor output,
                    with shape ``(18, batch_size)``.
                    ``layer1_outputs`` is the second layer's raw tensor output,
                    with shape ``(10, batch_size)``.
                    Note that these layer outputs are transposed
                    compared to :func:`.run_tflite_model`.
                    ``actuals`` is an np.ndarray of predicted digits.
        """

        # Convert single image to batchable shape
        if test_image.ndim == 2:
            test_image = np.expand_dims(test_image, axis=0)

        flat_batch = self.preprocess_image(test_image)
        layer0_outputs = self._run_layer(0, flat_batch, self.input_zero, run_relu=True)
        layer1_outputs = self._run_layer(
            1, layer0_outputs, self.layer[0].zero, run_relu=False
        )

        actuals = layer1_outputs.argmax(axis=0)
        return layer0_outputs, layer1_outputs, actuals
