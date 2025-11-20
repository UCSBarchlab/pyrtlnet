import numpy as np
from fxpmath import Fxp


def normalization_constants(
    s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> tuple[Fxp, np.ndarray]:
    """Normalize multiplier ``m`` to fixed-point ``m0`` and bit-shift ``n``.

    See Section 2.2 in
    `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference`_.
    The multiplier ``m`` (`Equation 5`) is computed from three scale factors
    ``s1, s2, s3``.

    .. _Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference: https://arxiv.org/pdf/1712.05877.pdf

    This multiplier ``m`` can then be expressed as a pair of ``(m0, n)``, where ``m0``
    is a fixed-point 32-bit multiplier and ``n`` is a bitwise right-shift amount. A
    floating-point multiplication by ``m`` is equivalent to a fixed-point multiplication
    by ``m0``, followed by a bitwise right-shift by ``n``. This fixed-point
    multiplication and bitwise shift are done by :func:`~.normalize`.

    In other words, ``m == (2 ** -n) * m0``, where ``m0`` must be in the interval
    ``[0.5, 1)``.

    A layer can have per-axis scale factors, so ``s1``, ``s2``, and ``s3`` are vectors
    of scale factors. This function returns a vector of fixed-point ``m0`` values and a
    vector of integer ``n`` values. See `per-axis quantization`_ for details.

    .. _per-axis quantization: https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

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


class QuantizedLayer:
    """Stores a layer's weights, biases, and quantization metadata.

    This class performs some additional pre-processing on the raw quantization metadata.
    For example, the layer's floating-point scale factor ``m`` is converted to a
    fixed-point scale factor ``m0`` and a bitwise right-shift ``n`` with
    :func:`normalization_constants`.
    """

    scale: np.ndarray
    """The layer output's floating point scale factor ``m``."""

    zero: np.ndarray
    """The layer output's zero point ``z``."""

    m0: np.ndarray
    """The layer's :attr:`scale` can be expressed as a fixed-point scale factor
    :attr:`m0` and a bitwise right-shift :attr:`n`. See :func:`normalization_constants`.
    """

    n: np.ndarray
    """The layer's :attr:`scale` can be expressed as a fixed-point scale factor
    :attr:`m0` and a bitwise right-shift :attr:`n`. See :func:`normalization_constants`.
    """

    weight: np.ndarray
    """The layer's quantized weight."""

    bias: np.ndarray
    """The layer's quantized bias."""

    def __init__(
        self,
        input_scale: np.ndarray,
        weight_scale: np.ndarray,
        weight_zero: np.ndarray,
        output_scale: np.ndarray,
        output_zero: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        """Store a layer's weights, biases, and quantization metadata.

        :param input_scale: Scale factor for the layer's input. The first layer's input
            is special, and must be retrieved from the model separately. The input for
            subsequent layers comes from the preceding layer, so subsequent layer inputs
            use the preceding layer's scale factor. A layer's scale factor can be
            retrieved with :attr:`scale`.
        :param weight_scale: Scale factor for the layer's weight.
        :param weight_zero: Zero point for the layer's weight.
        :param output_scale: Scale factor for the layer's output.
        :param output_zero: Zero point for the layer's output.
        :param weight: The layer's weight.
        :param bias: The layer's bias.
        """
        assert (weight_zero == 0).all()
        self.scale = output_scale
        self.zero = output_zero
        self.m0, self.n = normalization_constants(weight_scale, input_scale, self.scale)
        self.weight = weight
        self.bias = bias


class SavedTensors:
    """
    Loads weights, biases, and quantization metadata saved by :func:`.save_tensors`.
    """

    input_scale: np.ndarray
    """Floating-point scale factor for neural network's input.

    This scale factor can be converted to a fixed-point multiplier by
    :func:`.normalization_constants`.
    """

    input_zero: np.ndarray
    """Zero point for neural network's input."""

    layer: list[QuantizedLayer]
    """List of :class:`QuantizedLayer` containing per-layer weights, biases, and
    quantization metadata.
    """

    def __init__(self, quantized_model_name: str) -> None:
        tensors = np.load(quantized_model_name)

        self.input_scale = tensors.get("input.scale")
        self.input_zero = tensors.get("input.zero")

        self.layer = []
        input_scale = self.input_scale
        for layer in range(2):
            current_layer = QuantizedLayer(
                input_scale=input_scale,
                weight_scale=tensors.get(f"layer{layer}.weight.scale"),
                weight_zero=tensors.get(f"layer{layer}.weight.zero"),
                output_scale=tensors.get(f"layer{layer}.output.scale"),
                output_zero=tensors.get(f"layer{layer}.output.zero"),
                weight=tensors.get(f"layer{layer}.weight"),
                bias=tensors.get(f"layer{layer}.bias"),
            )
            self.layer.append(current_layer)
            input_scale = current_layer.scale
