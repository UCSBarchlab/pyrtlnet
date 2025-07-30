import numpy as np
from ai_edge_litert.interpreter import Interpreter


def get_tensor_scale_zero(
    interpreter: Interpreter, tensor_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve a tensor's scale and zero point from the LiteRT ``Interpreter``.

    These scales and zero points may be per-axis or per-tensor.

    For more information, see :ref:`numpy_inference` and these references:

    - `Quantization and Training of Neural Networks for Efficient
      Integer-Arithmetic-Only Inference`_

    - `per-axis quantization`_

    .. _Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference: https://arxiv.org/pdf/1712.05877.pdf

    .. _per-axis quantization: https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

    :param interpreter: LiteRT ``Interpreter`` to retrieve tensor metadata from.
    :param tensor_index: Index of the tensor to retrieve. These indices can be extracted
        from the `Model Explorer`_.

    .. _Model Explorer: https://github.com/google-ai-edge/model-explorer

    :returns: ``(scale, zero_point)`` for the tensor. These are one-dimensional tensors
        with length 1 for per-tensor quantization, and length > 1 for per-axis
        quantization.
    """  # noqa: E501
    tensors = interpreter.get_tensor_details()
    quantization_parameters = tensors[tensor_index]["quantization_parameters"]
    scales = quantization_parameters["scales"]
    # TODO: Support other quantized dimensions.
    if len(scales) > 1:
        assert quantization_parameters["quantized_dimension"] == 0
    return scales, quantization_parameters["zero_points"]


def _get_layer_tensors(
    interpreter: Interpreter,
    layer_name: str,
    weight_index: int,
    bias_index: int,
    output_index: int,
    tensors: dict[str, np.ndarray],
) -> None:
    tensors[f"{layer_name}.weight.scale"], tensors[f"{layer_name}.weight.zero"] = (
        get_tensor_scale_zero(interpreter=interpreter, tensor_index=weight_index)
    )
    tensors[f"{layer_name}.output.scale"], tensors[f"{layer_name}.output.zero"] = (
        get_tensor_scale_zero(interpreter=interpreter, tensor_index=output_index)
    )
    tensors[f"{layer_name}.weight"] = interpreter.get_tensor(weight_index)
    tensors[f"{layer_name}.bias"] = np.expand_dims(
        interpreter.get_tensor(bias_index), axis=1
    )


def save_tensors(interpreter: Interpreter, quantized_model_prefix: str) -> None:
    """Saves a quantized model's weights, biases, and quantization metadata.

    The tensors are saved to a NumPy ``.npz`` file with :func:`numpy.savez_compressed`,
    and can be loaded by :class:`.SavedTensors` or :func:`numpy.load`.

    :param interpreter: LiteRT ``Interpreter`` to retrieve tensor metadata from.
    :param quantized_model_prefix: Prefix for the ``.npz`` file to save, without the
        ``.npz`` suffix.
    """
    tensors = {}

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
    tensors["input.scale"], tensors["input.zero"] = get_tensor_scale_zero(
        interpreter=interpreter, tensor_index=2
    )
    _get_layer_tensors(
        interpreter=interpreter,
        layer_name="layer0",
        weight_index=3,
        bias_index=4,
        output_index=5,
        tensors=tensors,
    )
    _get_layer_tensors(
        interpreter=interpreter,
        layer_name="layer1",
        weight_index=6,
        bias_index=7,
        output_index=8,
        tensors=tensors,
    )

    np.savez_compressed(file=f"{quantized_model_prefix}.npz", **tensors)
