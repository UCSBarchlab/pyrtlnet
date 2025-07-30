"""
Run quantized `LiteRT`_ inference on test images from the `MNIST`_ dataset.

.. _MNIST: https://en.wikipedia.org/wiki/MNIST_database

This implementation uses the reference LiteRT ``Interpreter`` inference implementation.
It returns each layer's tensor output, which is useful for verifying the correctness of
:ref:`numpy_inference` and :ref:`pyrtl_inference`.

The `litert_inference demo`_ uses :func:`load_tflite_model` and :func:`run_tflite_model`
to implement quantized inference with `LiteRT`_.

.. _litert_inference demo: https://github.com/UCSBarchlab/pyrtlnet/blob/main/litert_inference.py
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def load_tflite_model(quantized_model_prefix: str) -> Interpreter:
    """Load the quantized model and return an initialized LiteRT ``Interpreter``.

    The quantized model should be produced by :func:`quantize_model`.

    :param quantized_model_prefix: Prefix of the ``.tflite`` file created by
            ``tensorflow_training.py``, without the ``.tflite`` suffix.

    :returns: An initialized LiteRT ``Interpreter``.
    """
    # Set preserve_all_tensors so we can inspect intermediate tensor values.
    # Intermediate values help when debugging other quantized inference implementations.
    interpreter = Interpreter(
        model_path=f"{quantized_model_prefix}.tflite",
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    return interpreter


def _normalize_input(interpreter: Interpreter, input: np.ndarray) -> np.ndarray:
    """Normalize input data to ``int8``.

    This effectively shifts the input data from ``[0, 255]`` to ``[-128, 127]``.
    """
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
    interpreter: Interpreter, test_image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run quantized inference on a single image with a TFLite ``Interpreter``.

    :param interpreter: An initialized TFLite ``Interpreter``, produced by
        :func:`load_tflite_model`.
    :param test_image: An image to run through the ``Interpreter``.

    :returns: ``(layer0_output, layer1_output, predicted_digit)``, where
              ``layer0_output`` is the first layer's raw tensor output, with shape ``(1,
              18)``. ``layer1_output`` is the second layer's raw tensor output, with
              shape ``(1, 10)``. ``predicted_digit`` is the actual predicted digit.
              ``predicted_digit`` is equivalent to ``layer1_output.flatten().argmax()``.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.reset_all_variables()

    # If the input type is quantized, rescale input data.
    if input_details["dtype"] == np.int8:
        test_image = _normalize_input(interpreter, test_image)

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
