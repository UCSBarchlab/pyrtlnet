"""These functions train a quantized two-layer dense `MNIST`_ neural network.

.. _MNIST: https://en.wikipedia.org/wiki/MNIST_database

This implementation is based on the "Quantization aware training in `Keras`_ example" at
https://www.tensorflow.org/model_optimization/guide/quantization/training_example

.. _Keras: https://keras.io/

The `tensorflow_training demo`_ uses :func:`train_unquantized_model` and
:func:`quantize_model` to implement quantized training with `TensorFlow`_.

.. _tensorflow_training demo: https://github.com/UCSBarchlab/pyrtlnet/blob/main/tensorflow_training.py

------------------
Model Architecture
------------------

The model processes 12×12 8-bit images of hand-drawn digits from the MNIST data set. The
image sizes are reduced from the data set's original size of 28×28 by
:func:`.load_mnist_images`.

The model consists of two dense layers::

    One input image, shape: (12, 12)
       │
       │
       ▼
    ┌─────────┐
    │ flatten │
    └─────────┘
       │
       │ Tensor shape: (1, 144)
       ▼
    ┌──────────────────┐
    │ layer0: 18 units │
    └──────────────────┘
       │
       │ Tensor shape: (1, 18)
       ▼
    ┌──────┐
    │ ReLU │
    └──────┘
       │
       │ Tensor shape: (1, 18)
       ▼
    ┌──────────────────┐
    │ layer1: 10 units │
    └──────────────────┘
       │
       │
       ▼
    Output tensor, shape: (1, 10)
"""

import pathlib

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras


def train_unquantized_model(
    learning_rate: float, epochs: int, train_images: tf.Tensor, train_labels: tf.Tensor
) -> keras.Model:
    """Train an unquantized, two-layer, dense MNIST neural network model.

    :param learning_rate: Controls how quickly the neural network adjusts its weights.
    :param epochs: Number of times the ``train_images`` are processed.
    :param train_images: Training image data from :func:`.load_mnist_images`.
    :param train_labels: Training labels from :func:`.load_mnist_images`.
    :returns: A trained Keras ``Model``.

    """
    # Define the model architecture. This model is unoptimized, so higher accuracy
    # can be achieved by changing the architecture or its hyperparameters.
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=train_images[0].shape),
            # Flattened input has 12×12 = 144 elements.
            keras.layers.Flatten(),
            # First layer has 18 outputs.
            keras.layers.Dense(18),
            keras.layers.ReLU(),
            # Second layer has 10 outputs. Each output is the probability that the input
            # depicts the corresponding digit, so output[7] is the probability that the
            # input is a picture of a `7`.
            keras.layers.Dense(10),
        ]
    )

    # Train and evaluate the unquantized model.
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_images, train_labels, epochs=epochs)

    return model


def evaluate_model(
    model: keras.Model, test_images: tf.Tensor, test_labels: tf.Tensor
) -> tuple[float, float]:
    """Evaluate a model on its test data set.

    :param model: A trained Keras ``Model``
    :param test_images: Test image data from :func:`.load_mnist_images`.
    :param test_labels: Test image labels from :func:`.load_mnist_images`.
    :returns: ``(loss, accuracy)``, where ``loss`` is the loss function's output (lower
        is better) and ``accuracy`` is the model's accuracy on the test data set (higher
        is better).

    """
    assert len(model.metrics_names) == 2
    assert model.metrics_names[0] == "loss"
    assert model.metrics_names[1] == "accuracy"
    return model.evaluate(test_images, test_labels)


def quantize_model(
    model: keras.Model,
    learning_rate: float,
    epochs: int,
    train_images: tf.Tensor,
    train_labels: tf.Tensor,
    model_file_name: str,
) -> keras.Model:
    """Quantize and save a ``model``.

    The ``model`` should be trained with :func:`train_unquantized_model`.

    The quantized model will be saved to a file named ``model_file_name``, and can be
    loaded with the LiteRT ``Interpreter``, :func:`.load_tflite_model`,
    :class:`.NumPyInference`, or :class:`.PyRTLInference`.

    :param model: A trained Keras ``Model`` from :func:`train_unquantized_model`.
    :param learning_rate: Controls how quickly the neural network adjusts its weights.
    :param epochs: Number of times the ``train_images`` are processed.
    :param train_images: Training image data from :func:`.load_mnist_images`.
    :param train_labels: Training labels from :func:`.load_mnist_images`.
    :param model_file_name: Name for the quantized model file.
    :returns: A quantized Keras ``Model``.

    """
    quantized_model = tfmot.quantization.keras.quantize_model(model)

    # `quantize_model` requires a recompile.
    quantized_model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    quantized_model.fit(train_images, train_labels, epochs=epochs)

    def representative_dataset():
        for data in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]

    if model_file_name is not None:
        # Fully quantize the model to int8. Our hardware implementation does not support
        # floating point computations.
        converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        quantized_tflite_model = converter.convert()

        # Save the quantized model.
        tflite_model_quant_file = pathlib.Path(".") / model_file_name
        tflite_model_quant_file.write_bytes(quantized_tflite_model)

    return quantized_model
