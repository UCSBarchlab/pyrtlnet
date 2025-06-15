"""Train a quantized two-layer dense neural network on the MNIST dataset.

This code is based on the "Quantization aware training in Keras example" at
https://www.tensorflow.org/model_optimization/guide/quantization/training_example

"""

import pathlib

import mnist_util
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras


def train_unquantized_model(
    learning_rate: float, epochs: int, train_images: tf.Tensor, train_labels: tf.Tensor
) -> keras.Model:
    """Train an unquantized, two-layer, dense MNIST neural network model."""
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

    # Save the unquantized model. Inspecting it in the Model Explorer can help while
    # debugging.
    # model.save("unquantized.keras")

    return model


def evaluate_model(
    model: keras.Model, test_images: tf.Tensor, test_labels: tf.Tensor
) -> (float, float):
    """Evaluate a model on a test data set. Returns (loss, accuracy)."""
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
    """Quantize, evaluate, and save a quantized model."""
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


def main():
    # Load MNIST dataset.
    (train_images, train_labels), (test_images, test_labels) = (
        mnist_util.load_mnist_images()
    )

    learning_rate = 0.001
    epochs = 10

    print("Training unquantized model.")
    model = train_unquantized_model(
        learning_rate=learning_rate,
        epochs=epochs,
        train_images=train_images,
        train_labels=train_labels,
    )
    print("Evaluating unquantized model.")
    loss, accuracy = evaluate_model(
        model=model, test_images=test_images, test_labels=test_labels
    )

    print("Training quantized model and writing ./quantized.tflite.")
    model = quantize_model(
        model=model,
        learning_rate=learning_rate / 10000,
        epochs=int(epochs / 5),
        train_images=train_images,
        train_labels=train_labels,
        model_file_name="quantized.tflite",
    )
    print("Evaluating quantized model.")
    loss, accuracy = evaluate_model(
        model=model, test_images=test_images, test_labels=test_labels
    )


if __name__ == "__main__":
    main()
