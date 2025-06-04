"""Train a quantized two-layer dense neural network on the MNIST dataset.

This code is based on the "Quantization aware training in Keras example" at
https://www.tensorflow.org/model_optimization/guide/quantization/training_example

"""

import pathlib

import inference_util
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras

def training():
    # Load MNIST dataset.
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize images to [0, 1] and resize from 28×28 to 12×12 to reduce hardware
    # simulation time.
    train_images = inference_util.preprocess_images(train_images / 255.0)
    test_images = inference_util.preprocess_images(test_images / 255.0)

    # Define the model architecture. This model is unoptimized, so higher accuracy
    # should be possible.
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=train_images[0].shape),
            # Flattened input has 12×12 = 144 elements.
            keras.layers.Flatten(),
            # First layer has 18 outputs.
            keras.layers.Dense(18),
            keras.layers.ReLU(),
            # Second layer has 10 outputs. Each output is the probability that the input
            # depicts the corresponding digit.
            keras.layers.Dense(10),
        ]
    )

    # Train, evaluate, and save the unquantized model.
    # ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    learning_rate = 0.001
    epochs = 10
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("training unquantized model")
    model.fit(train_images, train_labels, epochs=epochs)

    print("evaluating unquantized model")
    model.evaluate(test_images, test_labels)

    model.save("unquantized.keras")

    # Quantize, evaluate, and save the quantized model.
    # ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    quantized_model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate / 10000),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    quantized_model.summary()

    print("training quantized model")
    quantized_model.fit(train_images, train_labels, epochs=2)
    print("evaluating quantized model")
    quantized_model.evaluate(test_images, test_labels)


    def representative_dataset():
        for data in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]


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
    tflite_models_dir = pathlib.Path(".")
    tflite_model_quant_file = tflite_models_dir / "quantized.tflite"
    tflite_model_quant_file.write_bytes(quantized_tflite_model)

if __name__ == "__main__":
    training()
