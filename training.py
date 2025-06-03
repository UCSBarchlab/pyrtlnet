# This is based on the "Quantization aware training in Keras example"
# https://www.tensorflow.org/model_optimization/guide/quantization/training_example

import pathlib
import sys

import inference_util
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras

# Load MNIST dataset.
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images.
train_images = inference_util.preprocess_images(train_images / 255.0)
test_images = inference_util.preprocess_images(test_images / 255.0)

# Define the model architecture.
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=train_images[0].shape),
        keras.layers.Flatten(),
        # Hidden layer with 18 outputs.
        keras.layers.Dense(18),
        keras.layers.ReLU(),
        # The model has 10 outputs, each corresponding to the probability that the input
        # represents the corresponding digit.
        keras.layers.Dense(10),
    ]
)

# Train the unquantized model.
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

# Quantize the model.
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
