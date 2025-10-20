import random

import numpy as np
import tensorflow as tf

from pyrtlnet.inference_util import quantized_model_prefix
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.tensorflow_training import (
    evaluate_model,
    quantize_model,
    train_unquantized_model,
)


def main() -> None:
    seed = 42

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load MNIST dataset.
    (train_images, train_labels), (test_images, test_labels) = load_mnist_images()

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
    evaluate_model(model=model, test_images=test_images, test_labels=test_labels)

    print(
        f"Training quantized model and writing {quantized_model_prefix}.tflite and "
        f"{quantized_model_prefix}.npz."
    )
    model = quantize_model(
        model=model,
        learning_rate=learning_rate / 10000,
        epochs=int(epochs / 5),
        train_images=train_images,
        train_labels=train_labels,
        quantized_model_prefix=quantized_model_prefix,
    )
    print("Evaluating quantized model.")
    evaluate_model(model=model, test_images=test_images, test_labels=test_labels)

    # Save the preprocessed MNIST test data so the inference scripts can use it without
    # importing tensorflow. Importing tensorflow is slow and prints a bunch of debug
    # output.
    print("Writing mnist_test_data.npz.")
    np.savez_compressed(
        file="mnist_test_data.npz", test_images=test_images, test_labels=test_labels
    )


if __name__ == "__main__":
    main()
