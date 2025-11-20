import argparse
import pathlib
import random

import numpy as np
import tensorflow as tf

from pyrtlnet.constants import quantized_model_prefix, test_data_file
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.tensorflow_training import (
    evaluate_model,
    quantize_model,
    train_unquantized_model,
)
from pyrtlnet.training_util import save_mnist_data


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    parser.add_argument("--tensor_path", type=str, default=".")
    args = parser.parse_args()

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

    model_prefix = pathlib.Path(args.tensor_path) / quantized_model_prefix
    print(
        f"Training quantized model and writing {model_prefix}.tflite and "
        f"{model_prefix}.npz."
    )
    model = quantize_model(
        model=model,
        learning_rate=learning_rate / 10000,
        epochs=int(epochs / 5),
        train_images=train_images,
        train_labels=train_labels,
        quantized_model_prefix=model_prefix,
    )
    print("Evaluating quantized model.")
    evaluate_model(model=model, test_images=test_images, test_labels=test_labels)

    print(f"Saving MNIST test data to {test_data_file}")
    save_mnist_data(
        tensor_path=args.tensor_path, test_images=test_images, test_labels=test_labels
    )


if __name__ == "__main__":
    main()
