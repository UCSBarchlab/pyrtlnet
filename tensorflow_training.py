from pyrtlnet.inference_util import tflite_file_name
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.tensorflow_training import (
    evaluate_model,
    quantize_model,
    train_unquantized_model,
)


def main() -> None:
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
    loss, accuracy = evaluate_model(
        model=model, test_images=test_images, test_labels=test_labels
    )

    print(f"Training quantized model and writing {tflite_file_name}.")
    model = quantize_model(
        model=model,
        learning_rate=learning_rate / 10000,
        epochs=int(epochs / 5),
        train_images=train_images,
        train_labels=train_labels,
        model_file_name=tflite_file_name,
    )
    print("Evaluating quantized model.")
    loss, accuracy = evaluate_model(
        model=model, test_images=test_images, test_labels=test_labels
    )


if __name__ == "__main__":
    main()
