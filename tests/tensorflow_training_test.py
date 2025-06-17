import unittest

from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.tensorflow_training import (
    evaluate_model,
    quantize_model,
    train_unquantized_model,
)


class TestTensorFlowTraining(unittest.TestCase):
    def setUp(self):
        """Load MNIST training and test data sets."""
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = (
            load_mnist_images()
        )

    def test_unquantized_tensorflow_training(self):
        """Train and evalute an unquantized TensorFlow model.

        This test only runs one epoch of training to reduce run time.

        """
        learning_rate = 0.001
        epochs = 1

        model = train_unquantized_model(
            learning_rate=learning_rate,
            epochs=epochs,
            train_images=self.train_images,
            train_labels=self.train_labels,
        )
        loss, accuracy = evaluate_model(model, self.test_images, self.test_labels)
        # Accuracy should be over 75% after one epoch of training. MNIST has ten
        # possible outputs, so randomly guesses will have an accuracy around 10%
        self.assertTrue(accuracy > 0.75)

    def test_quantized_tensorflow_training(self):
        """Train and evalute a quantized TensorFlow model.

        This test only runs one epoch of training to reduce run time.

        """
        learning_rate = 0.001
        epochs = 1

        model = train_unquantized_model(
            learning_rate=learning_rate,
            epochs=epochs,
            train_images=self.train_images,
            train_labels=self.train_labels,
        )
        model = quantize_model(
            model=model,
            learning_rate=learning_rate / 10000,
            epochs=epochs,
            train_images=self.train_images,
            train_labels=self.train_labels,
            model_file_name=None,
        )
        loss, accuracy = evaluate_model(model, self.test_images, self.test_labels)
        # Accuracy should be over 75% after one epoch of training. MNIST has ten
        # possible outputs, so randomly guesses will have an accuracy around 10%
        self.assertTrue(accuracy > 0.75)


if __name__ == "__main__":
    unittest.main()
