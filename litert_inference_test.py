import tempfile
import unittest

import litert_inference
import mnist_util
import tensorflow_training


class TestLiteRTInference(unittest.TestCase):
    def setUp(self):
        """Train a quantized TensorFlow model and load it in the LiteRT Interpreter.

        The model is only trained for one epoch to reduce run time.

        """
        (train_images, train_labels), (self.test_images, self.test_labels) = (
            mnist_util.load_mnist_images()
        )

        with tempfile.NamedTemporaryFile(prefix="quantized_tflite_model") as file:
            model_file_name = file.name

            learning_rate = 0.001
            epochs = 1

            model = tensorflow_training.train_unquantized_model(
                learning_rate=learning_rate,
                epochs=epochs,
                train_images=train_images,
                train_labels=train_labels,
            )
            model = tensorflow_training.quantize_model(
                model=model,
                learning_rate=learning_rate / 10000,
                epochs=epochs,
                train_images=train_images,
                train_labels=train_labels,
                model_file_name=model_file_name,
            )
            self.interpreter = litert_inference.load_tflite_model(
                model_file_name=model_file_name
            )

    def test_litert_inference(self):
        """Run the LiteRT Interpreter on several images and check its accuracy."""
        num_images = 10
        correct = 0
        for test_index in range(num_images):
            _, _, actual = litert_inference.run_tflite_model(
                interpreter=self.interpreter, test_image=self.test_images[test_index]
            )
            expected = self.test_labels[test_index]

            if actual == expected:
                correct += 1

        accuracy = correct / num_images
        self.assertTrue(accuracy > 0.75)


if __name__ == "__main__":
    unittest.main()
