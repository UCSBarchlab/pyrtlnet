import pathlib
import tempfile
import unittest

import numpy as np

from pyrtlnet.constants import quantized_model_prefix
from pyrtlnet.litert_inference import load_tflite_model, run_tflite_model
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.tensorflow_training import quantize_model, train_unquantized_model


class TestLiteRTInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Train a quantized TensorFlow model for one epoch, to reduce run time."""
        (train_images, train_labels), (cls.test_images, cls.test_labels) = (
            load_mnist_images()
        )

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.quantized_model_prefix = str(
            pathlib.Path(cls.temp_dir.name) / quantized_model_prefix
        )

        learning_rate = 0.001
        epochs = 1

        model = train_unquantized_model(
            learning_rate=learning_rate,
            epochs=epochs,
            train_images=train_images,
            train_labels=train_labels,
        )
        model = quantize_model(
            model=model,
            learning_rate=learning_rate / 10000,
            epochs=epochs,
            train_images=train_images,
            train_labels=train_labels,
            quantized_model_prefix=cls.quantized_model_prefix,
        )

    def setUp(self) -> None:
        self.interpreter = load_tflite_model(tensor_path=self.temp_dir.name)

    def test_litert_inference(self) -> None:
        """Run the LiteRT Interpreter on several images and check its accuracy."""
        num_images = 10
        correct = 0

        for test_index in range(num_images):
            _, _, actual = run_tflite_model(
                interpreter=self.interpreter,
                test_batch=np.array([self.test_images[test_index]]),
            )
            expected = self.test_labels[test_index]

            if actual == expected:
                correct += 1

        accuracy = correct / num_images
        self.assertGreater(accuracy, 0.75)

    def test_litert_inference_batch(self) -> None:
        start_image = 10
        batch_size = 10
        correct = 0

        test_batch = self.test_images[start_image : start_image + batch_size]
        (
            _litert_layer0_batch_output,
            _litert_layer1_batch_output,
            litert_actual_batch,
        ) = run_tflite_model(interpreter=self.interpreter, test_batch=test_batch)
        for batch_index in range(batch_size):
            if litert_actual_batch[batch_index] == self.test_labels[10 + batch_index]:
                correct += 1
        accuracy = correct / batch_size
        self.assertGreater(accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
