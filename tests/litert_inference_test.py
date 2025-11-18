import pathlib
import tempfile
import unittest

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
                interpreter=self.interpreter, test_image=self.test_images[test_index]
            )
            expected = self.test_labels[test_index]

            if actual == expected:
                correct += 1

        accuracy = correct / num_images
        self.assertTrue(accuracy > 0.75)


if __name__ == "__main__":
    unittest.main()
