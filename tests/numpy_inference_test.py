import pathlib
import tempfile
import unittest

import numpy as np

from pyrtlnet.litert_inference import load_tflite_model, run_tflite_model
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.numpy_inference import NumPyInference
from pyrtlnet.tensorflow_training import quantize_model, train_unquantized_model


class TestNumPyInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Train a quantized TensorFlow model for one epoch, to reduce run time."""
        (train_images, train_labels), (cls.test_images, cls.test_labels) = (
            load_mnist_images()
        )

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.quantized_model_prefix = str(pathlib.Path(cls.temp_dir.name) / "quantized")

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
        """Prepare a LiteRT Interpreter and NumPyInference for a comparison test.

        The trained model is loaded in the LiteRT Interpreter, and an instance of
        NumPyInference.
        """
        self.interpreter = load_tflite_model(
            quantized_model_name=str(self.quantized_model_prefix) + ".tflite"
        )
        self.numpy_inference = NumPyInference(
            quantized_model_name=str(self.quantized_model_prefix) + ".npz"
        )

    def test_numpy_inference(self) -> None:
        """Check that LiteRT Interpreter and NumPyInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.
        """
        test_image = self.test_images[0]
        test_batch = [self.test_images[0]]

        litert_layer0_output, litert_layer1_output, litert_actual = run_tflite_model(
            interpreter=self.interpreter, test_image=test_image
        )

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_batch=test_batch)
        )

        # Check the first layer's outputs.
        np.testing.assert_array_equal(
            numpy_layer0_output,
            litert_layer0_output.transpose(),
            strict=True,
        )

        # Check the second layer's outputs.
        np.testing.assert_array_equal(
            numpy_layer1_output,
            litert_layer1_output.transpose(),
            strict=True,
        )
        # Also verify that the actual predicted digits match.
        self.assertEqual(litert_actual, numpy_actual)

    def test_numpy_inference_batch(self) -> None:
        """Check that LiteRT Interpreter and NumPyInference produce the same results.

        This runs 10 images through both inference systems and compares the tensor
        outputs from each layer.
        """
        start_image = 10
        batch_size = 10

        litert_test_images = self.test_images[start_image : batch_size + start_image]
        numpy_test_batch = [
            self.test_images[i] for i in range(start_image, batch_size + start_image)
        ]

        litert_layer0_batch_output = []
        litert_layer1_batch_output = []
        litert_actual_batch = []

        for test_image in litert_test_images:
            litert_layer0_output, litert_layer1_output, litert_actual = (
                run_tflite_model(interpreter=self.interpreter, test_image=test_image)
            )
            litert_layer0_batch_output.append(litert_layer0_output)
            litert_layer1_batch_output.append(litert_layer1_output)
            litert_actual_batch.append(litert_actual)

        numpy_layer0_batch_output, numpy_layer1_batch_output, numpy_actual_batch = (
            self.numpy_inference.run(test_batch=numpy_test_batch)
        )

        litert_layer0_batch_output = np.squeeze(
            np.array(litert_layer0_batch_output), axis=1
        )
        litert_layer1_batch_output = np.squeeze(
            np.array(litert_layer1_batch_output), axis=1
        )
        litert_actual_batch = np.array(litert_actual_batch)

        # Check the first layer's outputs.
        np.testing.assert_allclose(
            numpy_layer0_batch_output,
            litert_layer0_batch_output.transpose(),
            atol=1,
        )

        # Check the second layer's outputs.
        np.testing.assert_allclose(
            numpy_layer1_batch_output,
            litert_layer1_batch_output.transpose(),
            atol=1,
        )
        # Also verify that the actual predicted digits match.
        self.assertTrue(np.array_equal(litert_actual_batch, numpy_actual_batch))


if __name__ == "__main__":
    unittest.main()
