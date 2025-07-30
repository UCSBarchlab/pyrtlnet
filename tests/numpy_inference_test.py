import pathlib
import tempfile
import unittest

import numpy as np

from pyrtlnet.litert_inference import load_tflite_model, run_tflite_model
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.numpy_inference import NumPyInference
from pyrtlnet.tensorflow_training import quantize_model, train_unquantized_model


class TestNumPyInference(unittest.TestCase):
    def setUp(self) -> None:
        """Prepare a LiteRT Interpreter and NumPyInference for a comparison test.

        This trains a quantized TensorFlow model for one epoch, to reduce run time.

        The trained model is loaded in the LiteRT Interpreter, and an instance of
        NumPyInference.
        """
        (train_images, train_labels), (self.test_images, self.test_labels) = (
            load_mnist_images()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            quantized_model_prefix = str(pathlib.Path(temp_dir) / "quantized")

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
                quantized_model_prefix=quantized_model_prefix,
            )
            self.interpreter = load_tflite_model(
                quantized_model_prefix=quantized_model_prefix
            )
            self.numpy_inference = NumPyInference(
                quantized_model_prefix=quantized_model_prefix
            )

    def test_numpy_inference(self) -> None:
        """Check that LiteRT Interpreter and NumPyInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.
        """
        test_image = self.test_images[0]

        litert_layer0_output, litert_layer1_output, litert_actual = run_tflite_model(
            interpreter=self.interpreter, test_image=test_image
        )

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_image=test_image)
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


if __name__ == "__main__":
    unittest.main()
