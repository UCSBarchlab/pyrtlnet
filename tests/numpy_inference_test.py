import pathlib
import tempfile
import unittest

import numpy as np

from pyrtlnet.constants import quantized_model_prefix
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
        """Prepare a LiteRT Interpreter and NumPyInference for a comparison test.

        The trained model is loaded in the LiteRT Interpreter, and an instance of
        NumPyInference.
        """
        self.interpreter = load_tflite_model(tensor_path=self.temp_dir.name)
        self.numpy_inference = NumPyInference(tensor_path=self.temp_dir.name)

    def test_numpy_inference(self) -> None:
        """Check that LiteRT Interpreter and NumPyInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.
        """
        test_batch = np.array([self.test_images[0]])

        # input_details = self.interpreter.get_input_details()[0]
        # output_details = self.interpreter.get_output_details()[0]
        # self.interpreter.resize_tensor_input(input_details["index"], (1,12,12))
        # self.interpreter.resize_tensor_input(output_details["index"], ((1, 10)))
        # self.interpreter.allocate_tensors()

        litert_layer0_output, litert_layer1_output, litert_actual = run_tflite_model(
            interpreter=self.interpreter, test_batch=test_batch
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

        test_batch = np.array(
            [self.test_images[i] for i in range(start_image, batch_size + start_image)]
        )

        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]
        self.interpreter.resize_tensor_input(
            input_details["index"], (batch_size, 12, 12)
        )
        self.interpreter.resize_tensor_input(
            output_details["index"], ((batch_size), 10)
        )
        self.interpreter.allocate_tensors()

        litert_layer0_batch_output, litert_layer1_batch_output, litert_actual_batch = (
            run_tflite_model(interpreter=self.interpreter, test_batch=test_batch)
        )

        numpy_layer0_batch_output, numpy_layer1_batch_output, numpy_actual_batch = (
            self.numpy_inference.run(test_batch=test_batch)
        )

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
