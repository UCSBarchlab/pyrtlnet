import pathlib
import tempfile
import unittest

import numpy as np
import pyrtl

from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.numpy_inference import NumPyInference
from pyrtlnet.pyrtl_inference import PyRTLInference
from pyrtlnet.tensorflow_training import quantize_model, train_unquantized_model


class TestPyRTLInference(unittest.TestCase):
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
        """Prepare NumPyInference for a comparison test against PyRTLInference."""
        pyrtl.reset_working_block()

        self.numpy_inference = NumPyInference(
            quantized_model_prefix=self.quantized_model_prefix
        )

    def test_pyrtl_inference(self) -> None:
        """Check that NumPyInference and PyRTLInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.
        """
        pyrtl_inference = PyRTLInference(
            quantized_model_prefix=self.quantized_model_prefix,
            input_bitwidth=8,
            accumulator_bitwidth=32,
            axi=False,
        )

        test_image = self.test_images[0]

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_image=test_image)
        )

        pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
            pyrtl_inference.simulate(test_image=test_image)
        )

        # Check the first layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer0_output,
            numpy_layer0_output,
            strict=True,
        )

        # Check the second layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer1_output,
            numpy_layer1_output,
            strict=True,
        )
        # Also verify that the actual predicted digits match.
        self.assertEqual(numpy_actual, pyrtl_actual)

    def test_pyrtl_inference_axi(self) -> None:
        """Check that NumPyInference and PyRTLInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.
        """
        pyrtl_inference = PyRTLInference(
            quantized_model_prefix=self.quantized_model_prefix,
            input_bitwidth=8,
            accumulator_bitwidth=32,
            axi=True,
        )

        test_image = self.test_images[1]

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_image=test_image)
        )

        pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
            pyrtl_inference.simulate(test_image=test_image)
        )

        # Check the first layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer0_output,
            numpy_layer0_output,
            strict=True,
        )

        # Check the second layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer1_output,
            numpy_layer1_output,
            strict=True,
        )
        # Also verify that the actual predicted digits match.
        self.assertEqual(numpy_actual, pyrtl_actual)


if __name__ == "__main__":
    unittest.main()
