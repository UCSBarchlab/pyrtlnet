import pathlib
import tempfile
import unittest

import numpy as np
import pyrtl

from pyrtlnet.constants import quantized_model_prefix
from pyrtlnet.inference_util import batched_images
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
        """Prepare NumPyInference for a comparison test against PyRTLInference."""
        pyrtl.reset_working_block()

        self.numpy_inference = NumPyInference(tensor_path=self.temp_dir.name)

    def test_pyrtl_inference(self) -> None:
        """Check that NumPyInference and PyRTLInference produce the same results.

        This runs a batch of images through both inference systems and compares the
        tensor outputs from each layer.
        """
        start_image = 0
        batch_size = 2
        pyrtl_inference = PyRTLInference(
            tensor_path=self.temp_dir.name,
            input_bitwidth=8,
            accumulator_bitwidth=32,
            axi=False,
            batch_size=batch_size,
        )

        test_batch = self.test_images[start_image : start_image + batch_size]

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_batch=test_batch)
        )

        pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
            pyrtl_inference.simulate(test_batch=test_batch)
        )

        # Check the first layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer0_output, numpy_layer0_output, strict=True
        )

        # Check the second layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer1_output, numpy_layer1_output, strict=True
        )

        # Also verify that the actual predicted digits match.
        np.testing.assert_array_equal(pyrtl_actual, numpy_actual, strict=True)

    def test_pyrtl_inference_axi(self) -> None:
        """Check that NumPyInference and PyRTLInference produce the same results.
        PyrtlInference uses AXI-Lite in this test.

        This runs a batch through both inference systems and compares the tensor outputs
        from each layer.
        """
        start_image = 0
        batch_size = 2
        pyrtl_inference = PyRTLInference(
            tensor_path=self.temp_dir.name,
            input_bitwidth=8,
            accumulator_bitwidth=32,
            axi=True,
            batch_size=batch_size,
        )

        test_batch = self.test_images[start_image : start_image + batch_size]

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_batch=test_batch)
        )

        pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
            pyrtl_inference.simulate(test_batch=test_batch)
        )

        # Check the first layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer0_output, numpy_layer0_output, strict=True
        )

        # Check the second layer's outputs.
        np.testing.assert_array_equal(
            pyrtl_layer1_output, numpy_layer1_output, strict=True
        )
        # Also verify that the actual predicted digits match.
        np.testing.assert_array_equal(pyrtl_actual, numpy_actual, strict=True)

    def test_pyrtl_inference_uneven_batch(self) -> None:
        """
        Check that NumPyInference and PyRTLInference produce the same results, using
        uneven batches.

        This runs a two batches of different sizes through both inference systems and
        compares the tensor outputs from each layer.
        """

        start_image = 6
        batch_size = 2
        num_images = 3

        pyrtl_inference = PyRTLInference(
            tensor_path=self.temp_dir.name,
            input_bitwidth=8,
            accumulator_bitwidth=32,
            axi=False,
            batch_size=batch_size,
        )

        for _batch_number, (_batch_start_index, test_batch) in enumerate(
            batched_images(self.test_images, start_image, num_images, batch_size)
        ):
            numpy_layer0_output, numpy_layer1_output, numpy_actual = (
                self.numpy_inference.run(test_batch=test_batch)
            )

            pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
                pyrtl_inference.simulate(test_batch=test_batch)
            )

            # Check the first layer's outputs.
            np.testing.assert_array_equal(
                pyrtl_layer0_output, numpy_layer0_output, strict=True
            )

            # Check the second layer's outputs.
            np.testing.assert_array_equal(
                pyrtl_layer1_output, numpy_layer1_output, strict=True
            )
            # Also verify that the actual predicted digits match.
            np.testing.assert_array_equal(pyrtl_actual, numpy_actual, strict=True)

    def test_pyrtl_inference_axi_uneven_batch(self) -> None:
        """
        Check that NumPyInference and PyRTLInference produce the same results, using
        uneven batches. PyrtlInference uses AXI-Lite in this test.

        This runs a two batches of different sizes through both inference systems and
        compares the tensor outputs from each layer.
        """

        start_image = 9
        batch_size = 2
        num_images = 3

        pyrtl_inference = PyRTLInference(
            tensor_path=self.temp_dir.name,
            input_bitwidth=8,
            accumulator_bitwidth=32,
            axi=True,
            batch_size=batch_size,
        )

        for _batch_number, (_batch_start_index, test_batch) in enumerate(
            batched_images(self.test_images, start_image, num_images, batch_size)
        ):
            numpy_layer0_output, numpy_layer1_output, numpy_actual = (
                self.numpy_inference.run(test_batch=test_batch)
            )

            pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
                pyrtl_inference.simulate(test_batch=test_batch)
            )

            # Check the first layer's outputs.
            np.testing.assert_array_equal(
                pyrtl_layer0_output, numpy_layer0_output, strict=True
            )

            # Check the second layer's outputs.
            np.testing.assert_array_equal(
                pyrtl_layer1_output, numpy_layer1_output, strict=True
            )
            # Also verify that the actual predicted digits match.
            np.testing.assert_array_equal(pyrtl_actual, numpy_actual, strict=True)


if __name__ == "__main__":
    unittest.main()
