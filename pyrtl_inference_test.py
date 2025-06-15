import tempfile
import unittest

import numpy as np
from ai_edge_litert.interpreter import Interpreter

import mnist_util
import numpy_inference
import pyrtl_inference
import tensorflow_training


class TestPyRTLInference(unittest.TestCase):
    def setUp(self):
        """Prepare NumPyInference and PyRTLInference for a comparison test.

        This trains a quantized TensorFlow model for one epoch, to reduce run time.

        The trained model is loaded in instances of NumPyInference and PyRTLInference.

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
            interpreter = Interpreter(model_path=model_file_name)
            self.numpy_inference = numpy_inference.NumPyInference(interpreter)
            self.pyrtl_inference = pyrtl_inference.PyRTLInference(
                interpreter, input_bitwidth=8, accumulator_bitwidth=32
            )

    def test_pyrtl_inference(self):
        """Check that NumPyInference and PyRTLInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.

        """
        test_image = self.test_images[0]

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_image=test_image)
        )

        pyrtl_layer0_output, pyrtl_layer1_output, pyrtl_actual = (
            self.pyrtl_inference.run(test_image=test_image)
        )

        # Check the first layer's outputs.
        self.assertTrue(np.array_equal(numpy_layer0_output, pyrtl_layer0_output))

        # Check the second layer's outputs.
        self.assertTrue(np.array_equal(numpy_layer1_output, pyrtl_layer1_output))

        # Also verify that the actual predicted digits match.
        self.assertEqual(numpy_actual, pyrtl_actual)


if __name__ == "__main__":
    unittest.main()
