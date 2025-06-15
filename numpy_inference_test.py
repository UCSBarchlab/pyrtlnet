import tempfile
import unittest

import numpy as np

import litert_inference
import mnist_util
import numpy_inference
import tensorflow_training


class TestNumPyInference(unittest.TestCase):
    def setUp(self):
        """Prepare a LiteRT Interpreter and NumPyInference for a comparison test.

        This trains a quantized TensorFlow model for one epoch, to reduce run time.

        The trained model is loaded in the LiteRT Interpreter, and an instance of
        NumPyInference.

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
            self.numpy_inference = numpy_inference.NumPyInference(self.interpreter)

    def test_numpy_inference(self):
        """Check that LiteRT Interpreter and NumPyInference produce the same results.

        This runs one image through both inference systems and compares the tensor
        outputs from each layer.

        """
        test_image = self.test_images[0]

        litert_layer0_output, litert_layer1_output, litert_actual = (
            litert_inference.run_tflite_model(
                interpreter=self.interpreter, test_image=test_image
            )
        )

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_image=test_image)
        )

        # Check the first layer's outputs.
        self.assertTrue(np.array_equal(litert_layer0_output.T, numpy_layer0_output))

        # Check the second layer's outputs.
        self.assertTrue(np.array_equal(litert_layer1_output.T, numpy_layer1_output))

        # Also verify that the actual predicted digits match.
        self.assertEqual(litert_actual, numpy_actual)


if __name__ == "__main__":
    unittest.main()
