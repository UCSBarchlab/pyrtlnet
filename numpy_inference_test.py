import tempfile
import unittest

import litert_inference
import mnist_util
import numpy as np
import numpy_inference
import tensorflow_training

class TestNumpyInference(unittest.TestCase):
    def setUp(self):
        (train_images, train_labels), (self.test_images, self.test_labels) = (
            mnist_util.load_mnist_images())

        with tempfile.NamedTemporaryFile(prefix="quantized_tflite_model") as file:
            model_file_name = file.name

            learning_rate = 0.001
            epochs = 1

            model = tensorflow_training.train_unquantized_model(
                learning_rate=learning_rate, epochs=epochs,
                train_images=train_images, train_labels=train_labels)
            model = tensorflow_training.quantize_model(
                model=model, learning_rate=learning_rate / 10000, epochs=epochs,
                train_images=train_images, train_labels=train_labels,
                model_file_name=model_file_name)
            self.interpreter = litert_inference.load_tflite_model(
                model_file_name=model_file_name)
            self.numpy_inference = numpy_inference.NumpyInference(self.interpreter)


    def test_numpy_inference(self):
        test_image = self.test_images[0]

        litert_layer0_output, litert_layer1_output, litert_actual = (
            litert_inference.run_tflite_model(
                interpreter=self.interpreter, test_image=test_image))

        numpy_layer0_output, numpy_layer1_output, numpy_actual = (
            self.numpy_inference.run(test_image=test_image))

        self.assertEqual(litert_layer0_output.size, numpy_layer0_output.size)
        litert_layer0_output = np.reshape(
            litert_layer0_output, numpy_layer0_output.shape)

        self.assertTrue(np.array_equal(litert_layer0_output, numpy_layer0_output))

        self.assertEqual(litert_layer1_output.size, numpy_layer1_output.size)
        litert_layer1_output = np.reshape(
            litert_layer1_output, numpy_layer1_output.shape)

        self.assertTrue(np.array_equal(litert_layer1_output, numpy_layer1_output))

        self.assertEqual(litert_actual, numpy_actual)


if __name__ == "__main__":
    unittest.main()
