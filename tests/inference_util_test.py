import tempfile
import unittest

import numpy as np

from pyrtlnet import inference_util
from pyrtlnet.mnist_util import load_mnist_images
from pyrtlnet.training_util import save_mnist_data


class TestInferenceUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Write resized MNIST test data to ``mnist_test_data.npz``."""
        (_, _), (cls.test_images, cls.test_labels) = load_mnist_images()

        cls.temp_dir = tempfile.TemporaryDirectory()
        save_mnist_data(
            tensor_path=cls.temp_dir.name,
            test_images=cls.test_images,
            test_labels=cls.test_labels,
        )

    def test_load_mnist_data(self) -> None:
        """Load the resized MNIST test data written by :func:`.save_mnist_data`."""
        test_images, test_labels = inference_util.load_mnist_data(self.temp_dir.name)

        self.assertEqual(test_images.shape, (10000, 12, 12))
        self.assertEqual(test_labels.shape, (10000,))

    def test_batched_images_batch_size_1(self) -> None:
        iterator = inference_util.batched_images(
            self.test_images, start_image=0, num_images=2, batch_size=1
        )

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, 0)
        self.assertEqual(test_batch.shape, (1, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, 1)
        self.assertEqual(test_batch.shape, (1, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertIsNone(batch_start_index)
        self.assertIsNone(test_batch)

    def test_batched_images_even_batch_size_2(self) -> None:
        iterator = inference_util.batched_images(
            self.test_images, start_image=0, num_images=2, batch_size=2
        )

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, 0)
        self.assertEqual(test_batch.shape, (2, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertIsNone(batch_start_index)
        self.assertIsNone(test_batch)

    def test_batched_images_uneven_batch_size_2(self) -> None:
        iterator = inference_util.batched_images(
            self.test_images, start_image=0, num_images=3, batch_size=2
        )

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, 0)
        self.assertEqual(test_batch.shape, (2, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, 2)
        self.assertEqual(test_batch.shape, (1, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertIsNone(batch_start_index)
        self.assertIsNone(test_batch)

    def test_batched_images_insufficient_images_1(self) -> None:
        """Test requesting more images than available."""
        # This iterator starts at the last image, and requests two images. It should
        # yield one image.
        last_image_index = len(self.test_images) - 1
        iterator = inference_util.batched_images(
            self.test_images, start_image=last_image_index, num_images=2, batch_size=1
        )

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, last_image_index)
        self.assertEqual(test_batch.shape, (1, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertIsNone(batch_start_index)
        self.assertIsNone(test_batch)

    def test_batched_images_insufficient_images_3(self) -> None:
        """Test requesting more images than available, with batching."""
        # This iterator starts at the third-from-last image, and requests five images in
        # batches of 2. It should yield three images in two batches, and the last batch
        # should be a partial batch.
        third_from_last_image_index = len(self.test_images) - 3
        iterator = inference_util.batched_images(
            self.test_images,
            start_image=third_from_last_image_index,
            num_images=5,
            batch_size=2,
        )

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, third_from_last_image_index)
        self.assertEqual(test_batch.shape, (2, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertEqual(batch_start_index, third_from_last_image_index + 2)
        self.assertEqual(test_batch.shape, (1, 12, 12))

        batch_start_index, test_batch = next(iterator, (None, None))
        self.assertIsNone(batch_start_index)
        self.assertIsNone(test_batch)

    def test_batched_images_invalid_start_image(self) -> None:
        iterator = inference_util.batched_images(
            self.test_images, start_image=999999, num_images=1, batch_size=1
        )

        with self.assertRaises(ValueError):
            next(iterator)

    def test_batched_images_invalid_batch_size(self) -> None:
        iterator = inference_util.batched_images(
            self.test_images, start_image=0, num_images=1, batch_size=0
        )
        with self.assertRaises(ValueError):
            next(iterator)

    def test_preprocess_image(self) -> None:
        # Create a test batch with two 12×12 images. The image data is just a sequence
        # of monotonically increasing even numbers, to simplify output validation later.
        test_batch_shape = (2, 12, 12)
        array = np.array(
            list(
                range(
                    0,
                    test_batch_shape[0] * test_batch_shape[1] * test_batch_shape[2] * 2,
                    2,
                )
            )
        )
        test_batch = np.reshape(array, newshape=test_batch_shape).astype(np.int8)

        preprocessed_batch = inference_util.preprocess_image(
            test_batch, input_scale=2, input_zero=1
        )

        def check_image(index: int) -> None:
            """Extract one image from the preprocessed batch and verify that it was
            normalized properly.
            """
            expected_image = (test_batch[index].flatten() / 2 + 1).astype(np.int8)
            np.testing.assert_array_equal(
                preprocessed_batch[:, index], expected_image, strict=True
            )

        check_image(0)
        check_image(1)


if __name__ == "__main__":
    unittest.main()
