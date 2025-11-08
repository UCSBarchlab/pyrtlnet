import numpy as np


def preprocess_image(
    test_batch: np.ndarray, input_scale: np.ndarray, input_zero: np.ndarray
) -> np.ndarray:
    """Preprocess the raw image data in the batch. This is required by the quantized
    neural network.

    This adjusts the batch image data by ``input_scale`` and ``input_zero``. Then,
    it flattens each 2D image into a 1D column vector and stores them in a matrix of
    shape ``(144, batch_size)``.

    :param test_batch: Batch data to preprocess. This data should have already been
        normalized to ``[0.0, 1.0]`` and resized to ``(batch_size, 12, 12)``,
        usually by :func:`~pyrtlnet.mnist_util.load_mnist_images`.
    :param input_scale: Scale factor for ``test_batch``.
    :param input_zero: Zero point for ``test_batch``.

    :returns: Flattened batch data of shape ``(144, batch_size)``, adjusted by the
              quantized neural network's ``input_scale`` and ``input_zero``.
    """
    # The MNIST image data contains pixel values in the range [0, 255]. The neural
    # network was trained by first converting these values to floating point, in the
    # range [0, 1.0]. Dividing by input_scale below undoes this conversion,
    # converting the range from [0, 1.0] back to [0, 255].
    #
    # We could avoid these back-and-forth conversions by modifying
    # `load_mnist_images()` to skip the first conversion, and returning `x +
    # input_zero_point` below to skip the second conversion, but we do them anyway
    # to simplify the code and make it more consistent with existing sample code
    # like https://ai.google.dev/edge/litert/models/post_training_integer_quant
    #
    # Adding input_zero (-128) effectively converts the uint8 image data to int8, by
    # shifting the range [0, 255] to [-128, 127].
    test_batch = (test_batch / input_scale + input_zero).astype(np.int8)

    # Taking test_batch of shape (batch_size, 12, 12), each 2D matrix of shape
    # (12,12) is flattened to a 1D column vector of shape (144,), resulting in
    # test_batch's shape becoming (batch_size, 144). Then, we transpose, returning
    # the final shape (144, batch_size), where there are batch_size amount of column
    # vectors of shape (144,), each representing one image.
    return test_batch.reshape(test_batch.shape[0], -1).transpose()
