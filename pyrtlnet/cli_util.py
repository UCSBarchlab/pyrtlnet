import math
from numbers import Number

import numpy as np


def _set_fg(r: int, g: int, b: int) -> str:
    """Return terminal escape codes to set the foreground color to ``{r, g, b}``.

    Requires a terminal that supports 24-bit color.

    :param r: Amount of red, in the range [0, 255].
    :param g: Amount of green, in the range [0, 255].
    :param b: Amount of blue, in the range [0, 255].

    :returns: A terminal escape code to change the current foreground color to
        ``{r, g b}``.

    """
    return f"\033[38;2;{r};{g};{b}m"


def _set_bg(r: int, g: int, b: int) -> str:
    """Return terminal escape codes to set the background color to ``{r, g, b}``.

    Requires a terminal that supports 24-bit color.

    :param r: Amount of red, in the range [0, 255].
    :param g: Amount of green, in the range [0, 255].
    :param b: Amount of blue, in the range [0, 255].

    :returns: A terminal escape code to change the current background color to
        ``{r, g b}``.

    """
    return f"\033[48;2;{r};{g};{b}m"


def _reset() -> str:
    """:returns: Terminal escape codes to reset the foreground and background colors."""
    return "\033[39m\033[49m"


def display_image(
    script_name: str,
    image: np.ndarray,
    image_index: int,
    batch_number: int,
    batch_index: int,
    verbose: bool,
) -> None:
    """Print an image as ASCII art in a terminal and its metadata.

    A header line is always printed, which looks like::

        LiteRT Inference image_index 2 batch_number 1 batch_index 0

    This header line displays the ``script_name`` (``LiteRT Inference``), the
    ``image_index`` (``2``), ``batch_number`` (``1``), and the ``batch_index`` (``0``).

    Next, the image is displayed, when ``verbose`` is ``True``. The image display
    requires a terminal that supports 24-bit color.

    The image is presented as a 2D array of grayscale pixel values. The pixel values are
    normalized such that the largest value displays as white, and the smallest value
    displays as black. One line of terminal output contains up to two rows of pixels.

    After the image, a footer line is displayed, when ``verbose`` is ``True``::

        shape (12, 12) dtype float32

    This footer line displays ``image``'s :attr:`~numpy.ndarray.shape` and
    :attr:`~numpy.ndarray.dtype`.

    :param script_name: Name of the script processing the image data.
    :param image: Image to display in the terminal.
    :param image_index: Index of the displayed image in the full test data set.
    :param batch_number: Batch number that the displayed image belongs to. Multiple
        consecutive images may be grouped into batches for processing. When this
        grouping occurs, multiple images will share the same ``batch_number``. The first
        batch processed is ``batch_number`` ``0``, the second batch processed is
        ``batch_number`` ``1``, and so on.
    :param batch_index: Index of the displayed image in its batch. When batching is
        disabled, the ``batch_size`` is ``1``, so every ``batch_index`` will be ``0``.
    :param verbose: When ``False``, only the header line is displayed. When ``True``,
        the header line, image, and footer line are all displayed.
    """
    print(
        f"{script_name} image_index {image_index} batch {batch_number} "
        f"batch_index {batch_index}",
    )

    if not verbose:
        return

    num_rows, num_cols = image.shape
    smallest = np.min(image)
    largest = np.max(image)

    def normalize(x: Number) -> Number:
        """Normalize a pixel value to the range [0, 255]."""
        return int(255 * (x + smallest) / (largest - smallest))

    for row in range(0, num_rows, 2):
        line = ""
        for col in range(num_cols):
            # The current row's normalized intensity determines the foreground color.
            fg = normalize(image[row][col])
            bg = 0
            if row + 1 < num_rows:
                # The next row's normalized intensity determines the background color.
                bg = normalize(image[row + 1][col])
            line += f"{_set_fg(fg, fg, fg)}{_set_bg(bg, bg, bg)}▀"
        print(line + _reset())

    print("shape", image.shape, "dtype", image.dtype, "\n")


def _bar(
    index: Number, x: Number, expected: Number, actual: Number, smallest: Number
) -> str:
    """Return a string representing a horizontal bar in a bar chart.

    The bar corresponding to the ``expected`` digit is always colored green. If the
    ``actual`` digit is not the same as the ``expected`` digit, the bar corresponding to
    the ``actual`` digit will be colored red.

    :param index: The digit that the current bar represents.
    :param x: The probability that the input is an image of the digit ``index``.
    :param expected: The expected digit.
    :param actual: The highest-probability digit, according to the model.
    :param smallest: The smallest ``x``-value in the chart. Space will be reserved so it
        can be displayed.
    :returns: A string representing a bar in a bar chart.

    """
    max_bar_length = 10
    bar_length = math.ceil(abs(x) / max_bar_length)
    negative_bar_length = math.ceil(abs(smallest) / max_bar_length)
    if x < 0:
        padding = negative_bar_length - bar_length
    else:
        padding = negative_bar_length

    bar = " " * padding + "▄" * bar_length
    green = _set_fg(0x2C, 0xA0, 0x2C)
    red = _set_fg(0xD6, 0x27, 0x28)
    if expected == actual and actual == index:
        return green + bar + _reset() + " " + str(x) + " (expected, actual)"
    if expected != actual and actual == index:
        return red + bar + _reset() + " " + str(x) + " (actual)"
    if expected != actual and expected == index:
        return green + bar + _reset() + " " + str(x) + " (expected)"
    return bar + " " + str(x)


def display_outputs(
    script_name: str,
    layer0_output: np.ndarray,
    layer1_output: np.ndarray,
    expected: int,
    actual: int,
    verbose: bool,
    transposed_outputs: bool,
) -> None:
    """Display the neural network's outputs.

    Prints the raw outputs of each neural network layer, followed by a bar chart that
    interprets the final layer's output as each digit's un-normalized probability.

    Bars for higher probability digits are displayed before bars for lower probability
    digits.

    The bar corresponding to the ``expected`` digit is always colored green. If the
    ``actual`` digit is not the same as the ``expected`` digit, the bar corresponding to
    the ``actual`` digit will be colored red.

    Sample output with colors omitted::

        LiteRT Inference layer0 output shape (18,) dtype int8:
        [-123 -114 -123  -76 -123  -23  -94 -123  -65  -68 -123   -1  -63 -112 -123 ...]

        LiteRT Inference layer1 output shape (10,) dtype int8:
        [ 33 -48  29  58 -50  31 -87  93   9  49]

        LiteRT Inference layer1 output as bar chart:
        7▕          ▄▄▄▄▄▄▄▄▄▄ 93 (expected, actual)
        3▕          ▄▄▄▄▄▄ 58
        9▕          ▄▄▄▄▄ 49
        0▕          ▄▄▄▄ 33
        5▕          ▄▄▄▄ 31
        2▕          ▄▄▄ 29
        8▕          ▄ 9
        1▕     ▄▄▄▄▄ -48
        4▕     ▄▄▄▄▄ -50
        6▕ ▄▄▄▄▄▄▄▄▄ -87

    In the sample output above, the digit corresponding to each bar is displayed on the
    left, so the digit ``7`` has the highest probability, followed by the digit ``3``.
    The model predicted the digit is a ``7``, and the digit actually was a ``7``
    according to the labeled test data, so the first bar is annotated with ``(expected,
    actual)``.

    :param script_name: Name of the script processing the image data.
    :param layer0_output: Output of the neural network's first layer.
    :param layer1_output: Output of the neural network's second layer.
    :param expected: Expected prediction from labeled training data.
    :param actual: Actual prediction from the neural network.
    :param verbose: When ``False``, just print a summary of the expected and actual
        predictions. When ``True``, print each layer's output and an annotated bar
        chart.
    :param transposed_outputs: When ``True``, print ``(transposed)`` to indicate that
        the outputs have been transposed.
    """
    if not verbose:
        print(f"Expected: {expected} | Actual: {actual}")
        return

    transposed = ""
    if transposed_outputs:
        transposed = " (transposed)"

    print(
        f"{script_name} layer0 output{transposed} shape {layer0_output.shape} "
        f"dtype {layer0_output.dtype}:",
    )
    print(layer0_output, "\n")

    print(
        f"{script_name} layer1 output{transposed} shape {layer1_output.shape} "
        f"dtype {layer1_output.dtype}:",
    )
    print(layer1_output, "\n")

    print(f"{script_name} layer1 output as bar chart:")

    # If all outputs are positive, start the x-axis at 0, otherwise start the x-axis at
    # the smallest negative number.
    smallest = np.min(layer1_output)
    if smallest > 0:
        smallest = 0
    # Enumerate the probabilities. Each enumeration index corresponds to a digit, so
    # `chart_data` is a list of {digit, probability} pairs. Sort by probability,
    # highest to lowest.
    chart_data = sorted(
        enumerate(layer1_output), reverse=True, key=lambda pair: pair[1]
    )
    for index, value in chart_data:
        print(f"{index}▕ {_bar(index, value, expected, actual, smallest)}")


class Accuracy:
    """Update and display accuracy statistics over multiple tests."""

    def __init__(self) -> None:
        self.num_updates = 0
        self.correct = 0

    def update(self, actual: int, expected: int) -> None:
        """Update accuracy statistics for a single test.

        Records a correct prediction when ``actual == expected``.

        :param actual: Actual outcome of the test. This is the actual output of the
            neural network.
        :param expected: Expected outcome of the test. This is the output we're
            expecting, according to the labeled test data.
        """
        self.num_updates += 1
        if actual == expected:
            self.correct += 1

    def display(self) -> None:
        """Display accuracy statistics over all tests.

        The printed summary looks like:

        .. code-block:: text

            9/10 correct predictions, 90.0% accuracy
        """
        if self.num_updates > 1:
            print(
                f"{self.correct}/{self.num_updates} correct predictions, "
                f"{100.0 * self.correct / self.num_updates:.1f}% accuracy"
            )
