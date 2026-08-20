import time
from numbers import Number

import numpy as np


# Escape codes:
# https://en.wikipedia.org/wiki/ANSI_escape_code#Select_Graphic_Rendition_parameters
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


def _reverse_video() -> str:
    """:returns: Terminal escape code to swap foreground and background colors."""
    return "\033[7m"


def _reset() -> str:
    """:returns: Terminal escape code to disable all attributes."""
    return "\033[0m"


def force_verbose(num_images: int, batch_size: int, terminal_width: int) -> bool:
    """Returns ``True`` iff displaying a single batch, and the batch's images will fit
    in the terminal.
    """
    return (
        num_images <= batch_size and min(num_images, batch_size) * 12 <= terminal_width
    )


def display_images(
    script_name: str,
    images: np.ndarray,
    image_indices: list[int],
    batch_number: int,
    verbose: bool,
) -> None:
    """Display MNIST images as ASCII art in a terminal.

    A header line with metadata is always printed, which looks like::

        LiteRT Inference image_indices [2] batch_number 1

    This header line displays the ``script_name`` (``LiteRT Inference``), the
    ``image_indices`` (``[2]``), ``batch_number`` (``1``).

    Next, the images are displayed, when ``verbose`` is ``True``. Image display requires
    a terminal that supports 24-bit color.

    The images are presented as a 2D array of grayscale pixel values. The pixel values
    are normalized such that the largest value displays as white, and the smallest value
    displays as black. One line of terminal output contains up to two rows of pixels. If
    `images` contains multiple images, the images will be horizontally stacked
    side-by-side.

    After the images, a footer line is displayed, when ``verbose`` is ``True``::

        shape (1, 12, 12) dtype float32

    This footer line displays ``images``' :attr:`~numpy.ndarray.shape` and
    :attr:`~numpy.ndarray.dtype`.

    :param script_name: Name of the script processing the image data.
    :param images: Images to display in the terminal.
    :param image_indices: List of indices of the displayed image in the full test data
        set.
    :param batch_number: Batch number that the displayed images belong to. Multiple
        consecutive images may be grouped into batches for processing. When this
        grouping occurs, multiple images will share the same ``batch_number``. The first
        batch processed is ``batch_number`` ``0``, the second batch processed is
        ``batch_number`` ``1``, and so on.
    :param verbose: When ``False``, only the header line is displayed. When ``True``,
        the header line, image, and footer line are all displayed.
    """
    print(f"{script_name} image_indices {image_indices} batch {batch_number}")

    if not verbose:
        return

    shape_str = str(images.shape)
    if images.ndim == 3:
        images = np.hstack(images)

    num_rows, num_cols = images.shape
    smallest = np.min(images)
    largest = np.max(images)

    def normalize(x: Number) -> Number:
        """Normalize a pixel value to the range [0, 255]."""
        return int(255 * (x + smallest) / (largest - smallest))

    for row in range(0, num_rows, 2):
        line = ""
        for col in range(num_cols):
            # The current row's normalized intensity determines the foreground color.
            fg = normalize(images[row][col])
            bg = 0
            if row + 1 < num_rows:
                # The next row's normalized intensity determines the background color.
                bg = normalize(images[row + 1][col])
            line += f"{_set_fg(fg, fg, fg)}{_set_bg(bg, bg, bg)}▀"
        print(line + _reset())

    print("shape", shape_str, "dtype", images.dtype, "\n")


def _blocks(start: int, end: int) -> str:
    """Return a string containing consecutive blocks in [start, end).

    The returned string can be included in a bar chart.

    Bar charts are rendered at half-character resolution with Unicode quadrant block
    elements ("▗" "▄" "▖"). A character can contain 0-2 of these blocks. These
    characters can be assembled into a continuous bar, and the bar can start or stop
    halfway between characters. Example::

        >>> for start in range(4):
        ...     for end in range(start + 1, 5):
        ...         print(start, end, _blocks(start, end))
        ...
        0 1 ▖
        0 2 ▄
        0 3 ▄▖
        0 4 ▄▄
        1 2 ▗
        1 3 ▗▖
        1 4 ▗▄
        2 3  ▖
        2 4  ▄
        3 4  ▗

    This example shows four blocks, numbered [0, 1, 2, 3]. All four blocks are displayed
    with two characters::

        >>> len(_blocks(0, 4))
        2

    :param start: Starting block index, inclusive
    :param end: Ending block index, exclusive.

    :return: A string containing the specified blocks.
    """
    assert start <= end

    def round_down(x: int) -> int:
        """Round `x` down to the nearest even number."""
        return x - x % 2

    # Determine how many spaces to include before the bar. Each space covers two blocks.
    num_spaces = round_down(start) // 2
    spaces = " " * num_spaces

    # If the bar starts mid-character, emit an initial "▗" and increment `start`.
    # If the bar ends mid-character, emit a final "▖" and decrement `end`.
    first = ""
    last = ""
    if start < end:
        if start % 2 == 1:
            first = "▗"
            start += 1
        if end % 2 == 1:
            last = "▖"
            end -= 1

    # Figure out how many block-pairs "▄" to emit.
    num_pairs = (end - start) // 2
    pairs = "▄" * num_pairs

    return f"{spaces}{first}{pairs}{last}"


def _bar(
    index: Number,
    x: Number,
    chart_width: int,
    expected: Number,
    actual: Number,
    include_description: bool,
    smallest: Number,
    largest: Number,
) -> str:
    """Return a string representing a horizontal bar in a bar chart.

    The bar corresponding to the ``expected`` digit is always colored green. If the
    ``actual`` digit is not the same as the ``expected`` digit, the bar corresponding to
    the ``actual`` digit will be colored red.

    :param index: The digit that the current bar represents.
    :param x: The probability that the input is an image of the digit ``index``.
    :param chart_width: Total width of the chart, in characters.
    :param expected: The expected digit.
    :param actual: The highest-probability digit, according to the model.
    :param include_description: If ``True``, add "expected" and "actual" to the
        corresponding bars.
    :param smallest: The smallest ``x``-value in the chart. Space will be reserved so it
        can be displayed.
    :param largest: The largest ``x``-value in the chart. Space will be reserved so it
        can be displayed.

    :returns: A string representing a bar in a bar chart.
    """
    # Convert from `np.int8` to `int`.
    smallest = int(smallest)
    largest = int(largest)
    x = int(x)

    # Length of the longest {x} printed at the end of each bar, plus one to account for
    # the leading space.
    label_length = max(len(str(smallest)), len(str(largest))) + 1

    # Actual length of each bar, in characters. Minus two accounts for the "{index}│"
    # printed before each bar.
    bar_length = chart_width - 2 - label_length

    # Bars are rendered at half-character resolution with Unicode quadrant block
    # elements ("▄" "▗" "▖"). `block_length` counts these blocks, "▄" is two blocks,
    # while "▗" and "▖" are one block each.
    block_length = bar_length * 2
    # Blocks are numbered [0, block_length), so (block_length - 1) is the last valid
    # block number.
    blocks_per_unit = (block_length - 1) / (largest - smallest)

    # Figure out where the bar starts and ends, in terms of blocks.
    def _to_block_index(x: Number) -> int:
        """Convert `x` to a block index in the range [0, block_length)."""
        return round((x - smallest) * blocks_per_unit)

    start_block = _to_block_index(0)
    end_block = _to_block_index(x)
    if x < 0:
        start_block, end_block = end_block, start_block

    # Actually draw the bar.
    assert start_block >= 0
    assert end_block < block_length
    bar = _blocks(start_block, end_block)

    # If the prediction was incorrect, make the bar red. The bar for the expected label
    # is always green.
    green = _set_fg(0x2C, 0xA0, 0x2C)
    red = _set_fg(0xD6, 0x27, 0x28)

    color = ""
    description = ""
    if expected == actual and actual == index:
        color = green
        description = " (expected, actual)"
    elif expected != actual:
        if expected == index:
            color = green
            description = " (expected)"
        elif actual == index:
            color = red
            description = " (actual)"

    # Highlight the predicted digit with reverse-video.
    reverse_video = ""
    divider = "│"
    if x == largest:
        reverse_video = _reverse_video()
        divider = "▌"

    # Pad the bar out to `chart_width` so we can display horizontally-stacked bar
    # charts. We can't use `str.ljust` here because the string contains escape codes.
    right_padding = " " * (chart_width - len(f"{index}│{bar} {x}"))
    if not include_description:
        description = ""
    return (
        f"{reverse_video}{index}{_reset()}{divider}{color}{bar}{_reset()} {x}"
        f"{right_padding}{description}"
    )


def display_outputs(
    script_name: str,
    layer0_output: np.ndarray,
    layer1_output: np.ndarray,
    expected: int,
    actual: int,
    verbose: bool,
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

        LiteRT Inference layer0 output shape (1, 18) dtype int8:
        [[-123 -114 -123  -76 -123  -23  -94 -123  -65  -68 -123   -1  -64 -112 ...]]

        LiteRT Inference layer1 output shape (1, 10) dtype int8:
        [[ 33 -48  29  58 -50  31 -87  93   9  49]]

        LiteRT Inference layer1 output as bar chart:
        7▌       ▗▄▄▄▄▄▄▄▖ 93  (expected, actual)
        3│       ▗▄▄▄▄▖ 58
        9│       ▗▄▄▄▖ 49
        0│       ▗▄▄▖ 33
        5│       ▗▄▄ 31
        2│       ▗▄▄ 29
        8│       ▗▖ 9
        1│   ▗▄▄▄▖ -48
        4│   ▄▄▄▄▖ -50
        6│▄▄▄▄▄▄▄▖ -87

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
    """
    assert len(expected) == len(actual)
    if not verbose:
        green = _set_fg(0x2C, 0xA0, 0x2C)
        red = _set_fg(0xD6, 0x27, 0x28)

        expected_parts = []
        actual_parts = []
        for current_expected, current_actual in zip(expected, actual, strict=True):
            if current_expected == current_actual:
                actual_parts.append(str(current_actual))
                expected_parts.append(str(current_expected))
            else:
                actual_parts.append(f"{red}{current_actual}{_reset()}")
                expected_parts.append(f"{green}{current_expected}{_reset()}")

        expected = " ".join(expected_parts)
        actual = " ".join(actual_parts)

        print(f"Expected: [{expected}]")
        print(f"  Actual: [{actual}]")
        return

    print(
        f"{script_name} layer0 output shape {layer0_output.shape} "
        f"dtype {layer0_output.dtype}:",
    )
    print(layer0_output, "\n")

    print(
        f"{script_name} layer1 output shape {layer1_output.shape} "
        f"dtype {layer1_output.dtype}:",
    )
    print(layer1_output, "\n")

    print(f"{script_name} layer1 output as bar chart:")

    # Display a horizontally-stacked series of bar charts, one for each `batch_index`.
    #
    # `chart_limits` tracks the `x`-axis range for each chart.
    chart_limits = []
    for batch_output in layer1_output:
        # If all of a chart's outputs are positive, start the chart's x-axis at 0,
        # otherwise start the chart's x-axis at the smallest negative output.
        smallest = np.min(batch_output)
        smallest = min(smallest, 0)

        # If all of a chart's outputs are negative, end the chart's x-axis at 0,
        # otherwise end the chart's x-axis at the largest negative output.
        largest = np.max(batch_output)
        largest = max(largest, 0)

        chart_limits.append([smallest, largest])

    # Enumerate each batch's probabilities, then reverse-sort by probability.
    chart_data = []
    for batch_output in layer1_output:
        chart_data.append(
            sorted(enumerate(batch_output), reverse=True, key=lambda pair: pair[1])
        )

    # If we are displaying multiple charts, reduce the chart width and omit the bar
    # descriptions ("expected", "actual").
    chart_width = 22
    include_description = True
    if len(layer1_output) > 1:
        chart_width = 11
        include_description = False

    # Transpose `chart_data` to make it easier to display. After transposing,
    # `chart_data[0]` contains all the data we'll display in the first row of output,
    # which are the highest-probability digits, for each image in the batch.
    chart_data = np.transpose(np.array(chart_data), axes=[1, 0, 2])

    # Render each `row` of `chart_data`. Each `row` contains data for each image in the
    # batch. Render each batch's data as a `_bar()`.
    for row in chart_data:
        line_parts = []
        for batch_index, column in enumerate(row):
            index = column[0]
            value = column[1]
            line_parts.append(
                _bar(
                    index,
                    value,
                    chart_width,
                    expected[batch_index],
                    actual[batch_index],
                    include_description,
                    *chart_limits[batch_index],
                )
            )
        print(" ".join(line_parts))


class Accuracy:
    """Update and display accuracy statistics over multiple tests."""

    def __init__(self) -> None:
        self.num_updates = 0
        self.correct = 0

    def update(self, actual: list[int], expected: list[int]) -> None:
        """Update accuracy statistics for a batch of tests.

        Records a correct prediction when ``actual == expected``.

        :param actual: Actual outcomes of the test. This is the actual output of the
            neural network.
        :param expected: Expected outcomes of the test. This is the output we're
            expecting, according to the labeled test data.
        """
        for current_actual, current_expected in zip(actual, expected, strict=True):
            self.num_updates += 1
            if current_actual == current_expected:
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


def trim_batch(
    batch_start_index: int,
    batch_number: int,
    batch_size: int,
    num_images: int,
    layer0_output: np.ndarray,
    layer1_output: np.ndarray,
    actual: list[int],
    test_labels: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray, list[int], list[int]]:
    """Remove data corresponding to null images added by :func:`.batched_images`.

    The last batch may not be full. When that happens, :func:`.batched_images` pads the
    batch with null images. ``trim_batch`` filters out any results corresponding to
    these null images.

    :param batch_start_index: Index of the first image in the batch.
    :param batch_number: The first batch is ``batch_number`` 0, the second
        ``batch_number`` 1, and so on.
    :param batch_size: Number of images in each batch.
    :param num_images: Total number of images to process.
    :param layer0_output: Output of the neural network's first layer.
    :param layer1_output: Output of the neural network's second and final layer.
    :param actual: List of actual predicted digits for each image in the batch.

    :return: ``image_indices``, ``layer0_output``, ``layer1_output``, ``actual``,
             ``expected``, where each has been trimmed to remove data corresponding to
             null images. ``image_indices`` is a list of indices in the test data set
             for images included in the batch. ``expected`` is a list of the expected
             digit labels, from the labeled test data.
    """
    # Number of images in the batch, excluding any null images added by
    # `batched_images`.
    num_batch_images = batch_size
    if (batch_number + 1) * batch_size > num_images:
        num_batch_images = num_images % batch_size

    image_indices = list(range(batch_start_index, batch_start_index + num_batch_images))
    expected = test_labels[batch_start_index : batch_start_index + num_batch_images]
    return (
        image_indices,
        layer0_output[:num_batch_images],
        layer1_output[:num_batch_images],
        actual[:num_batch_images],
        expected,
    )


class PrintElapsedTime:
    """Report how long it takes to run the code in a ``with`` statement.

    This context manager first prints a ``message``, then runs the code in the ``with``
    statement, then prints a ``"done"`` message followed by the elapsed time. All output
    is printed on one line.

    When an interactive script pauses for more than a second, users will start to wonder
    if something is wrong. So use ``PrintElapsedTime`` to let the user know what's going
    on before starting an operation that's expected to take more than a second.

    Example::

        with PrintElapsedTime(message="Sleeping"):
            time.sleep(2)

    Example output::

        Sleeping... done (2.00 seconds)
    """

    def __init__(self, message: str) -> None:
        """
        :param message: Message to print before running the code in the ``with``
            statement.
        """
        self.message = message

    def __enter__(self) -> None:
        self.start = time.time()
        print(f"{self.message}... ", end="", flush=True)

    def __exit__(self, *exception_info) -> None:  #  noqa: ANN002
        elapsed = time.time() - self.start
        units = "s"

        if elapsed < 0.001:
            elapsed *= 1_000_000
            units = "µs"
        elif elapsed < 1:
            elapsed *= 1000
            units = "ms"
        print(f"done ({elapsed:.2f} {units})")
