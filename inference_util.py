import math

import numpy as np


def set_fg(r: int, g: int, b: int):
    """Emit escape codes to set the foreground color to {r, g, b}.

    Requires a terminal that supports 24-bit color.

    """
    return f"\033[38;2;{r};{g};{b}m"


def set_bg(r: int, g: int, b: int):
    """Emit escape codes to set the background color to {r, g, b}.

    Requires a terminal that supports 24-bit color.

    """
    return f"\033[48;2;{r};{g};{b}m"


def reset():
    """Emit escape codes to reset the foreground and background colors."""
    return "\033[39m\033[49m"


def display_image(image: np.ndarray):
    """Render an image as ASCII art in a terminal.

    The image is a 2D array of grayscale pixel values. The pixel values will be
    normalized such that the largest value displays as white, and the smallest value
    displays as black.

    One line of terminal output contains up to two rows of pixels.

    """
    num_rows, num_cols = image.shape
    smallest = np.min(image)
    largest = np.max(image)

    def normalize(x):
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
            line += f"{set_fg(fg, fg, fg)}{set_bg(bg, bg, bg)}▀"
        print(line + reset())


def bar(index, x, expected, actual, smallest):
    """Draw a horizontal bar for a bar chart.

    `index` is the digit that the current bar represents.
    `expected` is the expected digit.
    `actual` is the highest-probability digit, according to the model.
    `smallest` is the smallest x-value in the chart. Space will be reserved so it can be
        displayed.

    The bar corresponding to the `expected` digit is always colored green. If the
    `actual` digit is not the same as the `expected` digit, the bar corresponding to the
    `actual` digit will be colored red.

    """
    max_bar_length = 10
    bar_length = int(math.ceil(abs(x) / max_bar_length))
    negative_bar_length = int(math.ceil(abs(smallest) / max_bar_length))
    if x < 0:
        padding = negative_bar_length - bar_length
    else:
        padding = negative_bar_length

    bar = " " * padding + "▄" * bar_length
    green = set_fg(0x2C, 0xA0, 0x2C)
    red = set_fg(0xD6, 0x27, 0x28)
    if expected == actual and actual == index:
        return green + bar + reset() + " " + str(x) + " (expected, actual)"
    elif expected != actual and actual == index:
        return red + bar + reset() + " " + str(x) + " (actual)"
    elif expected != actual and expected == index:
        return green + bar + reset() + " " + str(x) + " (expected)"
    return bar + " " + str(x)


def display_outputs(output: np.ndarray, expected: int, actual: int):
    """Display the neural network's output as a bar chart.

    Bars for higher probability digits are displayed before bars for lower probability
    digits.

    The bar corresponding to the `expected` digit is always colored green. If the
    `actual` digit is not the same as the `expected` digit, the bar corresponding to the
    `actual` digit will be colored red. Colors are not shown in the sample below.

    Sample output:

    7▕           ▄▄▄▄▄▄▄▄▄▄▄ 106 (expected, actual)
    3▕           ▄▄▄▄▄▄ 59
    2▕           ▄▄▄▄▄ 44
    9▕           ▄▄▄▄▄ 44
    8▕           ▄▄▄ 27
    0▕           ▄▄▄ 24
    5▕           ▄ 4
    4▕        ▄▄▄ -30
    1▕       ▄▄▄▄ -31
    6▕ ▄▄▄▄▄▄▄▄▄▄ -100

    In the sample output above, the digit corresponding to each bar is displayed on the
    left, so the digit 7 has the highest probability, followed by the digit 3. The model
    predicted the digit is a 7, and the digit actually was a 7, so the first bar is
    annotated with "(expected, actual)".

    """
    # If all outputs are positive, start the x-axis at 0, otherwise start the x-axis at
    # the smallest negative number.
    smallest = np.min(output)
    if smallest > 0:
        smallest = 0
    # Enumerate the probabilities. Each enumeration index corresponds to a digit, so
    # `chart_data` is a list of {digit, probability} pairs. Sort by probability,
    # highest to lowest.
    chart_data = sorted(enumerate(output), reverse=True, key=lambda pair: pair[1])
    for index, value in chart_data:
        print(f"{index}▕ {bar(index, value, expected, actual, smallest)}")
