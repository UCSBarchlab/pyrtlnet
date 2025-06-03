import math

import numpy as np
import tensorflow as tf


def set_fg(r: int, g: int, b: int):
    return f"\033[38;2;{r};{g};{b}m"


def set_bg(r: int, g: int, b: int):
    return f"\033[48;2;{r};{g};{b}m"


def reset():
    return "\033[39m\033[49m"


def display_image(image: np.ndarray):
    num_rows, num_cols = image.shape
    smallest = np.min(image)
    largest = np.max(image)

    def normalize(x):
        return int(255 * (x + smallest) / (largest - smallest))

    for row in range(0, num_rows, 2):
        line = ""
        for col in range(num_cols):
            fg = normalize(image[row][col])
            bg = 0
            if row + 1 < num_rows:
                bg = normalize(image[row + 1][col])
            line += f"{set_fg(fg, fg, fg)}{set_bg(bg, bg, bg)}▀"
        print(line + reset())


def bar(index, x, expected, actual, smallest):
    bar_length = int(math.ceil(abs(x) / 10))
    negative_bar_length = int(math.ceil(abs(smallest) / 10))
    if x < 0:
        padding = negative_bar_length - bar_length
    else:
        padding = negative_bar_length

    bar = " " * padding + "▄" * bar_length
    green_bar = set_fg(0x2C, 0xA0, 0x2C) + bar + reset()
    red_bar = set_fg(0xD6, 0x27, 0x28) + bar + reset()
    if expected == actual and actual == index:
        return green_bar + " " + str(x) + " (expected, actual)"
    elif expected != actual and actual == index:
        return red_bar + " " + str(x) + " (actual)"
    elif expected != actual and expected == index:
        return green_bar + " " + str(x) + " (expected)"
    return bar + " " + str(x)


def display_outputs(output: np.ndarray, expected: int, actual: int):
    smallest = np.min(output)
    if smallest > 0:
        smallest = 0
    chart_data = sorted(enumerate(output), reverse=True, key=lambda pair: pair[1])
    for index, value in chart_data:
        print(f"{index}▕ {bar(index, value, expected, actual, smallest)}")


def resize_images(images, new_shape):
    images = tf.constant(images)
    images = images[..., tf.newaxis]
    images = tf.image.resize(images, new_shape)[..., 0].numpy()
    return images


def preprocess_images(images):
    assert len(images) > 0
    assert images[0].shape == (28, 28)

    # Resize images from 28×28 to 12×12.
    new_size = (12, 12)

    # Normalize images.
    return resize_images(images, new_size)
