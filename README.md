`pyrtlnet`
==========

[![Build Status](https://github.com/UCSBarchlab/pyrtlnet/actions/workflows/python-test.yml/badge.svg)](https://github.com/UCSBarchlab/pyrtlnet/actions/workflows/python-test.yml)
[![Documentation Status](https://readthedocs.org/projects/pyrtlnet/badge/?version=latest)](http://pyrtlnet.readthedocs.org/en/latest/?badge=latest)

Train it. Quantize it. Synthesize and simulate it — in hardware. All in Python.

`pyrtlnet` is a self-contained example of a quantized neural network that runs
end-to-end in Python. From model training, to software inference, to hardware
generation, all the way to simulating that custom inference hardware at the logic-gate
level — you can do it all right from the Python REPL. We hope you will find `pyrtlnet`
(rhymes with turtle-net) a complete and understandable walkthrough that goes from
[TensorFlow](https://www.tensorflow.org/) training to bit-accurate hardware simulation,
with the [PyRTL](https://github.com/UCSBarchlab/PyRTL) hardware description language.
Main features include:

* Quantized neural network training with [TensorFlow](https://www.tensorflow.org/). The
  resulting inference network is fully quantized, so all inference calculations are done
  with integers.

* Four different quantized inference implementations operating at different levels of
  abstraction. All four implementations produce the same output in the same format and,
  in doing so, provide a useful framework to extend either from the top-down or the
  bottom-up.

  1. A reference quantized inference implementation, using the standard
     [LiteRT](https://ai.google.dev/edge/litert) `Interpreter`.

  1. A software implementation of quantized inference, using [NumPy](https://numpy.org)
     and [fxpmath](https://github.com/francof2a/fxpmath), to verify the math performed
     by the reference implementation.

  1. A [PyRTL](https://github.com/UCSBarchlab/PyRTL) hardware implementation of
     quantized inference that is simulated right at the logic gate level.

  1. A deployment of the PyRTL hardware design to a
     [Pynq Z2 FPGA](https://www.amd.com/en/corporate/university-program/aup-boards/pynq-z2.html).

* A new [PyRTL](https://github.com/UCSBarchlab/PyRTL) linear algebra library, including
  a composable `WireMatrix2D` matrix abstraction and an output-stationary [systolic
  array](https://en.wikipedia.org/wiki/Systolic_array) for matrix multiplication.

* An extensive [suite of unit
  tests](https://github.com/UCSBarchlab/pyrtlnet/tree/main/tests), and [continuous
  integration testing](https://github.com/UCSBarchlab/pyrtlnet/actions).

* Understandable and
  [documented](https://pyrtlnet.readthedocs.io/en/latest/index.html) code!
  `pyrtlnet` is designed to be, first and foremost, understandable and readable
  (even when that comes at the expense of performance). [Reference
  documentation](https://pyrtlnet.readthedocs.io/en/latest/index.html) is
  extracted from docstrings with
  [Sphinx](https://www.sphinx-doc.org/en/master/index.html).

### Installation

1. Install [`git`](https://github.com/git-guides/install-git).

1. Clone this repository, and `cd` to the repository's root directory.

   ```shell
   $ git clone https://github.com/UCSBarchlab/pyrtlnet.git
   $ cd pyrtlnet
   ```

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

1. (optional) Install
   [Verilator](https://verilator.org/guide/latest/install.html) if you want to
   export the inference hardware to Verilog, and simulate the Verilog version
   of the hardware.

### Usage

1. Run
   `uv run` [`tensorflow_training.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/tensorflow_training.py)
   in this repository's root directory. This trains a quantized neural network
   with TensorFlow, on the MNIST data set, and produces a quantized `tflite`
   saved model file, named `quantized.tflite`.

   ```shell
   $ uv run tensorflow_training.py
   Training unquantized model.
   Epoch 1/10
   1875/1875 [==============================] - 1s 350us/step - loss: 0.6532 - accuracy: 0.8202
   Epoch 2/10
   1875/1875 [==============================] - 1s 346us/step - loss: 0.3304 - accuracy: 0.9039
   Epoch 3/10
   1875/1875 [==============================] - 1s 347us/step - loss: 0.2944 - accuracy: 0.9145
   Epoch 4/10
   1875/1875 [==============================] - 1s 350us/step - loss: 0.2719 - accuracy: 0.9205
   Epoch 5/10
   1875/1875 [==============================] - 1s 352us/step - loss: 0.2551 - accuracy: 0.9245
   Epoch 6/10
   1875/1875 [==============================] - 1s 348us/step - loss: 0.2403 - accuracy: 0.9288
   Epoch 7/10
   1875/1875 [==============================] - 1s 350us/step - loss: 0.2280 - accuracy: 0.9330
   Epoch 8/10
   1875/1875 [==============================] - 1s 346us/step - loss: 0.2178 - accuracy: 0.9358
   Epoch 9/10
   1875/1875 [==============================] - 1s 348us/step - loss: 0.2092 - accuracy: 0.9378
   Epoch 10/10
   1875/1875 [==============================] - 1s 350us/step - loss: 0.2023 - accuracy: 0.9403
   Evaluating unquantized model.
   313/313 [==============================] - 0s 235us/step - loss: 0.1994 - accuracy: 0.9414
   Training quantized model and writing quantized.tflite and quantized.npz.
   Epoch 1/2
   1875/1875 [==============================] - 1s 410us/step - loss: 0.1963 - accuracy: 0.9426
   Epoch 2/2
   1875/1875 [==============================] - 1s 408us/step - loss: 0.1936 - accuracy: 0.9423
   ...
   Evaluating quantized model.
   313/313 [==============================] - 0s 286us/step - loss: 0.1996 - accuracy: 0.9413
   Writing mnist_test_data.npz.
   ```

   The script's output shows that the unquantized model achieved `0.9414` accuracy on
   the test data set, while the quantized model achieved `0.9413` accuracy on the test
   data set.

   This script produces `quantized.tflite` and `quantized.npz` files which
   includes all the model's weights, biases, and quantization parameters.
   `quantized.tflite` is a standard `.tflite` saved model file that can be read
   by tools like the
   [Model Explorer](https://github.com/google-ai-edge/model-explorer).
   `quantized.npz` stores the weights, biases, and quantization parameters as
   [NumPy saved arrays](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html).
   `quantized.npz` is read by all the provided inference implementations.

1. Run
   `uv run` [`litert_inference.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/litert_inference.py) in this repository's root directory.
   This runs one test image through the reference LiteRT inference implementation.

   ![litert_inference.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/litert_inference.png?raw=true)

   The script outputs many useful pieces of information:

   1. A display of the input image, in this case a picture of the digit `7`. This
      display requires a terminal that supports 24-bit color, like
      [gnome-terminal](https://help.gnome.org/users/gnome-terminal/stable/) or
      [iTerm2](https://iterm2.com/). This is the first image in the test data set
      `(#0)`.

   1. The input shape, `(12, 12)`, and `dtype float32`.

   1. The output from the first layer of the network, with shape `(1, 18)` and `dtype
      int8`.

   1. The output from the second layer of the network, with shape `(1, 10)` and `dtype
      int8`.

   1. A bar chart displaying the network's final output, which is the inferred
      likelihood that the image contains each digit. The network only has two layers, so
      this is the same data from the `layer 1 output` line, reformatted into a bar
      chart.

      In this case, the digit `7` is the most likely, with a score of `93`, followed by
      the digit `3` with a score of `58`. The digit `7` is labeled as `actual` because
      it is the actual prediction generated by the neural network. It is also labeled as
      `expected` because the labled test data confirms that the image actually depicts
      the digit `7`.

   The `litert_inference.py` script also supports a `--start_image` command line flag,
   to run inference on other images from the test data set. There is also a
   `--num_images` flag, which will run several images from the test data set, one at a
   time, and print an accuracy score. All of the provided inference scripts accept these
   command line flags. For example:

   ```shell
   $ uv run litert_inference.py --start_image=7 --num_images=10
   ...
   9/10 correct predictions, 90% accuracy
   ```

1. Run
   `uv run` [`numpy_inference.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/numpy_inference.py) in this repository's root directory.
   This runs one test image through the software NumPy and fxpmath inference
   implementation. This implements inference for the quantized neural network as a
   series of NumPy calls, using the fxpmath fixed-point math library.

   ![numpy_inference.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/numpy_inference.png?raw=true)

   The tensors output by this script should exactly match the tensors output by
   `litert_inference.py`, except that each layer's outputs are transposed.

1. Run
   `uv run`
   [`pyrtl_inference.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_inference.py)
   `--verilog` in this repository's root directory.
   This runs one test image through the hardware PyRTL inference
   implementation. This implementation converts the quantized neural network
   into hardware logic, and simulates the hardware with a PyRTL
   [`Simulation`](https://pyrtl.readthedocs.io/en/latest/simtest.html#pyrtl.simulation.Simulation).

   ![pyrtl_inference.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/pyrtl_inference.png?raw=true)

   The tensors output by this script should exactly match the tensors output by
   `numpy_inference.py`.

   The `--verilog` flag makes `pyrtl_inference.py` generate a Verilog version
   of the hardware, which is written to `pyrtl_inference.v`, and a testbench
   written to `pyrtl_inference_test.v`. The next step will use these generated
   Verilog files.

1. If `verilator` is installed, run `verilator --trace -j 0 --binary pyrtl_inference_test.v`:

   ```shell
   $ verilator --trace -j 0 --binary pyrtl_inference_test.v
   ...
   - V e r i l a t i o n   R e p o r t: Verilator 5.032 2025-01-01 rev (Debian 5.032-1)
   - Verilator: Built from 0.227 MB sources in 3 modules, into 13.052 MB in 18 C++ files needing 0.022 MB
   - Verilator: Walltime 3.576 s (elab=0.014, cvt=0.598, bld=2.857); cpu 0.847 s on 32 threads; alloced 164.512 MB
   ```

   This converts the generated Verilog files to generated C++ code, and
   compiles the generated C++ code. The outputs of this process can be found in
   the `obj_dir` directory.

1. If `verilator` is installed, run `obj_dir/Vpyrtl_inference_test`:

   ```shell
   $ obj_dir/Vpyrtl_inference_test
   ...
   time 1930
   layer1 output (transposed):
   [[  33  -48   29   58  -50   31  -87   93    9   49]]
   argmax: 7

   - pyrtl_inference_test.v:858: Verilog $finish
   - S i m u l a t i o n   R e p o r t: Verilator 5.032 2025-01-01
   - Verilator: $finish at 2ns; walltime 0.005 s; speed 329.491 ns/s
   - Verilator: cpu 0.006 s on 1 threads; alloced 249 MB
   ```

   The final `layer1 output` printed by the Verilator simulation should
   exactly match the `layer1 output` tensors output by `pyrtl_inference.py`.

### Next Steps

See
[`fpga/README.md`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/fpga/README.md)
for instructions on running `pyrtlnet` inference on a
[Pynq Z2 FPGA](https://www.amd.com/en/corporate/university-program/aup-boards/pynq-z2.html).

The [reference
documentation](https://pyrtlnet.readthedocs.io/en/latest/index.html) has more
information on how these scripts work and their main interfaces.

Try the `pyrtl_matrix.py` demo script, with
`uv run` [`pyrtl_matrix.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_matrix.py)
to see how the PyRTL systolic array multiplies matrices. Also see the
documentation for
[`make_systolic_array`](https://pyrtlnet.readthedocs.io/en/latest/matrix.html#pyrtlnet.pyrtl_matrix.make_systolic_array):

![pyrtl_matrix.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/pyrtl_matrix.png?raw=true)

`pyrtl_matrix.py` also supports the `--verilog` flag, so this systolic array
simulation can be repeated with Verilator.

### Project Ideas

* Many TODOs are scattered throughout this code base. If one speaks to you, try
  addressing it! Some notable TODOs:

  * Support input batching, so the various inference systems can process more than one
    image at a time.

  * Extend `WireMatrix2D` to support an arbitrary number of dimensions, not just two.
    Extend the systolic array to support multiplying matrices with more dimensions. This
    is needed to support [convolutional neural
    networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), for example.

  * Add support for [block matrix](https://en.wikipedia.org/wiki/Block_matrix)
    multiplications, so all neural network layers can share one systolic array that
    processes uniformly-sized blocks of inputs at a time. Currently, each layer creates
    its own systolic array that's large enough to process all of its input data, which
    is not very realistic.

  * Support arbitrary neural network architectures. The current implementation
    assumes a model with exactly two layers. Instead, we should discover the
    number of layers, and how they are connected, by analyzing the saved model.

* Add an `inference_util` to collect image input data directly from the user. It would
  be cool to draw a digit with a mouse or touch screen, and see the prediction generated
  by one of the inference implementations.

* Support more advanced neural network architectures, like [convolutional neural
  networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) or
  [transformers](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)).

### Contributing

Contributions are welcome! Please check a few things before sending a pull request:

1. Before attempting a large change, please discuss your plan with maintainers.
   [Open an issue](https://github.com/UCSBarchlab/pyrtlnet/issues) or [start a
   discussion](https://github.com/UCSBarchlab/pyrtlnet/discussions) and
   describe your proposed change.

1. Ensure that all tests pass, and that new features are tested. Tests are run with
   [`pytest`](https://docs.pytest.org/en/stable/):

   ```shell
   $ uv run pytest
   ============================ test session starts ============================
   ...
   collected 20 items

   tests/litert_inference_test.py .                                      [  5%]
   tests/numpy_inference_test.py .                                       [ 10%]
   tests/pyrtl_inference_test.py .                                       [ 15%]
   tests/pyrtl_matrix_test.py ..........                                 [ 65%]
   tests/tensorflow_training_test.py ..                                  [ 75%]
   tests/wire_matrix_2d_test.py .....                                    [100%]

   ============================ 20 passed in 15.75s ============================
   ```

   [`pytest-xdist`](https://github.com/pytest-dev/pytest-xdist) is also installed, so
   testing can be accelerated by running the tests in parallel with `pytest -n auto`.

1. Ensure that [`ruff`](https://docs.astral.sh/ruff/) lint checks pass:

   ```shell
   $ uv run ruff check
   All checks passed!
   ```

1. Apply `ruff` automatic code formatting:

   ```shell
   $ uv run ruff format
   22 files left unchanged
   ```

### Maintenance

`uv` pins all `pip` dependencies to specific versions for reproducible
behavior. These pinned dependencies must be manually updated with
`uv lock --upgrade`.

When a new
[minor version version of Python](https://devguide.python.org/versions/) is
released, update the pinned Python version with `uv python pin $VERSION`, and
the Python version in `.readthedocs.yaml`.
