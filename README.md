`pyrtlnet`
==========

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

* Three different quantized inference implementations operating at different levels of
  abstraction. All three implementations produce the same output in the same format and,
  in doing so, provide a useful framework to extend either from the top-down or the
  bottom-up.

  1. A reference quantized inference implementation, using the standard
     [LiteRT](https://ai.google.dev/edge/litert) `Interpreter`.

  2. A software implementation of quantized inference, using [NumPy](https://numpy.org)
     and [fxpmath](https://github.com/francof2a/fxpmath), to verify the math performed
     by the reference implementation.

  3. A [PyRTL](https://github.com/UCSBarchlab/PyRTL) hardware implementation of
     quantized inference that is simulated right at the logic gate level.

* A new [PyRTL](https://github.com/UCSBarchlab/PyRTL) linear algebra library, including
  a composable `WireVector2D` matrix abstraction and a output-stationary [systolic
  array](https://en.wikipedia.org/wiki/Systolic_array) for matrix multiplication.

* An extensive [suite of unit
  tests](https://github.com/UCSBarchlab/pyrtlnet/tree/main/tests), and [continuous
  integration testing](https://github.com/UCSBarchlab/pyrtlnet/actions).

* Understandable and documented code! `pyrtlnet` is designed to be, first and foremost,
  understandable and readable (even when that comes at the expense of performance).
  Reference documentation is extracted from docstrings with
  [Sphinx](https://www.sphinx-doc.org/en/master/index.html).

### Installation

1. Install Python 3.12. `pyrtlnet` is only tested with Python 3.12. Note that TensorFlow
   currently does not support Python 3.13.

   ```shell
   $ python --version
   Python 3.12.8
   ```

2. Create a [venv](https://docs.python.org/3/library/venv.html) for `pyrtlnet`.
   `pyrtlnet` depends on many `pip` packages, pinned to specific versions for
   reproducible behavior. Installation of these packages should be done in a clean
   `venv` to avoid conflicts with installed system packages.

   ```shell
   $ python -m venv pyrtlnet-venv
   $ . pyrtlnet-venv/bin/activate
   (pyrtlnet-venv) $ pip list
   Package Version
   ------- -------
   pip     24.3.1
   (pyrtlnet-venv) $
   ```

3. Install `pip` packages.

   ```shell
   (pyrtlnet-venv) $ pip install --upgrade -r requirements.txt
   ...
   Successfully installed absl-py-1.4.0 ai-edge-litert-1.3.0 astunparse-1.6.3 attrs-25.3.0 backports-strenum-1.2.8 certifi-2025.6.15 charset-normalizer-3.4.2 dm-tree-0.1.9 execnet-2.1.1 flatbuffers-25.2.10 fxpmath-0.4.9 gast-0.6.0 google-pasta-0.2.0 grpcio-1.73.0 h5py-3.14.0 idna-3.10 iniconfig-2.1.0 keras-3.10.0 libclang-18.1.1 markdown-3.8 markdown-it-py-3.0.0 markupsafe-3.0.2 mdurl-0.1.2 ml-dtypes-0.5.1 namex-0.1.0 numpy-1.26.4 opt-einsum-3.4.0 optree-0.16.0 packaging-25.0 pluggy-1.6.0 protobuf-5.29.5 pygments-2.19.1 pyrtl-0.11.3 pytest-8.4.0 pytest-xdist-3.7.0 requests-2.32.4 rich-14.0.0 ruff-0.11.13 setuptools-80.9.0 six-1.17.0 tensorboard-2.19.0 tensorboard-data-server-0.7.2 tensorflow-2.19.0 tensorflow-model-optimization-0.8.0 termcolor-3.1.0 tf-keras-2.19.0 tqdm-4.67.1 typing-extensions-4.14.0 urllib3-2.4.0 werkzeug-3.1.3 wheel-0.45.1 wrapt-1.17.2
   (pyrtlnet-venv) $
   ```

### Usage

1. Run
   [`tensorflow_training.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/tensorflow_training.py).
   This trains a quantized neural network with TensorFlow, on the MNIST data set, and
   produces a quantized `tflite` saved model file, named `quantized.tflite`.

   ```shell
   (pyrtlnet-venv) $ python tensorflow_training.py
   Training unquantized model.
   Epoch 1/10
   1875/1875 [==============================] - 1s 395us/step - loss: 0.6666 - accuracy: 0.8176
   Epoch 2/10
   1875/1875 [==============================] - 1s 391us/step - loss: 0.3291 - accuracy: 0.9059
   Epoch 3/10
   1875/1875 [==============================] - 1s 395us/step - loss: 0.2955 - accuracy: 0.9146
   Epoch 4/10
   1875/1875 [==============================] - 1s 389us/step - loss: 0.2769 - accuracy: 0.9193
   Epoch 5/10
   1875/1875 [==============================] - 1s 394us/step - loss: 0.2622 - accuracy: 0.9225
   Epoch 6/10
   1875/1875 [==============================] - 1s 392us/step - loss: 0.2510 - accuracy: 0.9255
   Epoch 7/10
   1875/1875 [==============================] - 1s 399us/step - loss: 0.2406 - accuracy: 0.9293
   Epoch 8/10
   1875/1875 [==============================] - 1s 400us/step - loss: 0.2313 - accuracy: 0.9316
   Epoch 9/10
   1875/1875 [==============================] - 1s 390us/step - loss: 0.2231 - accuracy: 0.9331
   Epoch 10/10
   1875/1875 [==============================] - 1s 396us/step - loss: 0.2157 - accuracy: 0.9356
   Evaluating unquantized model.
   313/313 [==============================] - 0s 288us/step - loss: 0.2131 - accuracy: 0.9373
   Training quantized model and writing quantized.tflite.
   Epoch 1/2
   1875/1875 [==============================] - 1s 457us/step - loss: 0.2135 - accuracy: 0.9359
   Epoch 2/2
   1875/1875 [==============================] - 1s 463us/step - loss: 0.2118 - accuracy: 0.9359
   Evaluating quantized model.
   313/313 [==============================] - 0s 361us/step - loss: 0.2141 - accuracy: 0.9364
   (pyrtlnet-venv) $ ls -l quantized.tflite
   -rw-rw-r-- 1 lauj lauj 5568 Jun 17 19:58 quantized.tflite
   ```

   The script's output shows that the unquantized model achieved `0.9373` accuracy on
   the test data set, while the quantized model achieved `0.9364` accuracy on the test
   data set. `quantized.tflite` includes all the model's weights, biases, and
   quantization parameters. This file will be read by all the inference implementations.

2. Run
   [`litert_inference.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/litert_inference.py).
   This runs one test image through the reference LiteRT inference implementation.

   ![litert_inference.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/litert_inference.png?raw=true)

   The script outputs many useful pieces of information:

   1. A display of the input image, in this case a picture of the digit `7`. This
      display requires a terminal that supports 24-bit color, like
      [gnome-terminal](https://help.gnome.org/users/gnome-terminal/stable/) or
      [iTerm2](https://iterm2.com/). This is the first image in the test data set
      `(#0)`.

   2. The input shape, `(12, 12)`, and `dtype float32`.

   3. The output from the first layer of the network, with shape `(1, 18)` and `dtype
      int8`.

   4. The output from the second layer of the network, with shape `(1, 10)` and `dtype
      int8`.

   5. A bar chart displaying the network's final output, which is the inferred
      likelihood that the image contains each digit. The network only has two layers, so
      this is the same data from the `layer 1 output` line, reformatted into a bar
      chart.

      In this case, the digit `7` is the most likely, with a score of `95`, followed by
      the digit `3` with a score of `51`. The digit `7` is labeled as `actual` because
      it is the actual prediction generated by the neural network. It is also labeled as
      `expected` because the labled test data confirms that the image actually depicts
      the digit `7`.

   The `litert_inference.py` script also supports a `--start_image` command line flag,
   to run inference on other images from the test data set. There is also a
   `--num_images` flag, which will run several images from the test data set, one at a
   time, and print an accuracy score. All of the provided inference scripts accept these
   command line flags.

3. Run
   [`numpy_inference.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/numpy_inference.py).
   This runs one test image through the software NumPy and fxpmath inference
   implementation. This implements inference for the quantized neural network as a
   series of NumPy calls, using the fxpmath fixed-point math library.

   ![numpy_inference.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/numpy_inference.png?raw=true)

   The tensors output by this script should exactly match the tensors output by
   `litert_inference.py`, except that each layer's outputs are transposed.

4. Run
   [`pyrtl_inference.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_inference.py).
   This runs one test image through the hardware PyRTL inference
   implementation. This implementation converts the quantized neural network
   into hardware logic, and simulates the hardware with a PyRTL
   [`Simulation`](https://pyrtl.readthedocs.io/en/latest/simtest.html#pyrtl.simulation.Simulation).

   ![pyrtl_inference.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/pyrtl_inference.png?raw=true)

   The tensors output by this script should exactly match the tensors output by
   `numpy_inference.py`.

### Next Steps

The reference documentation has more information on how these scripts work and
their main interfaces.

Try the demo script
[`pyrtl_matrix.py`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_matrix.py)
to see how the PyRTL systolic array multiplies matrices:

![pyrtl_matrix.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/pyrtl_matrix.png?raw=true)

### Project Ideas

* Many TODOs are scattered throughout this code base. If one speaks to you, try
  addressing it! Some notable TODOs:

  * Support input batching, so the various inference systems can process more than one
    image at a time.

  * Extend `WireMatrix2D` to support an arbitrary number of dimensions, not just two.
    Extend the systolic array to support multiplying matrices with more dimensions. This
    is needed to support convolutional neural networks, for example.

  * Add support for tiled matrix multiplications, so we can use a smaller systolic array
    that processes part of the input at a time. Currently, each matrix multiplication
    creates a systolic array large enough to process all the input data.

* Add an `inference_util` to collect image input data directly from the user. It would
  be cool to draw a digit with a mouse or touch screen, and see the prediction generated
  by one of the inference implementations.

* Add FPGA suppport:

  * Export the PyRTL design to [Verilog](https://en.wikipedia.org/wiki/Verilog), with
    PyRTL's
    [`output_to_verilog()`](https://pyrtl.readthedocs.io/en/latest/export.html#pyrtl.importexport.output_to_verilog).
    Simulate the exported design with a standard Verilog simulator like
    [Verilator](https://github.com/verilator/verilator).

  * Synthesize the exported design and run it on a FPGA.

* Support more advanced neural network architectures, like ResNet or Transformers.

### Contributing

Contributions are welcome! Please check a few things before sending a pull request:

1. Ensure that all tests pass, and that new features are tested. Tests are run with
   [`pytest`](https://docs.pytest.org/en/stable/), which is installed by
   `requirements.txt`:

   ```shell
   (pyrtlnet-venv) $ pytest
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

2. Ensure that [`ruff`](https://docs.astral.sh/ruff/) lint checks pass. `ruff` is
   installed by `requirements.txt`:

   ```shell
   (pyrtlnet-venv) $ ruff check
   All checks passed!
   ```

3. Apply `ruff` automatic code formatting:

   ```shell
   (pyrtlnet-venv) $ ruff format
   22 files left unchanged
   ```

### Maintenance

Periodically update the pinned dependencies in
[`requirements.txt`](https://github.com/UCSBarchlab/pyrtlnet/blob/main/requirements.txt)
with `make requirements.txt`.

When a new version of Python is released, update the [GitHub testing
workflow](https://github.com/UCSBarchlab/pyrtlnet/blob/main/.github/workflows/python-test.yml)
and the installation instructions in `README.md`.
