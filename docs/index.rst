.. meta::
   :google-site-verification: sO_rsKD1QKb6nFywsuLnRDiz8Ekep-jVNpBDMm65wQc

#########################
`pyrtlnet`_ Documentation
#########################

`pyrtlnet`_ is a hardware implementation of quantized dense neural network
inference in the `PyRTL`_ hardware description language.

Let's unpack some of those keywords:

Neural Network
    A `neural network`_ is really just a very large math equation that happens
    to be good at making predictions for a particular problem. `pyrtlnet`_ uses
    the `MNIST`_ data set, which consists of images of hand-written digits. In
    our case, the problem is determining which digit is shown in each image.

Inference
    `Inference` means we're just using a `neural network` to make predictions, as
    opposed to `training`, which figures out how to adjust the neural network's
    equation so it makes better predictions.

Quantized
    `pyrtlnet`_ is fully `quantized`, which means it only does math with
    integers. No floating-point math is required. Integer operations are easier
    to implement in hardware, and `PyRTL`_ conveniently `implements signed
    integer arithmetic`_.

Dense
    `pyrtlnet`_ is a `dense`_ neural network, which means all of its layers are
    fully-connected, and all the linear algebra involves dense (as opposed to
    sparse) `matrices`_. Dense matrices look like your standard matrices from
    high school math, for example::

        ┌           ┐
        │  1  2  3  │
        │ -8  6  7  │
        └           ┘

Hardware Description Language (HDL)
    Most programming languages describe software, but a `hardware description
    language`_ describes hardware. `Verilog`_ is a commonly used hardware
    description language. `pyrtlnet`_ uses the `PyRTL`_ hardware description
    language, which is `Python`_-based.

.. _neural network: https://en.wikipedia.org/wiki/Neural_network
.. _MNIST: https://en.wikipedia.org/wiki/MNIST_database
.. _dense: https://keras.io/api/layers/core_layers/dense/
.. _matrices: https://en.wikipedia.org/wiki/Matrix_(mathematics)
.. _implements signed integer arithmetic: https://pyrtl.readthedocs.io/en/latest/helpers.html#extended-logic-and-arithmetic
.. _hardware description language: https://en.wikipedia.org/wiki/Hardware_description_language
.. _Verilog: https://en.wikipedia.org/wiki/Verilog
.. _Python: https://www.python.org

This is the detailed reference documentation for the `pyrtlnet`_ code, useful
when using or modifying the code. If you're just getting started, try the
installation instructions and tutorial in `README.md`_.

.. _README.md: https://github.com/UCSBarchlab/pyrtlnet/blob/main/README.md

This reference documentation is split into five sections:

* :ref:`training`, which covers the `TensorFlow`_ training functions.
* :ref:`litert_inference`, which uses the reference `LiteRT`_
  ``Interpreter`` implementation.
* :ref:`numpy_inference`, a software re-implementation in `NumPy`_ and
  `fxpmath`_.
* :ref:`pyrtl_inference`, a hardware re-implementation in `PyRTL`_.
* :ref:`matrix`, which covers the included PyRTL linear algebra library.

Contents
========

.. toctree::
   :maxdepth: 2

   training
   litert_inference
   numpy_inference
   pyrtl_inference
   matrix

.. _fxpmath: https://github.com/francof2a/fxpmath
.. _LiteRT: https://ai.google.dev/edge/litert
.. _NumPy: https://numpy.org/
.. _PyRTL: https://github.com/UCSBarchlab/PyRTL
.. _pyrtlnet: https://github.com/UCSBarchlab/pyrtlnet
.. _TensorFlow: https://www.tensorflow.org/
