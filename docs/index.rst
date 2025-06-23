#########################
`pyrtlnet`_ Documentation
#########################

`pyrtlnet`_ is a hardware implementation of quantized dense neural network
inference in the `PyRTL`_ hardware description language.

This is the detailed reference documentation for the `pyrtlnet`_ code, useful
when using or modifying the code.

If you're just getting started, try the installation instructions and tutorial
in `README.md`_.

.. _README.md: https://github.com/UCSBarchlab/pyrtlnet/blob/main/README.md

This reference documentation is split into three sections:

* :ref:`training`, which covers the `TensorFlow`_ training functions.
* :ref:`inference`, which covers the three included inference implementations:

  1. :ref:`litert_inference`, which uses the reference `LiteRT`_
     ``Interpreter`` implementation.
  2. :ref:`numpy_inference`, a software re-implementation in `NumPy`_ and
     `fxpmath`_.
  3. :ref:`pyrtl_inference`, a hardware re-implementation in `PyRTL`_.

* :ref:`matrix`, which covers the included PyRTL linear algebra library.

Contents
========

.. toctree::
   :maxdepth: 2

   training
   inference
   matrix

.. _fxpmath: https://github.com/francof2a/fxpmath
.. _LiteRT: https://ai.google.dev/edge/litert
.. _NumPy: https://numpy.org/
.. _PyRTL: https://github.com/UCSBarchlab/PyRTL
.. _pyrtlnet: https://github.com/UCSBarchlab/pyrtlnet
.. _TensorFlow: https://www.tensorflow.org/
