#########################
`pyrtlnet`_ Documentation
#########################

`pyrtlnet`_ is a hardware implementation of quantized dense neural network
inference in the `PyRTL`_ hardware description language.

.. _pyrtlnet: https://github.com/UCSBarchlab/pyrtlnet
.. _PyRTL: https://github.com/UCSBarchlab/PyRTL

This is the detailed reference documentation for the `pyrtlnet`_ code, useful
when using, extending, or modifying the code.

If you're just getting started, try the installation instructions and tutorial in
`README.md`_.

.. _README.md: https://github.com/UCSBarchlab/pyrtlnet/blob/main/README.md

This reference documentation is split into three sections:

* :ref:`training`, which covers the `TensorFlow`_ training functions.
* :ref:`inference`, which covers the three included inference implementations:

  1. :ref:`litert_inference`
  2. :ref:`numpy_inference`
  3. :ref:`pyrtl_inference`

* :ref:`matrix`, which covers the included PyRTL linear algebra library.

.. _TensorFlow: https://www.tensorflow.org/

Contents
========

.. toctree::
   :maxdepth: 2

   training
   inference
   matrix
