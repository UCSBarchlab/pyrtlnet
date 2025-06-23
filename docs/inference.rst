.. _inference:

Inference Implementations
=========================

Three inference implementations are included:

* :ref:`litert_inference`, a reference implementation based on the `LiteRT`_
  ``Interpreter``, in ``litert_inference.py``.
* :ref:`numpy_inference`, a software re-implementation in `NumPy`_ and `fxpmath`_, in
  ``numpy_inference.py``. This implementation is useful for understanding quantization
  math before moving on to a hardware implementation.
* :ref:`pyrtl_inference`, a hardware implementation in `PyRTL`_, in
  ``pyrtl_inference.py``.

.. _LiteRT: https://ai.google.dev/edge/litert
.. _NumPy: https://numpy.org/
.. _fxpmath: https://github.com/francof2a/fxpmath
.. _PyRTL: https://github.com/UCSBarchlab/PyRTL

All three implementations produce the same output.

.. _litert_inference:

LiteRT Inference
----------------

.. automodule:: pyrtlnet.litert_inference
    :members:


.. _numpy_inference:

NumPy Inference
---------------

.. automodule:: pyrtlnet.numpy_inference
    :members:
    :special-members: __init__

.. _pyrtl_inference:

PyRTL Inference
---------------

.. automodule:: pyrtlnet.pyrtl_inference
    :members:
    :special-members: __init__

Inference Utilities
-------------------

.. automodule:: pyrtlnet.inference_util
    :members:
