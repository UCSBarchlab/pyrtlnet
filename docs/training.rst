.. _training:

Training Implementation
=======================

Training uses `TensorFlow`_ to produce a quantized ``tflite`` file, which can be loaded by :ref:`litert_inference`, :ref:`numpy_inference`, or :ref:`pyrtl_inference`.

.. _TensorFlow: https://www.tensorflow.org/

TensorFlow Training
-------------------

.. automodule:: pyrtlnet.tensorflow_training
    :members:

MNIST Utilities
---------------

.. automodule:: pyrtlnet.mnist_util
    :members:
