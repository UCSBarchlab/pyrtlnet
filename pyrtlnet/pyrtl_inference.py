"""
Implement quantized inference with `PyRTL`_.

This does not invoke the :ref:`litert_inference` or :ref:`numpy_inference`
implementations.

This effectively reimplements :ref:`numpy_inference` in hardware, using the
:ref:`matrix`, running in a `PyRTL`_ :class:`~pyrtl.Simulation`.

The `pyrtl_inference demo`_ uses :class:`PyRTLInference` to implement quantized
inference with `PyRTL`_.

.. _pyrtl_inference demo: https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_inference.py
"""

import numpy as np
import pyrtl

import pyrtlnet.pyrtl_matrix as pyrtl_matrix
from pyrtlnet.inference_util import SavedTensors
from pyrtlnet.wire_matrix_2d import WireMatrix2D


class PyRTLInference:
    """
    Convert a quantized model to hardware, and simulate the hardware with a `PyRTL`_
    :class:`~pyrtl.Simulation`.
    """

    def __init__(
        self,
        quantized_model_prefix: str,
        input_bitwidth: int,
        accumulator_bitwidth: int,
    ) -> None:
        """Convert the quantized model to PyRTL inference hardware.

        This builds the necessary hardware for each layer's matrix operations. The
        diagram below shows ``layer0``'s data flow::

            layer0 weight, shape: (18, 144), input_bitwidth
                │
                │   layer0 input image, shape: (144, 1), input_bitwidth
                │       │
                ▼       ▼
            ┌─────────────────────────────────────────────────────────┐
            │ layer0 systolic_array (matrix multiplication by weight) │
            └─────────────────────────────────────────────────────────┘
                │
                │ output, shape: (18, 1), accumulator_bitwidth
                │
                │   layer0 bias, shape: (18, 1), accumulator_bitwidth
                │       │
                ▼       ▼
            ┌───────────────────────────────────┐
            │ layer0 elementwise_add (add bias) │
            └───────────────────────────────────┘
                │
                │ output, shape: (18, 1), accumulator_bitwidth
                ▼
            ┌─────────────────────────┐
            │ layer0 elementwise_relu │
            └─────────────────────────┘
                │
                │ output, shape: (18, 1), accumulator_bitwidth
                ▼
            ┌────────────────────────────────────────────────┐
            │ layer0 elementwise_normalize (reduce bitwidth) │
            └────────────────────────────────────────────────┘
                        │
                        │
                        ▼
                    layer0 output, shape: (18, 1), input_bitwidth

        And this diagram shows the ``layer1``'s data flow, where the ``layer0 output``
        from ``layer0`` is the second input to ``layer1``'s ``systolic_array``::

            layer1 weight, shape: (10, 18), input_bitwidth
                │
                │   layer0 output, shape: (18, 1), input_bitwidth
                │       │
                ▼       ▼
            ┌─────────────────────────────────────────────────────────┐
            │ layer1 systolic_array (matrix multiplication by weight) │
            └─────────────────────────────────────────────────────────┘
                │
                │ output, shape: (10, 1), accumulator_bitwidth
                │
                │   layer1 bias, shape: (10, 1), accumulator_bitwidth
                │       │
                ▼       ▼
            ┌───────────────────────────────────┐
            │ layer1 elementwise_add (add bias) │
            └───────────────────────────────────┘
                │
                │ output, shape: (10, 1), accumulator_bitwidth
                ▼
            ┌────────────────────────────────────────────────┐
            │ layer1 elementwise_normalize (reduce bitwidth) │
            └────────────────────────────────────────────────┘
                │
                │
                ▼
            layer1 output, shape: (10, 1), input_bitwidth


        * :func:`.make_systolic_array` performs the matrix multiplication of each
          layer's weight and input.
        * :func:`.make_elementwise_add` performs the elementwise addition of each
          layer's bias.
        * :func:`.make_elementwise_relu` performs ReLU (only for ``layer0``).
        * :func:`.make_elementwise_normalize` performs normalization to convert from
          intermediate values with bitwidth ``accumulator_bitwidth`` to each layer's
          output values with bitwidth ``input_bitwidth``.

        :param quantized_model_prefix: Prefix of the ``.npz`` file created by
            ``tensorflow_training.py``, without the ``.npz`` suffix.
        :param input_bitwidth: Bitwidth of each element in the input matrix. This should
            generally be ``8``.
        :param accumulator_bitwidth: Bitwidth of accumulator registers in the systolic
            array. This should generally be ``32``, and larger than ``input_bitwidth``.
        """
        self.input_bitwidth = input_bitwidth
        self.accumulator_bitwidth = accumulator_bitwidth

        saved_tensors = SavedTensors(quantized_model_prefix)
        self.input_scale = saved_tensors.input_scale
        self.input_zero = saved_tensors.input_zero
        self.layer = saved_tensors.layer

        # Create the MemBlock for the input image data.
        self._make_input_memblock()

        # Create hardware that implements neural network inference.
        self._make_inference()

    def _make_input_memblock(self) -> None:
        """Build the MemBlock that will hold the input image data."""
        weight_shape = self.layer[0].weight.shape
        batch_size = 1
        flat_image_shape = (weight_shape[1], batch_size)

        done_cycle = (
            pyrtl_matrix.num_systolic_array_cycles(weight_shape, flat_image_shape) - 1
        )
        self.flat_image_addrwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

        # Create a properly-sized empty MemBlock. The MemBlock's contents will be set
        # at simulation time in `simulate()`.
        _, num_columns = flat_image_shape
        self.flat_image_memblock = pyrtl.MemBlock(
            name="flat_image",
            addrwidth=self.flat_image_addrwidth,
            bitwidth=self.input_bitwidth * num_columns,
        )

        # Create a WireMatrix2D that wraps the empty MemBlock. This will be the first
        # layer's input.
        self.flat_image_matrix = WireMatrix2D(
            values=self.flat_image_memblock,
            shape=flat_image_shape,
            bitwidth=self.input_bitwidth,
            name="flat_image",
            valid=True,
        )

    def _make_layer(
        self,
        layer_num: int,
        input: WireMatrix2D,
        input_zero: np.ndarray,
        relu: bool,
    ) -> WireMatrix2D:
        """Build one layer of the PyRTL inference hardware."""
        layer_name = f"layer{layer_num}"

        # Matrix multiplication with 8-bit inputs and 32-bit output. This multiplies
        # the layer's weight and the layer's input data.
        product = pyrtl_matrix.make_systolic_array(
            name=layer_name + "_matmul",
            a=self.layer[layer_num].weight,
            b=input,
            b_zero=input_zero,
            input_bitwidth=self.input_bitwidth,
            accumulator_bitwidth=self.accumulator_bitwidth,
        )

        # Create a WireMatrix2D for the layer's bias.
        bias_matrix = WireMatrix2D(
            values=self.layer[layer_num].bias,
            bitwidth=pyrtl_matrix.minimum_bitwidth(self.layer[layer_num].bias),
            name=layer_name + "_bias",
            valid=True,
        )
        # Add the bias. This is a 32-bit add.
        sum = pyrtl_matrix.make_elementwise_add(
            name=layer_name + "_add",
            a=product,
            b=bias_matrix,
            output_bitwidth=self.accumulator_bitwidth,
        )

        # Perform ReLU, if the layer needs it. This is a 32-bit ReLU.
        if relu:
            relu = pyrtl_matrix.make_elementwise_relu(name=layer_name + "_relu", a=sum)
        else:
            relu = sum

        # Normalize from 32-bit to 8-bit. This effectively multiplies the layer's output
        # by its scale factor `m` and adds its zero point `z3`. `m` is represented as
        # a fixed-point multiplier `m0` and a right shift `n`.
        output = pyrtl_matrix.make_elementwise_normalize(
            name=layer_name,
            a=relu,
            m0=self.layer[layer_num].m0,
            n=self.layer[layer_num].n,
            z3=self.layer[layer_num].zero,
            output_bitwidth=self.input_bitwidth,
        )

        # Create pyrtl.Outputs for the layer's output. These can be inspected with
        # output.inspect().
        output.make_outputs(layer_name)

        return output

    def _make_inference(self) -> None:
        """Build the PyRTL inference hardware."""
        # Build all layers.
        layer0 = self._make_layer(
            layer_num=0,
            input=self.flat_image_matrix,
            input_zero=self.input_zero,
            relu=True,
        )

        layer1 = self._make_layer(
            layer_num=1, input=layer0, input_zero=self.layer[0].zero, relu=False
        )

        self.layer_outputs = [layer0, layer1]

        # Compute argmax for the last layer's output.
        argmax = pyrtl_matrix.make_argmax(a=layer1)

        num_rows, num_columns = layer1.shape
        assert num_columns == 1

        argmax_output = pyrtl.Output(
            name="argmax", bitwidth=pyrtl.infer_val_and_bitwidth(num_rows).bitwidth
        )
        argmax_output <<= argmax

        # Make a PyRTL Output for the second layer output's `valid` signal. When this
        # signal goes high, inference is complete.
        valid = pyrtl.Output(name="valid", bitwidth=1)
        valid <<= layer1.valid

    def simulate(
        self, test_image: np.ndarray, verilog: bool = False
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Simulate quantized inference on a single image.

        All calculations are done in PyRTL :class:`~pyrtl.Simulation`, using the
        hardware generated by :meth:`__init__`.

        :param test_image: An image to run through the NumPy inference implementation.
        :param verilog: If ``True``, export the inference implementation to Verilog. The
            generated Verilog file will be called ``pyrtl_inference.v``.

        :returns: ``(layer0_output, layer1_output, predicted_digit)``, where
                  ``layer0_output`` is the first layer's raw tensor output, with shape
                  ``(18, 1)``. ``layer1_output`` is the second layer's raw tensor
                  output, with shape ``(10, 1)``. Note that these layer outputs are
                  transposed compared to :func:`.run_tflite_model`. ``predicted_digit``
                  is the actual predicted digit. ``predicted_digit`` is equivalent to
                  ``layer1_output.flatten().argmax()``.
        """
        layer0_weight_shape = self.layer[0].weight.shape
        batch_size = 1
        input_shape = (layer0_weight_shape[1], batch_size)

        # The MNIST image data contains pixel values in the range [0, 255]. The neural
        # network was trained by first converting these values to floating point, in the
        # range [0, 1.0]. Dividing by input_scale below undoes this conversion,
        # converting the range from [0, 1.0] back to [0, 255].
        #
        # We could avoid these back-and-forth conversions by modifying
        # `load_mnist_images()` to skip the first conversion, and returning `x +
        # input_zero_point` below to skip the second conversion, but we do them anyway
        # to simplify the code and make it more consistent with existing sample code
        # like https://ai.google.dev/edge/litert/models/post_training_integer_quant
        #
        # Adding input_zero_point (-128) effectively converts the uint8 image data to
        # int8, by shifting the range [0, 255] to [-128, 127].
        flat_image = np.reshape(
            test_image / self.input_scale + self.input_zero, newshape=input_shape
        ).astype(np.int8)

        # Convert the flattened image data to a dictionary for use in Simulation's
        # `memory_value_map`. The `flat_image` is transposed because this data will be
        # the second input to the first layer's systolic array (`top` inputs to the
        # array).
        memblock_data = pyrtl_matrix.make_input_memblock_data(
            flat_image.transpose(),
            self.input_bitwidth,
            self.flat_image_addrwidth,
        )
        memblock_data_dict = dict(enumerate(memblock_data))

        # Run until the second layer's computations are done.
        sim = pyrtl.FastSimulation(
            memory_value_map={self.flat_image_memblock: memblock_data_dict}
        )
        done = False
        while not done:
            sim.step()
            done = sim.inspect("valid")

        # Retrieve each layer's outputs.
        layer0_output = self.layer_outputs[0].inspect(sim=sim).astype(np.int8)
        layer1_output = self.layer_outputs[1].inspect(sim=sim).astype(np.int8)

        # Retrieve the predicted digit.
        argmax = sim.inspect("argmax")

        if verilog:
            with open("pyrtl_inference.v", "w") as output:
                pyrtl.output_to_verilog(output)
                pyrtl.output_verilog_testbench(
                    output,
                    simulation_trace=sim.tracer,
                    vcd="pyrtl_inference.vcd",
                    cmd=(
                        '$display("time %3t\\n'
                        "layer1 output (transposed):\\n"
                        "[[%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d]]\\n"
                        'argmax: %1d\\n", '
                        "$time,"
                        "$signed(layer1_0_0), $signed(layer1_1_0), "
                        "$signed(layer1_2_0), $signed(layer1_3_0), "
                        "$signed(layer1_4_0), $signed(layer1_5_0), "
                        "$signed(layer1_6_0), $signed(layer1_7_0), "
                        "$signed(layer1_8_0), $signed(layer1_9_0), "
                        "argmax);"
                    ),
                )
        return layer0_output, layer1_output, argmax
