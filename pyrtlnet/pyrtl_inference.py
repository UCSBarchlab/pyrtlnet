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

import pathlib

import numpy as np
import pyrtl

import pyrtlnet.pyrtl_axi as pyrtl_axi
import pyrtlnet.pyrtl_matrix as pyrtl_matrix
from pyrtlnet.constants import quantized_model_prefix
from pyrtlnet.inference_util import preprocess_image
from pyrtlnet.saved_tensors import SavedTensors
from pyrtlnet.wire_matrix_2d import WireMatrix2D


class PyRTLInference:
    """
    Convert a quantized model to hardware, and simulate the hardware with a `PyRTL`_
    :class:`~pyrtl.Simulation`.
    """

    def __init__(
        self,
        tensor_path: str,
        input_bitwidth: int,
        accumulator_bitwidth: int,
        axi: bool,
        initial_delay_cycles: int = 0,
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

        - :func:`.make_systolic_array` performs the matrix multiplication of each
          layer's weight and input.

        - :func:`.make_elementwise_add` performs the elementwise addition of each
          layer's bias.

        - :func:`.make_elementwise_relu` performs ReLU (only for ``layer0``).

        - :func:`.make_elementwise_normalize` performs normalization to convert from
          intermediate values with bitwidth ``accumulator_bitwidth`` to each layer's
          output values with bitwidth ``input_bitwidth``.

        :param tensor_path: Path to the ``.npz`` file created by
            :func:`.quantize_model`.
        :param input_bitwidth: Bitwidth of each element in the input matrix. This should
            generally be ``8``.
        :param accumulator_bitwidth: Bitwidth of accumulator registers in the systolic
            array. This should generally be ``32``, and larger than ``input_bitwidth``.
        :param axi: If ``True``, receive input image data via an AXI-Stream, and return
            the output's ``argmax`` via AXI-Lite, at address 0. If ``False``, the input
            image data will be loaded in ``self.flat_image_memblock`` via
            :class:`~pyrtl.Simulation`'s ``memory_value_map``, and the output's
            ``argmax`` will be inspected as an :class:`~pyrtl.Output`.
        :param initial_delay_cycles: Number of cycles to wait before starting operation.
            This is a temporary hack that's currently required for correct synthesis
            with Vivado. No delay cycles should be required.
        """
        self.input_bitwidth = input_bitwidth
        self.accumulator_bitwidth = accumulator_bitwidth
        self.axi = axi
        self.initial_delay_cycles = initial_delay_cycles

        tensor_file = pathlib.Path(tensor_path) / f"{quantized_model_prefix}.npz"
        if not tensor_file.exists():
            msg = f"{tensor_file} not found. Run tensorflow_training.py first."
            raise FileNotFoundError(msg)

        saved_tensors = SavedTensors(tensor_file)
        self.input_scale = saved_tensors.input_scale
        self.input_zero = saved_tensors.input_zero
        self.layer = saved_tensors.layer

        # Create the MemBlock for the input image data.
        self._make_input_memblock()

        # Create hardware that implements neural network inference.
        self._make_inference()

        if self.axi:
            # Create hardware to load the MemBlock from an AXI-Stream.
            self.flat_image_matrix.valid <<= pyrtl_axi.make_axi_stream_subordinate(
                mem=self.flat_image_memblock
            )
        else:
            # Otherwise, the MemBlock will be loaded by `Simulation`'s
            # `memory_value_map`.
            self.flat_image_matrix.valid <<= True

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
            initial_delay_cycles=self.initial_delay_cycles,
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
        if not self.axi:
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

        if self.axi:
            # Make an AXI-Lite subordinate. Register map:
            #
            #      Register 0: argmax
            #  Registers 1-18: layer 0 output
            # Registers 19-28: layer 1 output
            num_registers = (
                1 + self.layer_outputs[0].shape[0] + self.layer_outputs[1].shape[0]
            )
            registers = pyrtl_axi.make_axi_lite_subordinate(
                num_registers=num_registers, num_writable_registers=0
            )
            registers[0].next <<= argmax
            for row in range(self.layer_outputs[0].shape[0]):
                registers[1 + row].next <<= self.layer_outputs[0][row][0]
            for row in range(self.layer_outputs[1].shape[0]):
                registers[
                    1 + self.layer_outputs[0].shape[0] + row
                ].next <<= self.layer_outputs[1][row][0]

    def _memblock_data(self, test_image: np.ndarray) -> list[int]:
        """Convert ``test_image`` to loadable data for :attr:`flat_image_memblock`.

        Each ``test_image`` is 12×12, with 8-bit pixels (see
        :func:`.load_mnist_images`). The pixel data will be linearized into a 144-entry
        list, and that list will be padded out to the next largest power of 2, which is
        256. So the returned list will have 256 entries, and each entry will fit in 8
        bits.

        :param test_image: A resized MNIST image to convert to MemBlock data.
        :returns: Image data that can be loaded into :attr:`flat_image_memblock`.
        """
        flat_image = preprocess_image(test_image, self.input_scale, self.input_zero)

        # Convert the flattened image data to a dictionary for use in Simulation's
        # `memory_value_map`. The `flat_image` is transposed because this data will be
        # the second input to the first layer's systolic array (`top` inputs to the
        # array).
        data = pyrtl_matrix.make_input_memblock_data(
            flat_image.transpose(),
            self.input_bitwidth,
            self.flat_image_addrwidth,
        )

        # Verify that the memblock data will actually fit in the memblock.
        assert len(data) == 2**self.flat_image_memblock.addrwidth
        for pixel in data:
            pixel_bitwidth = pyrtl.infer_val_and_bitwidth(pixel).bitwidth
            assert pixel_bitwidth <= self.flat_image_memblock.bitwidth

        return data

    def simulate(
        self, test_image: np.ndarray, verilog: bool = False
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Simulate quantized inference on a single image.

        All calculations are done in PyRTL :class:`~pyrtl.Simulation`, using the
        hardware generated by :meth:`__init__`.

        :param test_image: An image to run through the NumPy inference implementation.
        :param verilog: If ``True``, export the inference implementation to Verilog. Two
            Verilog files will be generated, one for the ``pyrtlnet`` module itself, and
            another for its testbench. The ``pyrtlnet`` module will be named
            ``pyrtl_inference.v``, or ``pyrtl_inference_axi.v`` when constructed with
            ``axi=True``. The testbench will be named ``pyrtl_inference_test.v``, or
            ``pyrtl_inference_axi_test.v`` when constructed with ``axi=True``.

        :returns: ``(layer0_output, layer1_output, predicted_digit)``, where
                  ``layer0_output`` is the first layer's raw tensor output, with shape
                  ``(1, 18)``. ``layer1_output`` is the second layer's raw tensor
                  output, with shape ``(1, 10)``. ``predicted_digit`` is the actual
                  predicted digit. ``predicted_digit`` is equivalent to
                  ``layer1_output.flatten().argmax()``.
        """
        memblock_data = self._memblock_data(test_image)

        # TODO: The `pyrtlnet` hardware can currently only process one image, so we have
        # to make a new `FastSimulation` for each image. Update the
        # `axi_stream_subordinate` and `systolic_array` state machines so they reset
        # after processing one image.
        if self.axi:
            sim = pyrtl.FastSimulation()

            # `provided_inputs` holds default values for all Inputs. `aresetn` is active
            # low.
            provided_inputs = {
                # AXI-Stream Inputs.
                "s0_axis_aclk": False,
                "s0_axis_aresetn": True,
                "s0_axis_tdata": 0,
                "s0_axis_tvalid": False,
                "s0_axis_tlast": False,
                # AXI-Lite Inputs.
                "s0_axi_clk": False,
                "s0_axi_aresetn": True,
                "s0_axi_araddr": 0,
                "s0_axi_arvalid": False,
                "s0_axi_rready": False,
                "s0_axi_awaddr": 0,
                "s0_axi_awvalid": False,
                "s0_axi_wdata": 0,
                "s0_axi_wvalid": False,
                "s0_axi_wstrb": 0,
                "s0_axi_bready": False,
            }

            # Transmit the memblock_data via AXI-Stream.
            pyrtl_axi.simulate_axi_stream_send(
                sim, provided_inputs, stream_data=memblock_data
            )
        else:
            memblock_data_dict = dict(enumerate(memblock_data))
            sim = pyrtl.FastSimulation(
                memory_value_map={self.flat_image_memblock: memblock_data_dict}
            )
            provided_inputs = {}

        # Wait until the second layer's computations are done.
        done = False
        while not done:
            sim.step(provided_inputs)
            done = sim.inspect("valid")

        # Retrieve each layer's outputs and the predicted digit.
        if self.axi:

            def retrieve_layer_outputs(start: int, end: int) -> list[int]:
                """Retrieve a layer's outputs via AXI-Lite.

                Each layer output is a signed 8-bit value, stored in a 32-bit AXI
                register.
                """
                outputs = []
                for addr in range(start, end, 4):
                    outputs.append(
                        [
                            pyrtl.val_to_signed_integer(
                                pyrtl_axi.simulate_axi_lite_read(
                                    sim, provided_inputs, address=addr
                                ),
                                bitwidth=8,
                            )
                        ]
                    )
                return np.array(outputs, dtype=np.int8)

            # Registers 1-18 hold the layer0's outputs, and registers 19-28 hold
            # layer1's outputs. Each register is 32-bits wide, and AXI addresses are
            # byte addresses.
            layer0_output = retrieve_layer_outputs(start=1 * 4, end=19 * 4)
            layer1_output = retrieve_layer_outputs(start=19 * 4, end=29 * 4)

            # Read the argmax via AXI-Lite. The sum is stored in AXI register 0.
            argmax = pyrtl_axi.simulate_axi_lite_read(sim, provided_inputs, address=0)
        else:
            layer0_output = self.layer_outputs[0].inspect(sim=sim).astype(np.int8)
            layer1_output = self.layer_outputs[1].inspect(sim=sim).astype(np.int8)

            argmax = sim.inspect("argmax")

        if verilog:
            suffix = ""
            if self.axi:
                suffix = "_axi"
            module_file_name = f"pyrtl_inference{suffix}.v"
            with open(module_file_name, "w") as output:
                pyrtl.output_to_verilog(output, add_reset=False)
            with open(f"pyrtl_inference{suffix}_test.v", "w") as output:
                pyrtl.output_verilog_testbench(
                    output,
                    simulation_trace=sim.tracer,
                    toplevel_include=module_file_name,
                    vcd=f"pyrtl_inference{suffix}.vcd",
                    cmd=(
                        '$display("time %3t\\n'
                        "layer1 output:\\n"
                        "[%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d]\\n"
                        'argmax: %1d\\n", '
                        "$time,"
                        "$signed(layer1_0_0), $signed(layer1_1_0), "
                        "$signed(layer1_2_0), $signed(layer1_3_0), "
                        "$signed(layer1_4_0), $signed(layer1_5_0), "
                        "$signed(layer1_6_0), $signed(layer1_7_0), "
                        "$signed(layer1_8_0), $signed(layer1_9_0), "
                        "argmax);"
                    ),
                    add_reset=False,
                )

        # Verify that all mem reads and writes are asynchronous. For each memory
        # operation, all of the operation's inputs must be constants or registers, and
        # all of the operation's outputs must be registers.
        gate_graph = pyrtl.GateGraph()
        for mem_read in gate_graph.mem_reads:
            for arg in mem_read.args:
                assert arg.op in "Cr", f"ERROR: async read arg {mem_read}"
            for dest in mem_read.dests:
                assert dest.op == "r", f"ERROR: async read dest {mem_read}"
        for mem_write in gate_graph.mem_writes:
            for arg in mem_write.args:
                assert arg.op in "Cr", f"ERROR: async write arg {mem_write}"

        return layer0_output.transpose(), layer1_output.transpose(), argmax
