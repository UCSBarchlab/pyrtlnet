import unittest

import numpy as np
import pyrtl
import pytest
from fxpmath import Fxp

import pyrtlnet.numpy_inference as numpy_inference
import pyrtlnet.pyrtl_matrix as pyrtl_matrix
from pyrtlnet.wire_matrix_2d import WireMatrix2D


class TestPyrtlMatrix(unittest.TestCase):
    def setUp(self) -> None:
        pyrtl.reset_working_block()

    def make_wire_matrix_2d(
        self, name: str, array: np.ndarray, bitwidth: int
    ) -> WireMatrix2D:
        return WireMatrix2D(
            values=array,
            shape=array.shape,
            bitwidth=bitwidth,
            name=name,
            valid=True,
        )

    @pytest.mark.filterwarnings("ignore:Both systolic array inputs are NumPy arrays")
    def test_systolic_array_two_ndarrays(self) -> None:
        """Test matrix multiplication with two NumPy arrays."""
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]])
        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        accumulator_bitwidth = input_bitwidth * 2

        b_zero = 1
        ab_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul",
            a=a,
            b=b,
            b_zero=b_zero,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        ab_matrix.ready <<= True
        ab_matrix.make_outputs("ab_matrix")

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = ab_matrix.inspect(sim=sim)

        ab_expected = a @ (b - b_zero)
        np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

    def test_systolic_array_one_wire_matrix_2d(self) -> None:
        """Test matrix multiplication with one WireMatrix2D."""
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]])
        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        accumulator_bitwidth = input_bitwidth * 2
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        b_zero = 1
        ab_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul",
            a=a_matrix,
            b=b,
            b_zero=b_zero,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        ab_matrix.ready <<= True
        ab_matrix.make_outputs("ab_matrix")

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = ab_matrix.inspect(sim=sim)

        ab_expected = a @ (b - b_zero)
        np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

    def test_systolic_array_two_wire_matrix_2d(self) -> None:
        """Test matrix multiplication with two WireMatrix2Ds."""
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]])
        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        accumulator_bitwidth = input_bitwidth * 2
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)
        b_matrix = self.make_wire_matrix_2d(name="b", array=b, bitwidth=input_bitwidth)

        b_zero = 1
        ab_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul",
            a=a_matrix,
            b=b_matrix,
            b_zero=b_zero,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        ab_matrix.ready <<= True
        ab_matrix.make_outputs("ab_matrix")

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = ab_matrix.inspect(sim=sim)

        ab_expected = a @ (b - b_zero)
        np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

    def make_memblock(
        self,
        name: str,
        array: np.ndarray,
        input_bitwidth: int,
        counter_bitwidth: int,
        left_input: bool,
    ) -> (pyrtl.MemBlock, dict):
        if left_input:
            element_size = array.shape[0]
        else:
            element_size = array.shape[1]
        memblock = pyrtl.MemBlock(
            addrwidth=counter_bitwidth, bitwidth=input_bitwidth * element_size
        )
        matrix = WireMatrix2D(
            values=memblock,
            shape=array.shape,
            bitwidth=input_bitwidth,
            name=name,
            valid=True,
        )

        if not left_input:
            array = array.transpose()
        memblock_data = pyrtl_matrix.make_input_memblock_data(
            array, input_bitwidth, counter_bitwidth
        )
        memblock_data = dict(enumerate(memblock_data))

        return matrix, memblock, memblock_data

    def test_systolic_array_memblock(self) -> None:
        """Test matrix multiplication with two MemBlocks."""
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]])
        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        accumulator_bitwidth = input_bitwidth * 2

        done_cycle = pyrtl_matrix.num_systolic_array_cycles(a.shape, b.shape) - 1
        counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

        matrix_a, memblock_a, memblock_data_a = self.make_memblock(
            name="a",
            array=a,
            input_bitwidth=input_bitwidth,
            counter_bitwidth=counter_bitwidth,
            left_input=True,
        )
        matrix_b, memblock_b, memblock_data_b = self.make_memblock(
            name="b",
            array=b,
            input_bitwidth=input_bitwidth,
            counter_bitwidth=counter_bitwidth,
            left_input=False,
        )

        b_zero = 1
        ab_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul",
            a=matrix_a,
            b=matrix_b,
            b_zero=b_zero,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        ab_matrix.ready <<= True
        ab_matrix.make_outputs("ab_matrix")

        sim = pyrtl.Simulation(
            memory_value_map={memblock_a: memblock_data_a, memblock_b: memblock_data_b}
        )
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = ab_matrix.inspect(sim=sim)

        ab_expected = a @ (b - b_zero)
        np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

    def test_systolic_array_idle(self) -> None:
        """Test that the systolic array remain idle before and after processing.

        The systolic array should remain idle before its inputs are ``valid`` and after
        its output is ``valid``.

        """
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]])
        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        accumulator_bitwidth = input_bitwidth * 2
        a_valid = pyrtl.Input(name="a_valid", bitwidth=1)
        a_matrix = WireMatrix2D(
            values=None, shape=a.shape, bitwidth=input_bitwidth, name="a", valid=a_valid
        )

        ab_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul",
            a=a_matrix,
            b=b,
            b_zero=0,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        ab_matrix.ready <<= True
        ab_matrix.make_outputs("ab_matrix")

        sim = pyrtl.Simulation()
        provided_inputs = a_matrix.make_provided_inputs(a)
        provided_inputs[a_valid.name] = False
        # Check that the systolic array remains in ``State.INIT`` before its input is
        # valid.
        for _ in range(10):
            sim.step(provided_inputs=provided_inputs)
            state = sim.inspect("matmul.state")
            self.assertEqual(state, pyrtl_matrix.State.INIT)

        # Mark the input as valid and run the systolic array.
        provided_inputs[a_valid.name] = True
        while not sim.inspect("matmul.output.valid"):
            sim.step(provided_inputs=provided_inputs)

        # The systolic array should be in ``State.DONE``.
        state = sim.inspect("matmul.state")
        self.assertEqual(state, pyrtl_matrix.State.DONE)

        # Check that the systolic array remains in ``State.DONE``.
        for _ in range(10):
            sim.step(provided_inputs=provided_inputs)
            state = sim.inspect("matmul.state")
            self.assertEqual(state, pyrtl_matrix.State.DONE)

        ab_actual = ab_matrix.inspect(sim=sim)

        ab_expected = a @ b
        np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

    def test_chained_systolic_arrays(self) -> None:
        """Test using the output of one systolic array as the input to another."""
        # Matrix a has shape (2, 3).
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        # Matrix b has shape (3, 4).
        b = np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]])
        # Matrix c has shape (4, 3).
        c = np.array([[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42]])
        input_bitwidth = max(
            [pyrtl_matrix.minimum_bitwidth(m) for m in [a, b, a @ b, c]]
        )
        accumulator_bitwidth = input_bitwidth * 2
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        ab_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul0",
            a=a_matrix,
            b=b,
            b_zero=0,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        abc_matrix = pyrtl_matrix.make_systolic_array(
            name="matmul1",
            a=ab_matrix,
            b=c,
            b_zero=0,
            input_bitwidth=input_bitwidth,
            accumulator_bitwidth=accumulator_bitwidth,
        )
        abc_matrix.ready <<= True
        abc_matrix.make_outputs("abc_matrix")

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul1.output.valid"):
            sim.step()

        abc_actual = abc_matrix.inspect(sim=sim)

        abc_expected = a @ b @ c
        np.testing.assert_array_equal(abc_actual, abc_expected, strict=True)

    def test_memblock_writer_blocks(self) -> None:
        """Test writing a matrix to a MemBlock using blocks."""
        # Matrix ``m`` has shape (5, 3), and is written in blocks with shape (2, 2). We
        # specifically chose this configuration to ensure that ``make_memblock_writer``
        # works correctly when the matrix size is not divisible by the block size, so
        # blocks on the bottom and right edges extend past ``m``.
        m = np.array(
            [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12], [13, -14, 15]]
        )
        block_shape = (2, 2)
        bitwidth = 8
        addrwidth = 3

        block_valid = pyrtl.Input(name="block_valid", bitwidth=1)
        block = WireMatrix2D(
            values=None,
            shape=block_shape,
            bitwidth=bitwidth,
            name="block",
            valid=block_valid,
        )
        m_matrix = pyrtl_matrix.make_memblock_writer(
            name="m",
            block=block,
            shape=m.shape,
            addrwidth=addrwidth,
            block_row_index=pyrtl.Input(name="block_row_index", bitwidth=2),
            block_column_index=pyrtl.Input(name="block_column_index", bitwidth=1),
        )
        m_matrix.ready <<= True

        # Pad ``m`` to a whole number of blocks. The padding values are outside ``m``,
        # so they should not be copied to the MemBlock.
        padded_m = np.pad(m, ((0, 1), (0, 1)), constant_values=99)

        sim = pyrtl.Simulation()
        for block_row_index, block_column_index in [
            (2, 1),
            (0, 0),
            (1, 1),
            (2, 0),
            (0, 1),
            (1, 0),
        ]:
            row = block_row_index * 2
            column = block_column_index * 2
            provided_inputs = block.make_provided_inputs(
                padded_m[row : row + 2, column : column + 2]
            )
            provided_inputs["block_valid"] = True
            provided_inputs["block_row_index"] = block_row_index
            provided_inputs["block_column_index"] = block_column_index
            # Hold the block data steady until the MemBlock writer indicates that
            # we can overwrite it.
            ready = False
            while not ready:
                sim.step(provided_inputs=provided_inputs)
                self.assertFalse(sim.inspect("m.valid"))
                ready = sim.inspect("block.ready")

        provided_inputs["block_valid"] = False
        while not sim.inspect("m.valid"):
            sim.step(provided_inputs=provided_inputs)

        expected = pyrtl_matrix.make_input_memblock_data(
            m.transpose(), input_bitwidth=bitwidth, addrwidth=addrwidth
        )
        mem = sim.inspect_mem(m_matrix.memblock)
        actual = [mem.get(addr, 0) for addr in range(2**addrwidth)]
        self.assertEqual(actual, expected)

    def test_memblock_writer_systolic_array(self) -> None:
        """Test matrix multiplication with a matrix from ``make_memblock_writer``."""
        test_cases = [
            # Normal test case.
            (
                np.array([[1, -2, 3], [-4, 5, -6]]),
                np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]),
            ),
            # Test with a 1x1 ``b`` to make sure ``make_memblock_writer`` doesn't raise
            # ``valid`` too early.
            (np.array([[1], [-4]]), np.array([[11]])),
        ]
        for a, b in test_cases:
            with self.subTest(b_shape=b.shape):
                pyrtl.reset_working_block()

                input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
                accumulator_bitwidth = input_bitwidth * 2

                done_cycle = (
                    pyrtl_matrix.num_systolic_array_cycles(a.shape, b.shape) - 1
                )
                counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

                # Write all of ``b`` as one block.
                b_matrix = pyrtl_matrix.make_memblock_writer(
                    name="b_memblock",
                    block=self.make_wire_matrix_2d(
                        name="b", array=b, bitwidth=input_bitwidth
                    ),
                    shape=b.shape,
                    addrwidth=counter_bitwidth,
                    block_row_index=pyrtl.Const(0),
                    block_column_index=pyrtl.Const(0),
                )

                b_zero = 1
                ab_matrix = pyrtl_matrix.make_systolic_array(
                    name="matmul",
                    a=a,
                    b=b_matrix,
                    b_zero=b_zero,
                    input_bitwidth=input_bitwidth,
                    accumulator_bitwidth=accumulator_bitwidth,
                )
                ab_matrix.ready <<= True
                ab_matrix.make_outputs("ab_matrix")

                sim = pyrtl.Simulation()
                while not sim.inspect("matmul.output.valid"):
                    sim.step()

                ab_actual = ab_matrix.inspect(sim=sim)

                ab_expected = a @ (b - b_zero)
                np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

    def test_elementwise_add(self) -> None:
        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[10, 11, 12], [13, 14, 15]])

        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        output_bitwidth = input_bitwidth + 1
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)
        b_matrix = self.make_wire_matrix_2d(name="b", array=b, bitwidth=input_bitwidth)

        ab_matrix = pyrtl_matrix.make_elementwise_add(
            name="add", a=a_matrix, b=b_matrix, output_bitwidth=output_bitwidth
        )
        ab_matrix.ready <<= True
        ab_matrix.make_outputs("ab_matrix")

        sim = pyrtl.Simulation()
        sim.step()

        ab_actual = ab_matrix.inspect(sim=sim)
        ab_expected = a + b

        np.testing.assert_array_equal(ab_actual, ab_expected, strict=True)

        # Test bias shape addition.

        a = np.array([[1, -2, 3], [-4, 5, -6]])
        b = np.array([[10], [13]])

        input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(m) for m in [a, b]])
        output_bitwidth = input_bitwidth + 1
        a_matrix = self.make_wire_matrix_2d(
            name="a_bias", array=a, bitwidth=input_bitwidth
        )
        b_matrix = self.make_wire_matrix_2d(
            name="b_bias", array=b, bitwidth=input_bitwidth
        )

        ab_matrix_bias = pyrtl_matrix.make_elementwise_add(
            name="add_bias", a=a_matrix, b=b_matrix, output_bitwidth=output_bitwidth
        )
        ab_matrix_bias.ready <<= True
        ab_matrix_bias.make_outputs("ab_matrix_bias")

        sim = pyrtl.Simulation()
        sim.step()

        ab_actual_bias = ab_matrix_bias.inspect(sim=sim)
        ab_expected_bias = a + b

        np.testing.assert_array_equal(ab_actual_bias, ab_expected_bias, strict=True)

    def test_elementwise_relu(self) -> None:
        a = np.array([[1, -2, 3], [-4, 5, -6]])

        input_bitwidth = pyrtl_matrix.minimum_bitwidth(a)
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        relu_matrix = pyrtl_matrix.make_elementwise_relu(name="relu", a=a_matrix)
        relu_matrix.ready <<= True
        relu_matrix.make_outputs("relu_matrix")

        sim = pyrtl.Simulation()
        sim.step()

        relu_actual = relu_matrix.inspect(sim=sim)
        relu_expected = np.maximum(0, a)

        np.testing.assert_array_equal(relu_actual, relu_expected, strict=True)

    def test_saturating_truncate(self) -> None:
        input = pyrtl.Input(name="input", bitwidth=9)
        output = pyrtl_matrix.saturating_truncate(input, bitwidth=8)
        output.name = "output"

        sim = pyrtl.Simulation()

        def inspect_signed_output() -> int:
            return pyrtl.val_to_signed_integer(sim.inspect("output"), bitwidth=8)

        sim.step({"input": 42})
        self.assertEqual(inspect_signed_output(), 42)

        sim.step({"input": 0})
        self.assertEqual(inspect_signed_output(), 0)

        sim.step({"input": -24})
        self.assertEqual(inspect_signed_output(), -24)

        sim.step({"input": 127})
        self.assertEqual(inspect_signed_output(), 127)

        sim.step({"input": 128})
        self.assertEqual(inspect_signed_output(), 127)

        sim.step({"input": -128})
        self.assertEqual(inspect_signed_output(), -128)

        sim.step({"input": -129})
        self.assertEqual(inspect_signed_output(), -128)

    def test_normalize(self) -> None:
        a = np.array([[1, -2, 3], [-4, 5, -6]]).astype(np.int32)

        input_bitwidth = pyrtl_matrix.minimum_bitwidth(a)
        # m0 must be in the interval [.5, 1).
        m0 = Fxp([0.5, 0.6], signed=False, n_word=input_bitwidth, n_frac=input_bitwidth)
        n = np.array([1, 2])
        z3 = np.array([3, 4])

        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        normal_matrix = pyrtl_matrix.make_elementwise_normalize(
            name="normalize",
            a=a_matrix,
            m0=m0,
            n=n,
            z3=z3,
            output_bitwidth=input_bitwidth,
        )
        normal_matrix.ready <<= True
        normal_matrix.make_outputs("normal_matrix")

        sim = pyrtl.Simulation()
        sim.step()

        normal_actual = normal_matrix.inspect(sim=sim)
        normal_expected = numpy_inference.normalize(product=a, m0=m0, n=n, z3=z3)

        np.testing.assert_array_equal(normal_actual, normal_expected, strict=True)

    def test_normalize_overflow(self) -> None:
        a = np.array([[1, -2, 3], [-4, 5, -6]]).astype(np.int32)

        input_bitwidth = 8
        # m0 must be in the interval [.5, 1).
        m0 = Fxp([0.5, 0.6], signed=False, n_word=input_bitwidth, n_frac=input_bitwidth)
        n = np.array([1, 2])
        z3 = np.array([127, -128])

        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        normal_matrix = pyrtl_matrix.make_elementwise_normalize(
            name="normalize",
            a=a_matrix,
            m0=m0,
            n=n,
            z3=z3,
            output_bitwidth=input_bitwidth,
        )
        normal_matrix.ready <<= True
        normal_matrix.make_outputs("normal_matrix")

        sim = pyrtl.Simulation()
        sim.step()

        normal_actual = normal_matrix.inspect(sim=sim)
        normal_expected = numpy_inference.normalize(product=a, m0=m0, n=n, z3=z3)

        np.testing.assert_array_equal(normal_actual, normal_expected, strict=True)

    def test_argmax(self) -> None:
        a = np.array([[1, -2, 3, -4, 5, -6]]).transpose()

        input_bitwidth = pyrtl_matrix.minimum_bitwidth(a)
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        SingleArgmaxValue = pyrtl.wire_matrix(component_schema=4, size=1)

        SingleArgmaxValue(
            name="argmax_out", values=pyrtl_matrix.make_argmax(a=a_matrix)
        )

        sim = pyrtl.Simulation()
        sim.step()

        argmax_actual = sim.inspect("argmax_out[0]")
        argmax_expected = a.argmax()

        self.assertEqual(argmax_expected, argmax_actual)

        # Now, test batched argmax
        pyrtl.reset_working_block()

        a = np.array(
            [[1, -2, 3, -4, 5, -6], [1337, 532, -2048, 111624, 914, 0]]
        ).transpose()

        input_bitwidth = pyrtl_matrix.minimum_bitwidth(a)
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        BatchArgmaxValues = pyrtl.wire_matrix(component_schema=4, size=2)

        BatchArgmaxValues(
            name="argmax_out", values=pyrtl_matrix.make_argmax(a=a_matrix)
        )

        sim = pyrtl.Simulation()
        sim.step()

        argmax_actual = np.array(
            [sim.inspect(f"argmax_out[{i}]") for i in range(a.shape[1])]
        )
        argmax_expected = a.argmax(axis=0)

        np.testing.assert_array_equal(argmax_actual, argmax_expected, strict=True)


if __name__ == "__main__":
    unittest.main()
