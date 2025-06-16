import unittest

import numpy as np
import pyrtl
import pytest
from fxpmath import Fxp

import numpy_inference
import pyrtl_matrix
import wire_matrix_2d


class TestPyrtlMatrix(unittest.TestCase):
    def setUp(self):
        pyrtl.reset_working_block()

    def make_wire_matrix_2d(
        self, name: str, array: np.ndarray, bitwidth: int
    ) -> wire_matrix_2d.WireMatrix2D:
        return wire_matrix_2d.WireMatrix2D(
            values=array,
            shape=array.shape,
            bitwidth=8,
            name=name,
            valid=True,
        )

    @pytest.mark.filterwarnings("ignore:Both systolic array inputs are NumPy arrays")
    def test_systolic_array_two_ndarrays(self):
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
        wire_matrix_2d.make_outputs(matrix=ab_matrix)

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = wire_matrix_2d.inspect_matrix(sim, matrix=ab_matrix)

        ab_expected = a @ (b - b_zero)
        self.assertTrue(np.array_equal(ab_expected, ab_actual))

    def test_systolic_array_one_wire_matrix_2d(self):
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
        wire_matrix_2d.make_outputs(matrix=ab_matrix)

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = wire_matrix_2d.inspect_matrix(sim, matrix=ab_matrix)

        ab_expected = a @ (b - b_zero)
        self.assertTrue(np.array_equal(ab_expected, ab_actual))

    def test_systolic_array_two_wire_matrix_2d(self):
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
        wire_matrix_2d.make_outputs(matrix=ab_matrix)

        sim = pyrtl.Simulation()
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = wire_matrix_2d.inspect_matrix(sim, matrix=ab_matrix)

        ab_expected = a @ (b - b_zero)
        self.assertTrue(np.array_equal(ab_expected, ab_actual))

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
        matrix = wire_matrix_2d.WireMatrix2D(
            values=memblock,
            shape=array.shape,
            bitwidth=input_bitwidth,
            name=name,
            valid=True,
        )

        if not left_input:
            array = array.transpose()
        romdata = pyrtl_matrix.make_input_romdata(
            array, input_bitwidth, counter_bitwidth
        )
        memblock_data = {i: d for i, d in enumerate(romdata)}

        return matrix, memblock, memblock_data

    def test_systolic_array_memblock(self):
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
        wire_matrix_2d.make_outputs(matrix=ab_matrix)

        sim = pyrtl.Simulation(
            memory_value_map={memblock_a: memblock_data_a, memblock_b: memblock_data_b}
        )
        while not sim.inspect("matmul.output.valid"):
            sim.step()

        ab_actual = wire_matrix_2d.inspect_matrix(sim, matrix=ab_matrix)

        ab_expected = a @ (b - b_zero)
        self.assertTrue(np.array_equal(ab_expected, ab_actual))

    def test_elementwise_add(self):
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
        wire_matrix_2d.make_outputs(matrix=ab_matrix)

        sim = pyrtl.Simulation()
        sim.step()

        ab_actual = wire_matrix_2d.inspect_matrix(sim, matrix=ab_matrix)
        ab_expected = a + b

        self.assertTrue(np.array_equal(ab_expected, ab_actual))

    def test_elementwise_relu(self):
        a = np.array([[1, -2, 3], [-4, 5, -6]])

        input_bitwidth = pyrtl_matrix.minimum_bitwidth(a)
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        relu_matrix = pyrtl_matrix.make_elementwise_relu(name="relu", a=a_matrix)
        relu_matrix.ready <<= True
        wire_matrix_2d.make_outputs(matrix=relu_matrix)

        sim = pyrtl.Simulation()
        sim.step()

        relu_actual = wire_matrix_2d.inspect_matrix(sim, matrix=relu_matrix)
        relu_expected = np.maximum(0, a)

        self.assertTrue(np.array_equal(relu_expected, relu_actual))

    def test_normalize(self):
        a = np.array([[1, -2, 3], [-4, 5, -6]])

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
            input_bitwidth=input_bitwidth,
            output_bitwidth=input_bitwidth,
        )
        normal_matrix.ready <<= True
        wire_matrix_2d.make_outputs(matrix=normal_matrix)

        sim = pyrtl.Simulation()
        sim.step()

        normal_actual = wire_matrix_2d.inspect_matrix(sim, matrix=normal_matrix)
        normal_expected = numpy_inference.normalize(product=a, m0=m0, n=n, z3=z3)

        self.assertTrue(np.array_equal(normal_expected, normal_actual))

    def test_argmax(self):
        a = np.array([[1, -2, 3, -4, 5, -6]]).transpose()

        input_bitwidth = pyrtl_matrix.minimum_bitwidth(a)
        a_matrix = self.make_wire_matrix_2d(name="a", array=a, bitwidth=input_bitwidth)

        argmax = pyrtl_matrix.make_argmax(a=a_matrix)
        argmax.name = "argmax"

        sim = pyrtl.Simulation()
        sim.step()

        argmax_actual = sim.inspect(argmax.name)
        argmax_expected = a.argmax()

        self.assertEqual(argmax_expected, argmax_actual)


if __name__ == "__main__":
    unittest.main()
