import unittest

import numpy as np
import pyrtl

import wire_matrix_2d


class TestWireMatrix2D(unittest.TestCase):
    def setUp(self):
        pyrtl.reset_working_block()

    def test_input_matrix(self):
        """Test a WireMatrix2D of Inputs."""
        expected_values = np.array([[1, 2, 3], [4, 5, 6]])
        shape = expected_values.shape
        bitwidth = 8
        matrix = wire_matrix_2d.WireMatrix2D(
            values=None, shape=shape, bitwidth=bitwidth, name="input", ready=True
        )

        # Create an Output for each matrix element.
        wire_matrix_2d.make_outputs(matrix)

        # Run simulation and verify that the correct matrix elements were retrieved.
        sim = pyrtl.Simulation()

        provided_inputs = wire_matrix_2d.make_provided_inputs(
            matrix.name, expected_values
        )
        sim.step(provided_inputs=provided_inputs)

        actual_values = wire_matrix_2d.inspect_matrix(sim=sim, matrix=matrix)
        self.assertTrue(np.array_equal(expected_values, actual_values))

    def test_lists(self):
        """Test a WireMatrix2D created from a list of lists."""
        lists = [[1, 2, 3], [4, 5, 6]]
        expected_values = np.array(lists)
        shape = expected_values.shape
        bitwidth = 8
        matrix = wire_matrix_2d.WireMatrix2D(
            values=lists,
            shape=shape,
            bitwidth=bitwidth,
            name="lists",
            ready=True,
            valid=True,
        )

        # Create an Output for each matrix element.
        wire_matrix_2d.make_outputs(matrix)

        # Run simulation and verify that the correct matrix elements were retrieved.
        sim = pyrtl.Simulation()
        sim.step()

        actual_values = wire_matrix_2d.inspect_matrix(sim=sim, matrix=matrix)
        self.assertTrue(np.array_equal(expected_values, actual_values))

    def test_numpy_array(self):
        """Test a WireMatrix2D created from a NumPy array."""
        expected_values = np.array([[1, 2, 3], [4, 5, 6]])
        shape = expected_values.shape
        bitwidth = 8
        matrix = wire_matrix_2d.WireMatrix2D(
            values=expected_values,
            shape=shape,
            bitwidth=bitwidth,
            name="ndarray",
            ready=True,
            valid=True,
        )

        # Create an Output for each matrix element.
        wire_matrix_2d.make_outputs(matrix)

        # Run simulation and verify that the correct matrix elements were retrieved.
        sim = pyrtl.Simulation()
        sim.step()

        actual_values = wire_matrix_2d.inspect_matrix(sim=sim, matrix=matrix)
        self.assertTrue(np.array_equal(expected_values, actual_values))

    def test_concatenated_input_wire_vector(self):
        """Test a WireMatrix2D created from a concatenated Input WireVector."""
        expected_values = np.array([[1, 2, 3], [4, 5, 6]])
        shape = expected_values.shape
        bitwidth = 8
        input_name = "concatenated_input"
        concatenated_input = pyrtl.Input(
            name=input_name, bitwidth=bitwidth * expected_values.size
        )
        matrix = wire_matrix_2d.WireMatrix2D(
            values=concatenated_input,
            shape=shape,
            bitwidth=bitwidth,
            name="input_matrix",
            ready=True,
            valid=True,
        )

        # Create an Output for each matrix element.
        wire_matrix_2d.make_outputs(matrix)

        # Run simulation and verify that the correct matrix elements were retrieved.
        sim = pyrtl.Simulation()
        sim.step(
            provided_inputs={
                input_name: wire_matrix_2d.make_concatenated_value(
                    expected_values, bitwidth
                )
            }
        )

        actual_values = wire_matrix_2d.inspect_matrix(sim=sim, matrix=matrix)
        self.assertTrue(np.array_equal(expected_values, actual_values))

    def test_transpose(self):
        """Test transposing a WireMatrix2D created from a NumPy array."""
        expected_values = np.array([[1, 2, 3], [4, 5, 6]])
        shape = expected_values.shape
        bitwidth = 8
        matrix = wire_matrix_2d.WireMatrix2D(
            values=expected_values,
            shape=shape,
            bitwidth=bitwidth,
            name="ndarray",
            ready=True,
            valid=True,
        )
        transposed = matrix.transpose()

        # Create an Output for each transposed matrix element.
        wire_matrix_2d.make_outputs(transposed)

        # Run simulation and verify that the correct matrix elements were retrieved.
        sim = pyrtl.Simulation()
        sim.step()

        actual_transposed_values = wire_matrix_2d.inspect_matrix(
            sim=sim, matrix=transposed
        )
        self.assertTrue(np.array_equal(expected_values.T, actual_transposed_values))


if __name__ == "__main__":
    unittest.main()
