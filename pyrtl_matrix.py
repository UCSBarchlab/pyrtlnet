import argparse

import numpy as np
import pyrtl

import pyrtlnet.pyrtl_matrix as pyrtl_matrix
from pyrtlnet.wire_matrix_2d import WireMatrix2D


def _verify_tensor(name: str, expected: np.ndarray, actual: np.ndarray) -> bool:
    """Compare ``expected`` to ``actual`` and print a report."""
    if expected.shape == actual.shape and np.logical_and.reduce(
        expected == actual, axis=None
    ):
        print(f"Correct result for {name} tensor:\n{actual}")
        return True
    print(f"{name} results DO NOT MATCH!")
    print(f"\nExpected {name} tensor is:\n{expected}")
    print(f"\nActual {name} tensor is:\n{actual}")
    return False


def _make_np_matrix(shape: tuple[int, int], start: int) -> np.ndarray:
    """Return an integer matrix with the specified shape.

    The matrix will be filled with increasing integers starting from ``start``.

    """
    num_rows, num_columns = shape
    array = np.array(list(range(start, start + num_rows * num_columns)))
    return np.reshape(array, newshape=shape)


def _render_trace(sim: pyrtl.Simulation, prefixes: list[str]) -> None:
    """Display traces that start with a prefix in ``prefixes``.

    This also displays signed integers instead of the default ``hex``, and displays
    state names.

    """
    # Collect trace names that start with one of the desired ``prefixes``.
    trace_list = []
    # Iterate over ``prefixes`` first so the display order matches the prefix order.
    for prefix in prefixes:
        for trace_name in sorted(sim.tracer.trace.keys()):
            if trace_name.startswith(prefix):
                trace_list.append(trace_name)

    sim.tracer.render_trace(
        trace_list=trace_list,
        repr_func=pyrtl.val_to_signed_integer,
        repr_per_name={"mm0.state": pyrtl.enum_name(pyrtl_matrix.State)},
    )


def main() -> None:
    """Simulate matrix multiplication and elementwise addition (x · (y - y_zero) + a).

    Do the calculations with numpy and PyRTL, and verify that the results are the same.

    """
    parser = argparse.ArgumentParser(prog="pyrtl_matrix.py")
    parser.add_argument("--x_shape", type=int, nargs=2, default=(2, 3))
    parser.add_argument("--x_start", type=int, default=1)
    parser.add_argument("--y_shape", type=int, nargs=2, default=(3, 4))
    parser.add_argument("--y_start", type=int, default=7)
    parser.add_argument("--y_zero", type=int, default=0)
    parser.add_argument("--a_shape", type=int, nargs=2, default=(2, 4))
    parser.add_argument("--a_start", type=int, default=1)
    parser.add_argument("--verilog", action="store_true", default=False)
    args = parser.parse_args()

    # Create input matrices.
    x = _make_np_matrix(args.x_shape, start=args.x_start)
    y = _make_np_matrix(args.y_shape, start=args.y_start)
    y_zero = args.y_zero
    a = _make_np_matrix(args.a_shape, start=args.a_start)

    assert x.shape[1] == y.shape[0]

    # Create ``matrix_y``, a ``WireMatrix2D`` whose values are stored in ``y_memblock``.
    # The actual values for ``y_memblock`` will be provided at simulation time, via
    # ``memory_value_map``.
    input_bitwidth = max([pyrtl_matrix.minimum_bitwidth(a) for a in [x, y]])
    done_cycle = pyrtl_matrix.num_systolic_array_cycles(x.shape, y.shape) - 1
    counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth
    _, num_columns = y.shape
    y_memblock = pyrtl.MemBlock(
        addrwidth=counter_bitwidth, bitwidth=input_bitwidth * num_columns
    )
    matrix_y = WireMatrix2D(
        values=y_memblock,
        shape=y.shape,
        bitwidth=input_bitwidth,
        name="y",
        valid=True,
    )

    # Build the systolic array hardware for matrix multiplication, and generate
    # pyrtl.Outputs for the array's output. The outputs will have names like
    # ``mm0.output[0][0]``.
    accumulator_bitwidth = 32
    matrix_xy = pyrtl_matrix.make_systolic_array(
        name="mm0",
        a=x,
        b=matrix_y,
        b_zero=y_zero,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )
    matrix_xy.make_outputs("matrix_xy")

    # Build the elementwise adder. It consumes the systolic array's output, and adds
    # ``matrix_a`` to it, which is an array of ``pyrtl.Const``s. Generate pyrtl.Outputs
    # for the adder's output. The outputs will have names like ``add0.output[0][0]``.
    matrix_a = WireMatrix2D(values=a, bitwidth=input_bitwidth, name="a", valid=True)
    matrix_xya = pyrtl_matrix.make_elementwise_add(
        name="add0", a=matrix_xy, b=matrix_a, output_bitwidth=accumulator_bitwidth
    )
    matrix_xya.make_outputs("matrix_xya")
    matrix_xya.ready <<= True

    # Provide the initial data for ``y_memblock``.
    memblock_data = pyrtl_matrix.make_input_memblock_data(
        y.transpose(), input_bitwidth, counter_bitwidth
    )
    memblock_data = dict(enumerate(memblock_data))

    # Simulate the systolic array and elementwise adder until the adder's output is
    # ``valid``.
    sim = pyrtl.Simulation(memory_value_map={y_memblock: memblock_data})
    while not sim.inspect("add0.output.valid"):
        sim.step()

    # Print the waveform.
    print("Computing x · (y - y_zero)")
    print(f"x (left) shape={x.shape}:\n{x}")
    print(f"y (top) shape={y.shape}:\n{y}")
    print("y_zero:", y_zero)
    _render_trace(
        sim=sim,
        prefixes=[
            "mm0.left",
            "mm0.top",
            "mm0.output[",
            "mm0.state",
            "mm0.output.valid",
        ],
    )

    # Calculate the expected value for x ⋅ (y - y_zero) with NumPy.
    expected_xy = x @ (y - y_zero)
    assert expected_xy.shape == a.shape

    # Verify that the matrix multiplication output is correct.
    actual_xy = matrix_xy.inspect(sim=sim)
    _verify_tensor("x ⋅ y", expected_xy, actual_xy)

    print("\nComputing x · (y - y_zero) + a")
    print(f"a shape={a.shape}:\n{a}\n")

    expected_xya = expected_xy + a
    actual_xya = matrix_xya.inspect(sim=sim)
    _verify_tensor("x ⋅ y + a", expected_xya, actual_xya)

    if args.verilog:
        with open("pyrtl_matrix.v", "w") as output:
            pyrtl.output_to_verilog(output)
            pyrtl.output_verilog_testbench(
                output,
                simulation_trace=sim.tracer,
                vcd="pyrtl_matrix.vcd",
                cmd=(
                    '$display("time %3t, matrix_xya:'
                    '\\n[[%3d %3d %3d %3d]\\n [%3d %3d %3d %3d]\\n", '
                    "$time, "
                    "$signed(matrix_xya_0_0), "
                    "$signed(matrix_xya_0_1), "
                    "$signed(matrix_xya_0_2), "
                    "$signed(matrix_xya_0_3), "
                    "$signed(matrix_xya_1_0), "
                    "$signed(matrix_xya_1_1), "
                    "$signed(matrix_xya_1_2), "
                    "$signed(matrix_xya_1_3));"
                ),
            )


if __name__ == "__main__":
    main()
