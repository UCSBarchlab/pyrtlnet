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
    else:
        print(f"{name} results DO NOT MATCH!")
        print(f"\nExpected {name} tensor is:\n{expected}")
        print(f"\nActual {name} tensor is:\n{actual}")
        return False


def main():
    """Simulate matrix multiplication and elementwise addition (x ⋅ y + a).

    Do the calculations with numpy and PyRTL, and verify that the results are the same.

    """
    parser = argparse.ArgumentParser(prog="systolic.py")
    parser.add_argument("--x_shape", type=int, nargs=2, default=(2, 3))
    parser.add_argument("--x_start", type=int, default=1)
    parser.add_argument("--y_shape", type=int, default=(3, 4))
    parser.add_argument("--y_start", type=int, default=7)
    parser.add_argument("--a_shape", type=int, nargs=2, default=(2, 4))
    parser.add_argument("--a_start", type=int, default=1)
    args = parser.parse_args()

    def make_np_matrix(shape: tuple[int, int], start: int) -> np.ndarray:
        """Return an integer matrix with the specified shape.

        The matrix will be filled with increasing integers starting from ``start``.

        """
        num_rows, num_columns = shape
        array = np.array(list(range(start, start + num_rows * num_columns)))
        return np.reshape(array, newshape=shape)

    x = make_np_matrix(args.x_shape, start=args.x_start)
    y = make_np_matrix(args.y_shape, start=args.y_start)
    a = make_np_matrix(args.a_shape, start=args.a_start)

    assert x.shape[1] == y.shape[0]

    y_zero = 0
    expected_xy = x @ (y - y_zero)
    assert expected_xy.shape == a.shape

    input_bitwidth = max(
        [pyrtl_matrix.minimum_bitwidth(a) for a in [x, y, expected_xy]]
    )
    accumulator_bitwidth = 32

    done_cycle = pyrtl_matrix.num_systolic_array_cycles(x.shape, y.shape) - 1
    counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

    _, num_columns = y.shape
    y_memblock = pyrtl.MemBlock(
        addrwidth=counter_bitwidth, bitwidth=input_bitwidth * num_columns
    )
    y_valid = pyrtl.Input(name="y_valid", bitwidth=1)
    matrix_y = WireMatrix2D(
        values=y_memblock,
        shape=y.shape,
        bitwidth=input_bitwidth,
        name="y",
        valid=y_valid,
    )

    matrix_xy = pyrtl_matrix.make_systolic_array(
        name="mm0",
        a=x,
        b=matrix_y,
        b_zero=y_zero,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )
    matrix_xy.make_outputs()
    matrix_a = WireMatrix2D(values=a, bitwidth=input_bitwidth, name="a", valid=True)
    matrix_xya = pyrtl_matrix.make_elementwise_add(
        name="add0", a=matrix_xy, b=matrix_a, output_bitwidth=accumulator_bitwidth
    )
    matrix_xya.make_outputs()
    matrix_xya.ready <<= True

    # Simulate the systolic array by providing inputs for each cycle.
    memblock_data = pyrtl_matrix.make_input_memblock_data(
        y.transpose(), input_bitwidth, counter_bitwidth
    )
    memblock_data = dict(enumerate(memblock_data))

    sim = pyrtl.Simulation(memory_value_map={y_memblock: memblock_data})
    while not sim.inspect("add0.output.valid"):
        sim.step(provided_inputs={"y_valid": True})

    # Simulation complete, print the waveform.
    def render_trace(prefixes: str):
        # Only show traces with the maximum number of brackets. This will display
        # ``output[1][2]``, and skip ``output[1]`` and ``output``.
        def count_brackets(name):
            return name.count("[")

        def strip_brackets(name):
            return name.split("[")[0]

        trace_counts = [
            (strip_brackets(name), count_brackets(name))
            for name in sorted(sim.tracer.trace.keys())
        ]
        max_brackets = {}
        for name, count in trace_counts:
            if name not in max_brackets:
                max_brackets[name] = count
            else:
                max_brackets[name] = max(count, max_brackets[name])

        trace_list = []
        for prefix in prefixes:
            matches = []
            for name in sorted(sim.tracer.trace.keys()):
                if (
                    name.startswith(prefix)
                    and count_brackets(name) == max_brackets[strip_brackets(name)]
                ):
                    matches.append(name)
            trace_list.extend(matches)

        sim.tracer.render_trace(
            trace_list=trace_list,
            symbol_len=4,
            repr_func=pyrtl.val_to_signed_integer,
            repr_per_name={
                "mm0.state": pyrtl.enum_name(pyrtl_matrix.State),
                "mm1.state": pyrtl.enum_name(pyrtl_matrix.State),
                "mm0.counter": int,
            },
        )

    print("Computing x ⋅ y with y_zero", y_zero)
    print(f"x (left) shape={x.shape}:\n{x}")
    print(f"y (top) shape={y.shape}:\n{y}")
    render_trace(
        prefixes=[
            "mm0.left",
            "mm0.top",
            "mm0.output[",
            "mm0.state",
            "mm0.output.valid",
        ]
    )

    actual_xy = matrix_xy.inspect(sim=sim)
    _verify_tensor("x ⋅ y", expected_xy, actual_xy)

    print("\nComputing x ⋅ y + a")
    print(f"a shape={a.shape}:\n{a}\n")

    expected_xya = expected_xy + a
    actual_xya = matrix_xya.inspect(sim=sim)
    _verify_tensor("x ⋅ y + a", expected_xya, actual_xya)


if __name__ == "__main__":
    main()
