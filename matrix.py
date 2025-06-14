import argparse
import enum

import numpy as np
import pyrtl
from fxpmath import Fxp

import wire_matrix_2d


def make_input_romdata(a: np.ndarray, input_bitwidth: int, addrwidth: int) -> list[int]:
    """Convert a numpy matrix to romdata for the systolic array.

    The systolic array reads one entry of this romdata each cycle. The romdata entry
    contains concatenated values for all {left, top} inputs needed in the next cycle.

    The matrix is padded with zeroes and shifted into a parallelogram shape. See the
    "top inputs for each cycle" comment in `make_systolic_array()` for an example.

    The returned romdata can be converted to a dict and used with a MemBlock, via
    Simulation's memory_value_map.

    """
    # Pad and shift the data.
    num_rows, num_inner = a.shape
    num_cycles = 2**addrwidth
    data = [[None for _ in range(num_rows)] for _ in range(num_cycles)]
    for cycle in range(num_cycles):
        for row in range(num_rows):
            if cycle < row or cycle >= row + num_inner:
                data[cycle][row] = 0
            else:
                data[cycle][row] = a[row][cycle - row]

    def cycle_to_int(cycle_datas: list[int]) -> int:
        """Concatenate all the input data needed in one cycle into an integer."""
        output = 0
        input_mask = -1 & (2**input_bitwidth - 1)
        for cycle_data in cycle_datas:
            output = output << input_bitwidth
            # `output` may be very large, easily exceeding 64 bits. The `int()` below
            # ensures we do these computations with Python's arbitrary precision
            # integers, rather than a fixed-width type like np.int64.
            output |= int(cycle_data) & input_mask
        return int(output)

    # Pack the per-cycle data into romdata.
    romblock_data = [None for _ in range(num_cycles)]
    for cycle in range(num_cycles):
        romblock_data[cycle] = cycle_to_int(data[cycle])
    return romblock_data


def make_input_romblock(
    a: np.ndarray, input_bitwidth: int, addrwidth: int
) -> pyrtl.RomBlock:
    """Convert a numpy array to a RomBlock for use with a systolic array."""
    num_rows, num_inner = a.shape
    romblock_data = make_input_romdata(a, input_bitwidth, addrwidth)
    romblock = pyrtl.RomBlock(
        addrwidth=addrwidth,
        bitwidth=input_bitwidth * num_rows,
        romdata=romblock_data,
        max_read_ports=1,
    )

    return romblock


class State(enum.IntEnum):
    INIT = 0  # Initialize systolic array inputs.
    READ = 1  # Read first MemBlock address.
    BUSY = 2  # Multiply matrices.
    DONE = 3  # Wait for output to be consumed.


def _make_systolic_array_wire_inputs(
        a: wire_matrix_2d.WireMatrix2D, reset: pyrtl.WireVector, input_bitwidth: int
) -> list[pyrtl.WireVector]:
    """Generate left inputs for wire_matrix `a` from a WireMatrix2D of WireVectors.

    `a` has shape `(num_rows, num_inner)`.

    The generated input will arrive over `(num_inner + num_rows - 1)` cycles.
    If `a` is:
        ┌       ┐
    a = │ 1 2 3 │
        │ 4 5 6 │
        └       ┘

    Then the left inputs will be:

       │  cycle
       │ 0 1 2 3
    ───┼───────
    l0 │ 1 2 3 0
    l1 │ 0 4 5 6

    Returns a list of WireVectors for each row. In the example above, this function
    would return [l0, l1]. Over the first four cycles of simulation, l0 produces the
    values [1, 2, 3, 0] and l1 produces the values [0, 4, 5, 6].

    """
    assert a.memblock is None

    num_rows, num_inner = a.shape
    num_cycles = num_inner + num_rows - 1

    # Start with the rightmost column, cycle 3 in the example above. Each row of the
    # 'left inputs' table above is implemented with a chain of registers. These
    # registers shift their values left each cycle, so the leftmost register will
    # have the correct value for the current cycle.
    all_registers = [
        [None for column in range(num_cycles)] for row in range(num_rows)
    ]
    for cycle in reversed(range(num_cycles)):
        for row in range(num_rows):
            reg = pyrtl.Register(bitwidth=input_bitwidth)

            # reg_init is this register's initial value. These are the values shown
            # in the 'left inputs' table above.
            if cycle >= num_inner + row or cycle < row:
                # Fill in upper right triangular and bottom left triangular zeroes.
                reg_init = 0
            else:
                reg_init = a[row][cycle - row]

            # Link the registers together in a chain. The rightmost register has no
            # right neighbor, so we set its next value to zero to keep the output
            # stable. This works because multiplying by zero and accumulating does not
            # change the accumulator's state.
            if cycle == num_cycles - 1:
                reg_next = 0
            else:
                reg_next = all_registers[row][cycle + 1]

            reg.next <<= pyrtl.select(reset, reg_init, reg_next)

            all_registers[row][cycle] = reg
    # Return the leftmost register for each row.
    return [all_registers[row][0] for row in range(num_rows)]

def _make_systolic_array_memblock_inputs(
        shape: tuple, addr: pyrtl.WireVector, mem: pyrtl.MemBlock, input_bitwidth: int
) -> list[pyrtl.WireVector]:
    """Generate left inputs for wire_matrix `a` from a WireMatrix2D with a MemBlock.

    This is like _make_systolic_array_wire_inputs, except it generates the systolic
    array's inputs for each cycle by reading a MemBlock instead of building a chain of
    shift registers.

    The MemBlock's contents must be formatted with `make_input_romdata` or
    `make_input_romblock`.

    """
    num_rows, num_inner = shape
    InputRow = pyrtl.wire_matrix(component_schema=input_bitwidth, size=num_rows)
    input_row = InputRow(concatenated_type=pyrtl.Register)
    input_row.next <<= mem[addr]
    return [input_row[i] for i in range(num_rows)]

def make_systolic_array(
    name: str,
    a: wire_matrix_2d.WireMatrix2D | np.ndarray,
    b: wire_matrix_2d.WireMatrix2D | np.ndarray,
    b_zero: int,
    input_bitwidth: int,
    accumulator_bitwidth: int,
) -> wire_matrix_2d.WireMatrix2D:
    """Generate an output-stationary systolic array, computing `a ⋅ (b - b_zero)`.

    `b_zero` can be useful for quantized neural network computations. Set it to zero for
    standard matrix multiplication.

    `input_bitwidth` is the bitwidth of each input element.

    `accumulator_bitwidth` is the bitwidth used when summing dot products. It should be
    larger than `input_bitwidth`.

    This implementation follows Figure 5 in "A Compiler Infrastructure for Accelerator
    Generators", https://arxiv.org/pdf/2102.09713.pdf :

    In the diagram below, l0' is l0, delayed by one cycle, and l0'' is l0, delayed by
    two cycles.

                     t0                         t1
                     │                          │
                     ▼                          ▼
                ┌─────────┐    l0'         ┌─────────┐    l0''
       l0 ─────▶│ reg_0_0 │─────┬─────────▶│ reg_0_1 │─────┬───── ...
                └─────────┘     │          └─────────┘     │
                     │          │               │          │
                     │          ▼               │          ▼
                     │      ┌────────┐          │      ┌────────┐
                 t0' ├─────▶│ pe_0_0 │      t1' ├─────▶│ pe_0_1 │
                     │      └────────┘          │      └────────┘
                     │                          │
                     ▼                          ▼
                ┌─────────┐    l1'         ┌─────────┐    l1''
       l1 ─────▶│ reg_1_0 │─────┬─────────▶│ reg_1_1 │─────┬───── ...
                └─────────┘     │          └─────────┘     │
                     │          │               │          │
                     │          ▼               │          ▼
                     │      ┌────────┐          │      ┌────────┐
                t0'' ├─────▶│ pe_1_0 │     t1'' ├─────▶│ pe_1_1 │
                     │      └────────┘          │      └────────┘
                     │                          │
                    ...                        ...

    The systolic array multiplies matrices A and B. A has shape (num_rows, num_inner)
    and B has shape (num_inner, num_columns).

    The systolic array is an array of processing elements, with num_rows rows and
    num_columns columns.

    Matrix A streams in the left inputs (l0, l1, ... ln), over (num_inner + num_rows -
    1) cycles.

    Matrix B streams in the top inputs (t0, t1, ... tn), over (num_inner + num_columns -
    1) cycles.

    Data streams from these left and top inputs, through registers (reg_0_0, reg_0_1,
    ...), to processing elements (pe_0_0, pe_0_1, ...). The processing elements store
    the matrix multiplication output in accumulator registers. The output does not move
    through the array, so this array is "output-stationary."

    The left and top inputs change over time. If the matrix A is:
        ┌       ┐
    A = │ 1 2 3 │
        │ 4 5 6 │
        └       ┘
    then num_rows=2 and num_inner=3 because matrix A has shape (2, 3). There are two
    left inputs because num_rows=2. It will take 4 cycles to stream matrix A (3 + 2 - 1
    = 4). The left inputs for each cycle are:

       │  cycle
       │ 0 1 2 3
    ───┼───────
    l0 │ 1 2 3 0
    l1 │ 0 4 5 6

    Note how l1 is shifted forward one cycle, and the holes have been filled with
    zeroes.

    If the matrix B is:
        ┌             ┐
    B = │  7  8  9 10 │
        │ 11 12 13 14 │
        │ 15 16 17 18 │
        └             ┘
    then num_inner=3 and num_columns=4 because matrix B has shape (3, 4). There are four
    top inputs because num_columns=4. It will take 6 cycles to stream matrix B (3 + 4 -
    1 = 6). The top inputs for each cycle are:

       │        cycle
       │  0  1  2  3  4  5
    ───┼──────────────────
    t0 │  7 11 15  0  0  0
    t1 │  0  8 12 16  0  0
    t2 │  0  0  9 13 17  0
    t3 │  0  0  0 10 14 18

    Note how matrix B has been transposed. t0 is [7 11 15] over the first three cycles,
    which corresponds to the leftmost column of matrix B. t1 is shifted forward one
    cycle, t2 is shifted forward two cycles, and t3 is shifted forward three cycles, and
    the holes have been filled with zeroes.

    Compare t0 and l0. l0 corresponds to the topmost row of matrix A, and t0 corresponds
    to the leftmost column of matrix B. t0 and l0 can be generated by following the same
    procedure, except matrix B is initially transposed, while matrix A is not.

    When there is no more input to stream in to the left or top inputs, the
    corresponding input should be set to zero. The final result will be ready in
    (num_rows + num_inner + num_columns) cycles. The matrix multiplication result can be
    read from the pe_{row}_{col} registers.

    """
    @pyrtl.wire_struct
    class TileIn:
        """Collects a systolic array tile's left and top inputs.

        Each tile's input_register stores the tile's TileIn.

        """

        left: input_bitwidth
        top: input_bitwidth

    @pyrtl.wire_struct
    class TileOut:
        """Collects a systolic array tile's right and bottom outputs.

        Each tile's input_register produces the tile's TileOut.

        """

        right: input_bitwidth
        bottom: input_bitwidth

    def _make_systolic_array_pe(tile_out: TileOut, row: int, column: int,
                                reset: pyrtl.WireVector):
        """Make a multiply-and-accumulate processing element.

        Multiply the tile's outputs and accumulate their sum in a register.

        """
        accumulator = pyrtl.Register(
            bitwidth=accumulator_bitwidth, name=f"{name}.pe[{row}][{column}]"
        )
        accumulator.next <<= pyrtl.select(
            reset,
            0,
            pyrtl.signed_add(
                accumulator,
                # q1 * (q2 - z2) == (q1 * q2) - (q1 * z2)
                pyrtl.signed_mult(
                    tile_out.right, pyrtl.signed_sub(tile_out.bottom, b_zero_const)
                ),
            ),
        )
        return accumulator

    def _make_systolic_array_tile(
        tile_in: TileIn, row: int, column: int, reset: pyrtl.WireVector
    ) -> TileOut:
        """Make a systolic tile's input register (reg) and processing element (pe).

        Returns the tile's outputs.

        We construct the systolic array by composing these tiles:

                            tile_in.top
                      ┌──────── │ ──────────────┐
                      │ tile    ▼               │
                      │ ┌────────────────┐      │
        tile_in.left ──▶│ input_register │─┬──────▶ tile_out.right
                      │ └────────────────┘ │    │
                      │         │          ▼    │
                      │         │        ┌────┐ │
                      │         ├───────▶│ pe │ │
                      │         │        └────┘ │
                      └──────── │ ──────────────┘
                                ▼
                          tile_out.bottom

        tile_out.right is tile_in.left, delayed by one cycle.
        tile_out.bottom is tile_in.top, delayed by one cycle.

        This function generates the tile at {row}, {column}.

        """
        input_register = TileIn(
            name=f"{name}.reg_{row}_{column}", concatenated_type=pyrtl.Register
        )
        input_register.next <<= pyrtl.select(reset, 0, tile_in)
        tile_out = TileOut(bottom=input_register.top, right=input_register.left)
        accumulator = _make_systolic_array_pe(
            tile_out=tile_out, row=row, column=column, reset=reset)
        return (tile_out, accumulator)

    # If b_zero is a length-1 vector, convert it to an integer.
    if not isinstance(b_zero, int):
        assert len(b_zero) == 1
        b_zero = b_zero[0]
    b_zero_const = pyrtl.Const(b_zero, signed=True, bitwidth=input_bitwidth)

    num_rows, num_inner = a.shape
    assert num_inner == b.shape[0]
    num_columns = b.shape[1]

    # 'done_next_cycle' is high when the matrix multiplication is one cycle away from
    # completion. We need to know one cycle ahead to update 'state'.
    done_cycle = num_rows + num_inner + num_columns - 1

    # This counter determines when the matrix multiplication is complete. The counter is
    # also used as an address for reading input data from MemBlocks.
    counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth
    counter = pyrtl.Register(bitwidth=counter_bitwidth)
    counter.name = f"{name}.counter"

    reset = pyrtl.WireVector(bitwidth=1, name=f"{name}.reset")
    reset <<= counter == 0

    def process_input(
            a: wire_matrix_2d.WireMatrix2D | np.ndarray) -> list[pyrtl.WireVector]:
        """Convert a left input matrix `a` into a list of `left` WireVector inputs.

        This can also be used to convert a right input matrix `b` into a list of `top`
        WireVector inputs by transposing the matrix `b`.

        """
        if isinstance(a, np.ndarray):
            left_romblock = make_input_romblock(
                a, input_bitwidth=input_bitwidth, addrwidth=counter_bitwidth
            )
            return _make_systolic_array_memblock_inputs(
                a.shape, counter, left_romblock, input_bitwidth)
        else:
            assert isinstance(a, wire_matrix_2d.WireMatrix2D)
            if a.memblock is not None:
                return _make_systolic_array_memblock_inputs(
                    a.shape, counter, a.memblock, input_bitwidth)
            else:
                return _make_systolic_array_wire_inputs(a, reset, input_bitwidth)

    left = process_input(a)
    for row in range(num_rows):
        left[row].name = f"{name}.left[{row}]"

    top = process_input(b.transpose())
    for col in range(num_columns):
        top[col].name = f"{name}.top[{col}]"

    num_columns = len(top)
    num_rows = len(left)
    num_cycles = num_inner + num_rows - 1

    # Collect a 2D array of tile outputs.
    tile_outs = [[None for column in range(num_columns)] for row in range(num_rows)]
    # Collect a 2D array of accumulator registers.
    accumulators = [[None for column in range(num_columns)] for row in range(num_rows)]

    for row in range(num_rows):
        for column in range(num_columns):
            # If we are on the left edge, this tile's left input comes from the systolic
            # array's left input. Otherwise the left input comes from the left
            # neighboring tile.
            if column == 0:
                current_left = left[row]
            else:
                current_left = tile_outs[row][column - 1].right

            # If we are on the top edge, this tile's top input comes from the systolic
            # array's top input. Otherwise the top input comes from the upper
            # neighboring tile.
            if row == 0:
                current_top = top[column]
            else:
                current_top = tile_outs[row - 1][column].bottom

            tile_in = TileIn(left=current_left, top=current_top)
            tile_out, accumulator = _make_systolic_array_tile(
                tile_in=tile_in, column=column, row=row, reset=reset
            )
            tile_outs[row][column] = tile_out
            accumulators[row][column] = accumulator

    product = wire_matrix_2d.WireMatrix2D(
        values=accumulators,
        shape=(num_rows, num_columns),
        bitwidth=accumulator_bitwidth,
        name=f"{name}.output",
    )

    # State machine.
    state = pyrtl.Register(
        bitwidth=max(state.value for state in State), name=f"{name}.state"
    )

    done_next_cycle = counter == done_cycle
    done_next_cycle.name = f"{name}.done_next_cycle"
    with pyrtl.conditional_assignment:
        # Reset the counter in INIT and READ states.
        with (state == State.INIT) | (state == State.READ):
            counter.next |= 0
        # Stop advancing the counter when the matrix multiplication is done.
        with done_next_cycle:
            counter.next |= counter
        # Otherwise, advance the counter.
        with pyrtl.otherwise:
            counter.next |= counter + 1

    with pyrtl.conditional_assignment:
        with state == State.INIT:
            b.ready |= True
            with b.valid:
                state.next |= State.READ
        with state == State.READ:
            state.next |= State.BUSY
        with (state == State.BUSY) & done_next_cycle:
            state.next |= State.DONE
        with state == State.DONE:
            product.valid |= True

    return product


def make_elementwise_add(
    name: str,
    a: wire_matrix_2d.WireMatrix2D,
    b: wire_matrix_2d.WireMatrix2D,
    output_bitwidth: int,
) -> wire_matrix_2d.WireMatrix2D:
    """Combinationally add matricies `a` and `b` elementwise.

    This implementation is entirely combinational (no registers).

    """
    assert a.shape == b.shape
    num_rows, num_columns = a.shape

    # Collect a 2D array of sums.
    sums = [[None for column in range(num_columns)] for row in range(num_rows)]

    for row in range(num_rows):
        for column in range(num_columns):
            sums[row][column] = pyrtl.signed_add(
                a[row][column], b[row][column]).truncate(output_bitwidth)

    # Combinational adder is always ready for input.
    a.ready <<= True
    b.ready <<= True
    # Combinational adder's output is valid when both inputs are valid.
    sums_matrix = wire_matrix_2d.WireMatrix2D(
        values=sums,
        shape=a.shape,
        bitwidth=output_bitwidth,
        name=f"{name}.output",
        valid=a.valid & b.valid,
    )
    return sums_matrix


def make_elementwise_relu(
    name: str, a: wire_matrix_2d.WireMatrix2D
) -> wire_matrix_2d.WireMatrix2D:
    """Combinationally relu matrix `a`. This computes max(a, 0) elementwise.

    This implementation is entirely combinational (no registers).

    """
    num_rows, num_columns = a.shape

    # Collect a 2D array of relu outputs.
    outputs = [[None for column in range(num_columns)] for row in range(num_rows)]

    for row in range(num_rows):
        for column in range(num_columns):
            current_a = a[row][column]
            outputs[row][column] = pyrtl.select(
                pyrtl.signed_ge(current_a, 0), current_a, 0
            )

    # Combinational relu is always ready for input.
    a.ready <<= True
    # Combinational relu's output is valid when its input is valid.
    outputs_matrix = wire_matrix_2d.WireMatrix2D(
        values=outputs,
        shape=a.shape,
        bitwidth=a.bitwidth,
        name=f"{name}.output",
        valid=a.valid,
    )
    return outputs_matrix


def make_elementwise_normalize(
    name: str,
    a: wire_matrix_2d.WireMatrix2D,
    m0: Fxp,
    n: np.ndarray,
    z3: np.ndarray,
    input_bitwidth: int,
    output_bitwidth: int,
) -> wire_matrix_2d.WireMatrix2D:
    """Convert an un-normalized layer output to a normalized output.

    This function effectively multiplies the layer's output by its scale factor `m` and
    adds its zero point `z3`.

    `m` is a floating-point number, which is represented by a 32-bit fixed-point
    multiplier `m0` and bitwise right shift `n`, see numpy_inference.py's
    `normalization_constants()`. So instead of doing a floating-point multiplication, we
    do a fixed-point multiplication, followed by a bitwise right shift.

    See https://arxiv.org/pdf/1712.05877.pdf for more details. This implements the part
    of equation 7 that's outside the parentheses (addition of `z3` and multiplication by
    `m`).

    Layers can have per-axis scale factors, so `m0` and `n` will be vectors of scale
    factors and shift amounts. See
    https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor

    input_bitwidth is generally 32, and output_bitwidth is generally 8, so this
    multiplies and shifts 32-bit `a` values into 8-bit outputs, utilizing the 8-bit
    output range as effectively as possible.

    This implementation is entirely combinational (no registers).

    """
    assert input_bitwidth >= output_bitwidth
    num_rows, num_columns = a.shape

    assert len(m0) == len(n)
    if len(m0) == 1:
        # Per-tensor quantization: There is only one (m0, n) for the whole tensor. The
        # code below assumes per-axis quantization, so we copy the values.
        m0 = [m0[0] for _ in range(num_rows)]
        n = [n[0] for _ in range(num_rows)]
    assert len(m0) == num_rows

    # `m0` is always positive, so zero-extend it by one bit to ensure its high bit is
    # always zero. This ensures that the `signed_mult` below does not interpret `m0` as
    # a negative number.
    m0 = [pyrtl.Const(multiplier.val.item(), bitwidth=input_bitwidth + 1)
          for multiplier in m0]

    # Collect a 2D array of normalized `outputs`. Intermediate results are collected in
    # these additional `multiplied`, `round_up`, and `shifted` arrays to make them
    # easier to inspect while debugging.
    multiplied = [[None for column in range(num_columns)] for row in range(num_rows)]
    round_up = [[None for column in range(num_columns)] for row in range(num_rows)]
    shifted = [[None for column in range(num_columns)] for row in range(num_rows)]
    outputs = [[None for column in range(num_columns)] for row in range(num_rows)]

    assert len(z3) == 1
    z3 = pyrtl.Const(z3[0], signed=True)
    for row in range(num_rows):
        for column in range(num_columns):
            # Elementwise fixed-point multiply the input tensor by its per-axis `m0`
            # value. This multiplies two fixed-point 32-bit values, resulting in a
            # fixed-point 64-bit value. There may be an additional zero-valued high bit
            # because we zero-extended `m0` above.
            multiplied[row][column] = pyrtl.signed_mult(a[row][column], m0[row])

            # Rounding right shift by `input_bitwidth + n[row]`. Shifting by
            # `input_bitwidth` converts the multiplier output from a 64-bit value to a
            # 32-bit value. Shifting by `n[row]` shifts the output by its per-axis `n`
            # value, which drops all fractional bits, and results in a signed integer
            # with bitwidth between 8 and 32 bits.
            #
            # This rounding right shift drops all fractional bits. Fractions are rounded
            # to the nearest integer:
            #   100.4 -> 100
            #   100.5 -> 101
            #   -10.4 -> -10
            #   -10.5 -> -11
            #
            # `round_up` is the value of the most significant fractional bit (0.5).
            # `round_up` indicates if the fractional part is greater than or equal to
            # 0.5 for positive numbers. The value is two's complement encoded, so if the
            # value is negative, this bit will be inverted and indicate if the
            # fractional part is less than 0.5.
            #
            # The `round_up` bit must be `zero_extended` to two bits so the `signed_add`
            # below does not interpret it as a negative number.
            #
            # See
            # https://github.com/tensorflow/tensorflow/issues/25087#issuecomment-634262762
            # for more details.
            shift_amount = input_bitwidth + n[row]
            round_up[row][column] = (
                multiplied[row][column][shift_amount - 1: shift_amount]
                .zero_extended(2))
            shifted[row][column] = pyrtl.signed_add(
                multiplied[row][column][shift_amount:], round_up[row][column])

            # Elementwise add `z3`, then keep only the low 8 bits of the result. The
            # high bits may not all be zero, so this truncation may overflow.
            outputs[row][column] = pyrtl.signed_add(
                z3, shifted[row][column]).truncate(output_bitwidth)

    # Combinational normalize is always ready for input.
    a.ready <<= True
    outputs_matrix = wire_matrix_2d.WireMatrix2D(
        values=outputs,
        shape=a.shape,
        bitwidth=output_bitwidth,
        name=f"{name}.output",
        valid=a.valid,
    )
    return outputs_matrix


def make_argmax(a: wire_matrix_2d.WireMatrix2D) -> pyrtl.WireVector:
    """Combinationally argmax signed matrix a.

    This implementation is entirely combinational (no registers).

    """
    num_rows, num_columns = a.shape

    assert num_columns == 1
    assert num_rows > 0

    index_bitwidth = pyrtl.infer_val_and_bitwidth(num_rows).bitwidth

    # Combinational argmax is always ready for input.
    a.ready <<= True

    if num_rows == 1:
        return pyrtl.Const(val=0)

    @pyrtl.wire_struct
    class IndexedValue:
        """Pairs a value with its index in `a`."""

        value: a.bitwidth
        index: index_bitwidth

    indexed_values = [IndexedValue(value=a[index][0], index=index)
                      for index in range(num_rows)]

    def argmax2(a: IndexedValue, b: IndexedValue) -> IndexedValue:
        """Two-input argmax."""
        return IndexedValue(
            IndexedValue=pyrtl.select(pyrtl.signed_gt(a.value, b.value), a, b)
        )

    # Compose two-input argmaxes into a wider argmax that accepts `num_rows` inputs.
    argmax = argmax2(indexed_values[0], indexed_values[1])
    for index in range(2, num_rows):
        argmax = argmax2(argmax, indexed_values[index])

    return argmax.index


def inspect_matrix(
    sim: pyrtl.Simulation, prefix: str, shape: tuple, bitwidth: int, suffix=".output"
) -> np.ndarray:
    """Collect output values from a PyRTL Simulation and return them as a numpy matrix.

    """
    num_rows, num_columns = shape
    array = [[None for _ in range(num_columns)] for _ in range(num_rows)]
    for row in range(num_rows):
        for column in range(num_columns):
            array[row][column] = pyrtl.val_to_signed_integer(
                sim.inspect(f"{prefix}{suffix}[{row}][{column}]"), bitwidth=bitwidth)
    return np.array(array)


def minimum_bitwidth(a: np.ndarray) -> int:
    """Return the number of bits needed to represent all values in `a`.

    `a` may contain negative numbers, so this ensures there are enough bits to represent
    both the largest and smallest values.

    """
    max_bitwidth = pyrtl.infer_val_and_bitwidth(np.max(a), signed=True).bitwidth
    min_bitwidth = pyrtl.infer_val_and_bitwidth(np.min(a), signed=True).bitwidth
    return max(max_bitwidth, min_bitwidth)


def verify_tensor(name: str, expected: np.ndarray, actual: np.ndarray):
    """Compare `expected` to `actual` and print a report."""
    if np.logical_and.reduce(expected == actual, axis=None):
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
    parser.add_argument("--x_start", type=int, default=-128)
    parser.add_argument("--y_shape", type=int, default=(3, 4))
    parser.add_argument("--y_start", type=int, default=-64)
    parser.add_argument("--a_shape", type=int, nargs=2, default=(2, 4))
    parser.add_argument("--a_start", type=int, default=1)
    args = parser.parse_args()

    def make_np_matrix(shape: tuple[int, int], start: int):
        """Return an integer matrix with the specified shape.

        The matrix will be filled with increasing integers starting from `start`.

        """
        num_rows, num_columns = shape
        array = np.array(list(range(start, start + num_rows * num_columns)))
        return np.reshape(array, newshape=shape)

    x = make_np_matrix(args.x_shape, start=args.x_start)
    y = make_np_matrix(args.y_shape, start=args.y_start)
    a = make_np_matrix(args.a_shape, start=args.a_start)

    assert x.shape[1] == y.shape[0]

    y_zero = 1
    expected_xy = x @ (y - y_zero)
    assert expected_xy.shape == a.shape

    input_bitwidth = max(
        [minimum_bitwidth(a) for a in [x, y, expected_xy]]
    )
    accumulator_bitwidth = 32

    num_rows, num_inner = x.shape
    _, num_columns = y.shape
    done_cycle = num_rows + num_inner + num_columns - 1
    counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

    y_memblock = pyrtl.MemBlock(
        addrwidth=counter_bitwidth, bitwidth=input_bitwidth * num_columns
    )
    y_valid = pyrtl.Input(name="y_valid", bitwidth=1)
    matrix_y = wire_matrix_2d.WireMatrix2D(
        values=y_memblock,
        shape=y.shape,
        bitwidth=input_bitwidth,
        name="y",
        valid=y_valid,
    )

    matrix_xy = make_systolic_array(
        "mm0",
        x,
        matrix_y,
        y_zero,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )
    matrix_a = wire_matrix_2d.WireMatrix2D(
        values=a, bitwidth=input_bitwidth, name="a", valid=True
    )
    matrix_xya = make_elementwise_add(
        name="add0", a=matrix_xy, b=matrix_a, output_bitwidth=accumulator_bitwidth
    )
    matrix_xya.ready <<= True

    # Simulate the systolic array by providing inputs for each cycle.
    data_dict = {
        i: d
        for i, d in enumerate(
            make_input_romdata(y.transpose(), input_bitwidth, counter_bitwidth)
        )
    }
    sim = pyrtl.Simulation(memory_value_map={y_memblock: data_dict})
    sim.step(provided_inputs={"y_valid": False})
    sim.step(provided_inputs={"y_valid": False})
    sim.step(provided_inputs={"y_valid": False})
    while not sim.inspect("add0.output.valid"):
        sim.step(provided_inputs={"y_valid": True})
    sim.step(provided_inputs={"y_valid": True})

    # Simulation complete, print the waveform.
    def render_trace(prefixes):
        # Only show traces with the maximum number of brackets. This will display
        # `output[1][2]`, and skip `output[1]` and `output`.
        def count_brackets(name):
            return name.count("[")

        def strip_brackets(name):
            return name.split("[")[0]

        trace_counts = [
            (strip_brackets(name), count_brackets(name))
            for name in list(sorted(sim.tracer.trace.keys()))
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
                "mm0.state": pyrtl.enum_name(State),
                "mm1.state": pyrtl.enum_name(State),
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

    actual_xy = inspect_matrix(sim, "mm0", expected_xy.shape, accumulator_bitwidth)
    verify_tensor("x ⋅ y", expected_xy, actual_xy)

    print("\nComputing x ⋅ y + a")
    print(f"a shape={a.shape}:\n{a}")
    render_trace(
        prefixes=[
            "mm0.output[",
            "a[",
            "add0.output[",
            "mm0.output.valid",
            "add0.output.valid",
        ]
    )

    expected_xya = expected_xy + a
    actual_xya = inspect_matrix(
        sim, "add0", expected_xy.shape, bitwidth=accumulator_bitwidth
    )
    verify_tensor("x ⋅ y + a", expected_xya, actual_xya)


if __name__ == "__main__":
    main()
