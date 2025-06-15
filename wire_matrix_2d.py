import numpy as np
import pyrtl


class WireMatrix2D:
    """``WireMatrix2D`` represents a 2D matrix of PyRTL ``WireVectors``.

    It functions like a 2D ``pyrtl.wire_matrix`` and serves as the input and output type
    for all operations in ``matrix.py``. These matrix operations can be composed. For
    example, when computing x ⋅ y + a, there is an intermediate ``WireMatrix2D`` that
    serves as both the output of the multiplication, and the input to the addition.

    ``WireMatrix2D`` provides several useful fields and methods:

    * ``self.shape`` is the matrix's shape, as a pair of integers.
    * ``self.ready`` is a 1-bit ``WireVector`` indicating if the downstream operation is
      ready for input.
    * ``self.valid`` is a 1-bit ``WireVector`` indicating if the upstream operation has
      finished writing its output.
    * ``self.bitwidth`` is the bitwidth of each element in the matrix.
    * ``self[i][j]`` accesses the ``(i, j)``th element of the matrix.

    ``WireMatrix2D`` supports two underlying representations:

    1. ``self.Matrix``, which is a 2D ``pyrtl.wire_matrix``. ``pyrtl.wire_matrix``
       supports any PyRTL ``WireVector`` type, so you could have a ``self.Matrix`` of
       ``pyrtl.Register`` for example.
    2. ``pyrtl.MemBlock``, where the matrix data is stored in a ``pyrtl.MemBlock`` or
       ``pyrtl.RomBlock``. This representation is currently experimental and not
       completely supported.

    """

    def __init__(
        self,
        values: np.ndarray
        | list[list[pyrtl.WireVector]]
        | pyrtl.WireVector
        | pyrtl.MemBlock,
        shape: (int, int) = (),
        bitwidth: int = 0,
        name: str = "",
        ready=None,
        valid=None,
    ):
        """Construct a 2D wire_matrix containing ``values``.

        ``values`` must be one of the following:
        - ``None``. Creates a ``WireMatrix2D`` of ``pyrtl.Input``.
        - A NumPy ndarray. ``shape`` will be inferred from the ndarray.
        - A list of lists of ``WireVectors``, with shape ``shape``.
        - A concatenated ``WireVector`` representing all values in the matrix.
        - A ``MemBlock`` containing all values in the matrix. This is EXPERIMENTAL!

        The 2D matrix has shape ``shape`` and each element has bitwidth ``bitwidth``.

        ``WireMatrix2D`` also provides ``ready`` and ``valid`` signals. ``ready`` and
        ``valid`` can be any valid input to ``pyrtl.as_wires``.

        """
        if isinstance(values, np.ndarray):
            shape = values.shape

        assert len(shape) == 2
        self.shape = shape
        self.bitwidth = bitwidth
        self.name = name

        num_rows, num_columns = self.shape
        # Row is the type for each row in the 2D matrix. Row is just a 1D array of
        # elements.
        self.Row = pyrtl.wire_matrix(component_schema=bitwidth, size=num_columns)
        # Matrix is the actual type for the matrix data. Matrix is a 1D array of Rows.
        self.Matrix = pyrtl.wire_matrix(component_schema=self.Row, size=num_rows)

        self.matrix = None
        self.memblock = None
        if values is None:
            # Create a Matrix of Inputs.
            rows = []
            for row_index in range(num_rows):
                row = self.Row(component_type=pyrtl.Input)
                for column in range(num_columns):
                    row[column].name = f"{name}[{row_index}][{column}]"
                rows.append(row)
            self.matrix = self.Matrix(values=rows)
            # Inputs are always valid.
            valid = True
        elif isinstance(values, pyrtl.WireVector):
            # Create a Matrix from a concatenated WireVector containing all matrix
            # values.
            self.matrix = self.Matrix(name=name, values=[values])
        elif isinstance(values, pyrtl.MemBlock):
            # Create a WireMatrix2D from a MemBlock containing all matrix values.
            self.memblock = values
        else:
            # Create a Matrix of pyrtl.Const from an ndarray or list of lists.
            assert len(values) == num_rows
            rows = []
            for row in values:
                assert len(row) == num_columns
                if isinstance(row, np.ndarray):
                    row = [value.item() for value in row]
                rows.append(self.Row(values=row))
            self.matrix = self.Matrix(name=name, values=rows)

        def create_ready_valid(value, suffix: str) -> pyrtl.WireVector:
            """Return a 1-bit ready or valid wire. Create one if necessary."""
            output_name = ""
            if name != "":
                output_name = f"{name}{suffix}"
            if value is None:
                output = pyrtl.WireVector(name=output_name, bitwidth=1)
            else:
                output = pyrtl.as_wires(value)
                assert output.bitwidth == 1
                if output.name.startswith("tmp") or output.name.startswith("const_"):
                    output.name = output_name
            return output

        # Set up ready and valid signals.
        self.ready = create_ready_valid(value=ready, suffix=".ready")
        self.valid = create_ready_valid(value=valid, suffix=".valid")

    def __getitem__(self, key):
        """Implements WireMatrix2D's [] operator.

        If ``matrix`` is a WireMatrix2D, its elements can be accessed with
        ``matrix[row][column]``.

        WARNING: If ``self.MemBlock`` is not ``None``, this can currently only retrieve
        a full row of values.

        """
        if self.memblock is not None:
            return self.memblock[key]
        else:
            return self.matrix[key]

    def transpose(self):
        """Return a transposed version of ``self``, as another WireMatrix2D.

        WARNING: If ``self.memblock`` is not ``None``, this does not reformat the
        MemBlock data. It only changes the shape; the MemBlock is assumed to already
        contain correctly formatted data.

        """
        num_rows, num_columns = self.shape
        if self.memblock is not None:
            outputs = self.memblock
        else:
            # Collect a 2D array of transposed outputs.
            outputs = [
                [None for column in range(num_rows)] for row in range(num_columns)
            ]

            for row in range(num_rows):
                for column in range(num_columns):
                    outputs[column][row] = self[row][column]

        outputs_matrix = WireMatrix2D(
            values=outputs,
            shape=(num_columns, num_rows),
            bitwidth=self.bitwidth,
            name=f"{self.name}.transposed",
            ready=self.ready,
            valid=self.valid,
        )
        return outputs_matrix


def make_provided_inputs(name: str, values: np.ndarray) -> {str: int}:
    """Create a ``provided_inputs`` dict for use with ``pyrtl.Simulation.step``.

    The returned dict contains matrix data from ``values``.

    """
    assert len(values.shape) == 2
    num_rows, num_columns = values.shape

    output = {}
    for row in range(num_rows):
        for column in range(num_columns):
            output[f"{name}[{row}][{column}]"] = values[row][column]
    return output


def make_concatenated_value(values: np.ndarray, bitwidth: int) -> int:
    """Pack all elements of ``values`` into a large integer.

    This can be useful for setting a ``pyrtl.RomBlock``'s value, or setting the initial
    value of a ``pyrtl.MemBlock``.

    """
    assert len(values.shape) == 2
    num_rows, num_columns = values.shape

    output = 0
    input_mask = -1 & (2**bitwidth - 1)
    for row in range(num_rows):
        for column in range(num_columns):
            output = output << bitwidth
            # `output` may be very large, easily exceeding 64 bits. The `int()` below
            # ensures we do these computations with Python's arbitrary precision
            # integers, rather than a fixed-width type like np.int64.
            output |= int(values[row][column]) & input_mask
    return int(output)


def make_outputs(matrix: WireMatrix2D):
    """Create ``pyrtl.Output``s for a ``WireMatrix2D``.

    These ``Output``s can be inspected with ``inspect_matrix``.

    """
    num_rows, num_columns = matrix.shape

    for row in range(num_rows):
        for column in range(num_columns):
            output = pyrtl.Output(
                name=f"output_{matrix.name}[{row}][{column}]", bitwidth=matrix.bitwidth
            )
            output <<= matrix[row][column]


def inspect_matrix(sim: pyrtl.Simulation, matrix: WireMatrix2D) -> np.ndarray:
    """Collect Output values from a Simulation and return them as a NumPy matrix.

    The collected Output values should be created by ``make_outputs``.

    """
    num_rows, num_columns = matrix.shape
    array = [[None for _ in range(num_columns)] for _ in range(num_rows)]
    for row in range(num_rows):
        for column in range(num_columns):
            array[row][column] = pyrtl.val_to_signed_integer(
                sim.inspect(f"output_{matrix.name}[{row}][{column}]"),
                bitwidth=matrix.bitwidth,
            )
    return np.array(array)
