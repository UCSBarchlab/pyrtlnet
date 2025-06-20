import numpy as np
import pyrtl


class WireMatrix2D:
    """``WireMatrix2D`` represents a 2D matrix of :class:`~pyrtl.wire.WireVector`.

    It functions like a 2D :func:`~pyrtl.helperfuncs.wire_matrix` and serves as the
    input and output type for all operations in ``matrix.py``. These matrix operations
    can be composed. For example, when computing x ⋅ y + a, there is an intermediate
    ``WireMatrix2D`` that serves as both the output of the multiplication, and the input
    to the addition.

    ``WireMatrix2D`` provides several useful fields and methods:

    * ``self.shape`` is the matrix's shape, as a pair of integers.
    * ``self.ready`` is a 1-bit ``WireVector`` indicating if the downstream operation is
      ready for input.
    * ``self.valid`` is a 1-bit ``WireVector`` indicating if the upstream operation has
      finished writing its output.
    * ``self.bitwidth`` is the bitwidth of each element in the matrix.
    * ``self[i][j]`` accesses the ``(i, j)`` th element of the matrix.

    ``WireMatrix2D`` supports two underlying representations:

    1. ``self.Matrix``, which is a 2D ``wire_matrix``. ``wire_matrix`` supports any
       PyRTL ``WireVector`` type, so you could have a ``self.Matrix`` of
       :class:`~pyrtl.wire.Register` for example.
    2. ``MemBlock``, where the matrix data is stored in a
       :class:`~pyrtl.memory.MemBlock` or :class:`~pyrtl.memory.RomBlock`. This
       representation is currently experimental and not completely supported.

    """

    def __init__(
        self,
        values: np.ndarray
        | list[list[pyrtl.WireVector]]
        | pyrtl.WireVector
        | pyrtl.MemBlock,
        shape: tuple[int, int] = (),
        bitwidth: int = 0,
        name: str = "",
        ready=None,
        valid=None,
    ):
        """Construct a 2D :func:`~pyrtl.helperfuncs.wire_matrix` containing ``values``.

        :param values: Values for the ``WireMatrix2D``. If ``None``, creates a
            ``WireMatrix2D`` of :class:`~pyrtl.wire.Input`. ``values`` can also be a
            ``ndarray``, a list of lists of ``WireVector``, one large concatenated
            ``WireVector`` containing all the values for matrix, or a ``MemBlock``.
        :param shape: Shape of the ``WireMatrix2D``. Must be two dimensional. If
            ``values`` is a ``ndarray``, the shape will be inferred from the ``ndarray``
            and this ``shape`` argument can be omitted.
        :param bitwidth: The bitwidth of each element.
        :param name: Names for all elements in the ``WireMatrix2D`` will be generated
            based on this prefix. For example, if ``name`` is ``foo`` then the top left
            element will be named ``foo[0][0]``.
        :param ready: 1-bit signal indicating if the consumer of this ``WireMatrix2D``
            is ready to read the data in the matrix.
        :param valid: 1-bit signal indicating if the producer of this ``WireMatrix2D``
            has finished writing the data in the matrix.

        """
        if isinstance(values, np.ndarray):
            shape = values.shape

        # TODO: Generalize this class from 2D to N-dimensional.
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
            self.is_input = True
            rows = []
            for row_index in range(num_rows):
                row = self.Row(component_type=pyrtl.Input)
                for column in range(num_columns):
                    row[column].name = f"{name}[{row_index}][{column}]"
                rows.append(row)
            self.matrix = self.Matrix(values=rows)
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
                    rows.append(self.Row(values=[value.item() for value in row]))
                else:
                    rows.append(self.Row(values=row))
            self.matrix = self.Matrix(name=name, values=rows)

        if self.matrix is not None:
            # Replace the default names for the top-level WireVector and Row-level
            # WireVectors with autogenerated wire names. Autogenerated wire names are
            # omitted from traces by default. The top-level and Row-level traces are not
            # very useful for debugging, and they clutter the traces.
            self.matrix.name = pyrtl.wire.next_tempvar_name()
            for row in range(num_rows):
                self.matrix[row].name = pyrtl.wire.next_tempvar_name()

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

    def transpose(self) -> "WireMatrix2D":
        """Return a transposed version of ``self``, as another ``WireMatrix2D``.

        :returns: a transposed version of ``self``.

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

    def make_outputs(self):
        """Create :class:`~pyrtl.wire.Output` for ``self``.

        Use :meth:`.inspect` to retrieve these ``Output`` values.

        """
        num_rows, num_columns = self.shape

        for row in range(num_rows):
            for column in range(num_columns):
                output = pyrtl.Output(
                    name=f"output_{self.name}[{row}][{column}]", bitwidth=self.bitwidth
                )
                output <<= self[row][column]

    def inspect(self, sim: pyrtl.Simulation) -> np.ndarray:
        """Collect ``Output`` values from a ``Simulation`` and return them as a ndarray.

        Retrieves :class:`~pyrtl.wire.Output` values for ``self`` from a
        :class:`~pyrtl.simulation.Simulation`, and returns the retrieved values in a
        :class:`~numpy.ndarray`.

        Use :meth:`.make_outputs` to create the retrieved ``Output`` values.

        :param sim: ``Simulation`` to retrieve values from.
        :returns: Retrieved values as a ``ndarray``.

        """
        num_rows, num_columns = self.shape
        array = [[None for _ in range(num_columns)] for _ in range(num_rows)]
        for row in range(num_rows):
            for column in range(num_columns):
                array[row][column] = pyrtl.val_to_signed_integer(
                    sim.inspect(f"output_{self.name}[{row}][{column}]"),
                    bitwidth=self.bitwidth,
                )
        return np.array(array)

    def make_provided_inputs(self, values: np.ndarray) -> dict[str, int]:
        """Create a ``provided_inputs`` ``dict`` for use in ``Simulation``.

        This should only be used with a ``WireMatrix2D`` of :class:`~pyrtl.wire.Input`.
        This ``WireMatrix2D`` should have been constructed with ``values=None``.

        :param values: Values to pack into a ``provided_inputs`` :class:`dict`.

        :returns: A ``provided_inputs`` :class:`dict` that contains matrix data from
            ``values``. This :class:`dict` can be passed to
            :meth:`~pyrtl.simulation.Simulation.step`, and it will set the
            :class:`~pyrtl.wire.Input` for the ``WireMatrix2D`` to ``values``.

        """
        assert self.is_input
        assert len(values.shape) == 2
        assert self.shape == values.shape
        num_rows, num_columns = values.shape

        output = {}
        for row in range(num_rows):
            for column in range(num_columns):
                output[f"{self.name}[{row}][{column}]"] = values[row][column]
        return output


def make_concatenated_value(values: np.ndarray, bitwidth: int) -> int:
    """Pack all elements of ``values`` into a large integer, in row-major order.

    When using a :class:`.WireMatrix2D` with a :class:`~pyrtl.memory.MemBlock`, this
    function is useful for setting setting the initial value of a
    :class:`~pyrtl.memory.MemBlock` or :class:`~pyrtl.memory.RomBlock`.

    :param values: Values to concatenate.
    :param bitwidth: Bitwidth of each element.
    :returns: A large integer containing all bits from ``values``, concatenated together
        in row-major order. The total number of bits returned will be ``values.size *
        bitwidth``.

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
