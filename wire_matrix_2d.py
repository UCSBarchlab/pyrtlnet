import numpy as np
import pyrtl


class WireMatrix2D:
    """WireMatrix2D is a 2D wrapper around pyrtl.wire_matrix.

    WireMatrix2D serves as the input and output type for all operations in `matrix.py`.
    These matrix operations can be composed. For example, when computing x ⋅ y + a,
    there is an intermediate WireMatrix2D that serves as both the output of the
    multiplication, and the input to the addition.

    WireMatrix2D provides several useful fields and methods:

    * `self.shape` is the matrix's shape, as a pair of integers.
    * `self.ready` is a 1-bit WireVector indicating if the downstream operation is ready
      for input.
    * `self.valid` is a 1-bit WireVector indicating if the upstream operation has
      finished writing its output.
    * `self.bitwidth` is the bitwidth of each element in the matrix.
    * `self[i][j]` accesses the `(i, j)`th element of the matrix.

    """
    def __init__(
        self,
        values,
        shape: tuple[int, int] = (),
        bitwidth: int = 0,
        name: str = "",
        ready=None,
        valid=None,
    ):
        """Construct a 2D wire_matrix containing ``values``.

        ``values`` is one of the following:
        - None. Creates a WireMatrix2D of Inputs.
        - A Numpy ndarray. ``shape`` will be inferred from the ndarray.
        - A list of lists of WireVectors, with shape ``shape``.
        - a concatenated WireVector representing all values in the matrix.
        - a MemBlock containing the input data.

        The 2D matrix has shape ``shape`` and bitwidth ``bitwidth``.

        WireMatrix2D also provides `ready` and `valid` signals.

        """
        if isinstance(values, np.ndarray):
            shape = values.shape

        assert len(shape) == 2
        self.shape = shape
        self.bitwidth = bitwidth
        self.name = name

        num_rows, num_columns = self.shape
        self.Row = pyrtl.wire_matrix(component_schema=bitwidth, size=num_columns)
        self.Matrix = pyrtl.wire_matrix(component_schema=self.Row, size=num_rows)

        self.matrix = None
        self.memblock = None
        if values is None:
            # Creating a WireMatrix2D of Inputs.
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
            # Creating a WireMatrix2D from a concatenated WireVector containing all
            # values.
            self.matrix = self.Matrix(name=name, values=[values])
        elif isinstance(values, pyrtl.MemBlock):
            # Creating a WireMatrix2D from a MemBlock containing all values.
            self.memblock = values
        else:
            # Creating a WireMatrix2D of constants from an ndarray or list of lists.
            assert len(values) == num_rows
            rows = []
            for row in values:
                assert len(row) == num_columns
                if isinstance(row, np.ndarray):
                    row = [value.item() for value in row]
                rows.append(self.Row(values=row))
            self.matrix = self.Matrix(name=name, values=rows)

        def create_ready_valid(value, suffix: str) -> pyrtl.WireVector:
            """Generate a 1-bit ready or valid wire, if necessary."""
            output_name = ""
            if not name == "":
                output_name = f"{name}{suffix}"
            if value is None:
                output = pyrtl.WireVector(name=output_name, bitwidth=1)
            else:
                output = pyrtl.as_wires(value)
                assert output.bitwidth == 1
                if output.name.startswith("tmp") or output.name.startswith("const_"):
                    output.name = output_name
            return output

        self.ready = create_ready_valid(value=ready, suffix=".ready")
        self.valid = create_ready_valid(value=valid, suffix=".valid")

    def __getitem__(self, key):
        """Implements WireMatrix2D's [] operator."""
        if self.memblock is not None:
            return self.memblock[key]
        else:
            return self.matrix[key]

    def transpose(self):
        """Return a transposed version of `self`, as another WireMatrix2D.

        WARNING: If `self.memblock` is not `None`, this does not reformat the MemBlock
        data. It only changes the shape; the MemBlock is assumed to already contain
        correctly formatted data.

        """
        num_rows, num_columns = self.shape
        if self.memblock is not None:
            outputs = self.memblock
        else:
            # Collect a 2D array of transposed outputs.
            outputs = [[None for column in range(num_rows)]
                       for row in range(num_columns)]

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


def make_provided_inputs(name, image):
    num_rows, num_columns = image.shape

    output = {}
    for row in range(num_rows):
        for column in range(num_columns):
            output[f"{name}[{row}][{column}]"] = image[row][column]
    return output


def make_outputs(matrix):
    num_rows, num_columns = matrix.shape

    for row in range(num_rows):
        for column in range(num_columns):
            output = pyrtl.Output(
                name=f"output_{row}_{column}", bitwidth=matrix.bitwidth
            )
            output <<= matrix[row][column]


def main():
    pass


if __name__ == "__main__":
    main()
