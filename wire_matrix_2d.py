import numpy as np
import pyrtl


class WireMatrix2D:
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
        - a MemBlock containing the input data. The matrix's shape must be (N, 1).

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
            self.matrix = self.Matrix(name=name, values=[values])
        elif isinstance(values, pyrtl.MemBlock):
            self.memblock = values
        else:
            assert len(values) == num_rows
            rows = []
            for row in values:
                assert len(row) == num_columns
                if isinstance(row, np.ndarray):
                    row = [value.item() for value in row]
                rows.append(self.Row(values=row))
            self.matrix = self.Matrix(name=name, values=rows)

        ready_name = ""
        if not name == "":
            ready_name = f"{name}.ready"
        if ready is None:
            self.ready = pyrtl.WireVector(name=ready_name, bitwidth=1)
        else:
            self.ready = pyrtl.as_wires(ready)
            assert self.ready.bitwidth == 1
            if self.ready.name.startswith("tmp") or self.ready.name.startswith(
                "const_"
            ):
                self.ready.name = ready_name

        valid_name = ""
        if not name == "":
            valid_name = f"{name}.valid"
        if valid is None:
            self.valid = pyrtl.WireVector(name=valid_name, bitwidth=1)
        else:
            self.valid = pyrtl.as_wires(valid)
            assert self.valid.bitwidth == 1
            if self.valid.name.startswith("tmp") or self.valid.name.startswith(
                "const_"
            ):
                self.valid.name = valid_name

    def __getitem__(self, key):
        if self.memblock is not None:
            return self.memblock[key]
        else:
            return self.matrix[key]

    def transpose(self):
        num_rows, num_columns = self.shape
        # Collect a 2D array of transposed outputs.
        outputs = [[None for column in range(num_rows)] for row in range(num_columns)]

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
