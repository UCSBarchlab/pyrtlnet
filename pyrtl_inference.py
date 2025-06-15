import argparse
import time
import shutil

from ai_edge_litert.interpreter import Interpreter
import inference_util
import mnist_util
import numpy as np
import numpy_inference
import pyrtl
import matrix
import wire_matrix_2d

class PyRTLInference:
    """Run quantized inference with PyRTL."""
    def __init__(self, interpreter: Interpreter,
                 input_bitwidth: int, accumulator_bitwidth):
        """Build the PyRTL inference hardware."""

        # Extract weights, biases, and quantization metadata from the LiteRT
        # Interpreter. The Interpreter is not actually used for inference.
        self.interpreter = interpreter
        self.input_bitwidth = input_bitwidth
        self.accumulator_bitwidth = accumulator_bitwidth

        # Tensor metadata, from the Model Explorer
        # (https://github.com/google-ai-edge/model-explorer):
        #
        # tensor 0: input          int8[1, 12, 12]
        #
        # tensor 1: reshape shape  int32[2]
        # tensor 2: reshape output int8[1, 144]
        #
        # tensor 3: layer 0 weight int8[18, 144]
        # tensor 4: layer 0 bias   int32[18]
        # tensor 5: layer 0 output int8[1, 18]
        #
        # tensor 6: layer 1 weight int8[10, 18]
        # tensor 7: layer 1 bias   int32[10]
        # tensor 8: layer 1 output int8[1, 10]
        self.input_scale, self.input_zero = numpy_inference.get_tensor_scale_zero(
            interpreter=interpreter, tensor_index=2)

        # Extract weights, biases, and quantization metadata from the TFLite model.
        layer0 = numpy_inference.QuantizedLayer(
            interpreter=interpreter,
            input_scale=self.input_scale,
            weight_index=3,
            bias_index=4,
            output_index=5,
        )
        layer1 = numpy_inference.QuantizedLayer(
            interpreter=interpreter,
            input_scale=layer0.scale,
            weight_index=6,
            bias_index=7,
            output_index=8,
        )
        self.layer = [layer0, layer1]

        # Create the MemBlock for the input image data.
        self._make_input_memblock()

        # Create hardware that implements the neural network.
        self._make_inference()


    def _make_input_memblock(self):
        """Build the MemBlock that will hold the input image data."""
        weight_shape = self.layer[0].weight.shape
        batch_size = 1
        flat_image_shape = (weight_shape[1], batch_size)

        done_cycle = (
            matrix.num_systolic_array_cycles(weight_shape, flat_image_shape) - 1)
        self.flat_image_addrwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

        # Create a properly-sized empty MemBlock. The MemBlock's contents will be set
        # at simulation time in `run()`.
        _, num_columns = flat_image_shape
        self.flat_image_memblock = pyrtl.MemBlock(
            addrwidth=self.flat_image_addrwidth, bitwidth=self.input_bitwidth * num_columns
        )

        # Create a WireMatrix2D that wraps the empty MemBlock. This will be the first
        # layer's input.
        self.flat_image_matrix = wire_matrix_2d.WireMatrix2D(
            values=self.flat_image_memblock,
            shape=flat_image_shape,
            bitwidth=self.input_bitwidth,
            name="flat_image",
            valid=True,
        )


    def _make_layer(
            self, layer_num: int, input: wire_matrix_2d.WireMatrix2D,
            input_zero: np.ndarray, relu: bool) -> wire_matrix_2d.WireMatrix2D:
        """Build one layer of the PyRTL inference hardware."""
        layer_name = f"layer{layer_num}"

        # Matrix multiplication with 8-bit inputs and 32-bit output. This multiplies
        # the layer's weight and the layer's input data.
        product = matrix.make_systolic_array(
            name=layer_name + "_matmul",
            a=self.layer[layer_num].weight,
            b=input,
            b_zero=input_zero,
            input_bitwidth=self.input_bitwidth,
            accumulator_bitwidth=self.accumulator_bitwidth,
        )

        # Create a WireMatrix2D for the layer's bias.
        bias_matrix = wire_matrix_2d.WireMatrix2D(
            values=self.layer[layer_num].bias,
            bitwidth=matrix.minimum_bitwidth(self.layer[layer_num].bias),
            name=layer_name + "_bias",
            valid=True,
        )
        # Add the bias. This is a 32-bit add.
        sum = matrix.make_elementwise_add(
            name=layer_name + "_add",
            a=product,
            b=bias_matrix,
            output_bitwidth=self.accumulator_bitwidth,
        )

        # Perform ReLU, if the layer needs it. This is a 32-bit ReLU.
        if relu:
            relu = matrix.make_elementwise_relu(
                name=layer_name + "_relu", a=sum
            )
        else:
            relu = sum

        # Normalize from 32-bit to 8-bit. This effectively multiplies the layer's output
        # by its scale factor `m` and adds its zero point `z3`. `m` is represented as
        # a fixed-point multiplier `m0` and a right shift `n`.
        output = matrix.make_elementwise_normalize(
            name=layer_name,
            a=relu,
            m0=self.layer[layer_num].m0,
            n=self.layer[layer_num].n,
            z3=self.layer[layer_num].zero,
            input_bitwidth=self.accumulator_bitwidth,
            output_bitwidth=self.input_bitwidth,
        )

        # Create a WireMatrix2D for this layer's output. This can be used as the next
        # layer's input.
        wire_matrix_2d.make_outputs(output)

        return output


    def _make_inference(self):
        """Build the PyRTL inference hardware."""
        # Build all layers.
        layer0 = self._make_layer(
            layer_num=0, input=self.flat_image_matrix, input_zero=self.input_zero,
            relu=True)

        layer1 = self._make_layer(
            layer_num=1, input=layer0, input_zero=self.layer[0].zero,
            relu=False)

        # Compute argmax for the last layer's output.
        argmax = matrix.make_argmax(a=layer1)

        num_rows, num_columns = layer1.shape
        assert num_columns == 1

        argmax_output = pyrtl.Output(
            name="argmax", bitwidth=pyrtl.infer_val_and_bitwidth(num_rows).bitwidth
        )
        argmax_output <<= argmax

        # Make a PyRTL Output for the second layer output's `valid` signal. When this
        # signal goes high, inference is complete.
        valid = pyrtl.Output(name="valid", bitwidth=1)
        valid <<= layer1.valid


    def run(self, test_image: np.ndarray) -> (np.ndarray, np.ndarray, int):
        """Run quantized inference on a single image.

        All calculations are done in PyRTL Simulation.

        Returns (layer0_output, layer1_output, predicted_digit), where:

        * `layer0_output` is the first layer's raw Tensor output (shape (18, 1)).
        * `layer1_output` is the second layer's raw Tensor output (shape (10, 1)).
        * `predicted_digit` is the actual predicted digit. It is equivalent to
          `layer1_output.flatten().argmax()`.

        """
        layer0_weight_shape = self.layer[0].weight.shape
        batch_size = 1
        input_shape = (layer0_weight_shape[1], batch_size)
        layer0_output_shape = (layer0_weight_shape[0], batch_size)

        layer1_weight_shape = self.layer[1].weight.shape
        layer1_output_shape = (layer1_weight_shape[0], batch_size)

        # The MNIST image data contains pixel values in the range [0, 255]. The neural
        # network was trained by first converting these values to floating point, in the
        # range [0, 1.0]. Dividing by input_scale below undoes this conversion,
        # converting the range from [0, 1.0] back to [0, 255].
        #
        # We could avoid these back-and-forth conversions by modifying
        # `load_mnist_images()` to skip the first conversion, and returning `x +
        # input_zero_point` below to skip the second conversion, but we do them anyway
        # to simplify the code and make it more consistent with existing sample code
        # like https://ai.google.dev/edge/litert/models/post_training_integer_quant
        #
        # Adding input_zero_point (-128) effectively converts the uint8 image data to
        # int8, by shifting the range [0, 255] to [-128, 127].
        flat_image = np.reshape(
            test_image / self.input_scale + self.input_zero, newshape=input_shape
        ).astype(np.int8)

        # Convert the flattened image data to a dictionary for use in Simulation's
        # `memory_value_map`. The `flat_image` is transposed because this data will be
        # the second input to the first layer's systolic array (`top` inputs to the
        # array).
        image_data_dict = {
            i: d
            for i, d in enumerate(
                matrix.make_input_romdata(
                    flat_image.transpose(), self.input_bitwidth,
                    self.flat_image_addrwidth
                )
            )
        }

        # Run until the second layer's computations are done.
        sim = pyrtl.FastSimulation(
            memory_value_map={self.flat_image_memblock: image_data_dict})
        done = False
        while not done:
            sim.step()
            done = sim.inspect("valid")

        # Retrieve each layer's outputs.
        layer0_output = matrix.inspect_matrix(
            sim, "output_layer0", layer0_output_shape, bitwidth=self.input_bitwidth
        )
        layer1_output = matrix.inspect_matrix(
            sim, "output_layer1", layer1_output_shape, bitwidth=self.input_bitwidth
        )

        # Retrieve the predicted digit.
        argmax = sim.inspect("argmax")

        return layer0_output, layer1_output, argmax


def main():
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST data set.
    _, (test_images, test_labels) = mnist_util.load_mnist_images()

    # Load quantized model.
    tflite_file = "quantized.tflite"
    interpreter = Interpreter(model_path=tflite_file)

    # Create PyRTL inference hardware.
    input_bitwidth = 8
    accumulator_bitwidth = 32
    pyrtl_inference = PyRTLInference(interpreter, input_bitwidth, accumulator_bitwidth)

    correct = 0
    for test_index in range(args.start_image, args.start_image + args.num_images):
        # Print the test image.
        test_image = test_images[test_index]
        print(f"PyRTL network input (#{test_index}):")
        inference_util.display_image(test_image)

        # Run PyRTL inference on the test image.
        layer0_output, layer1_output, actual = pyrtl_inference.run(test_image)

        # Print results.
        print("PyRTL layer0 output (transposed)",
              layer0_output.shape, layer0_output.dtype)
        print(layer0_output.T, "\n")

        print("PyRTL layer1 output (transposed)",
              layer1_output.shape, layer1_output.dtype)
        print(layer1_output.T, "\n")

        print(f"\nPyRTL network output (#{test_index}):")
        expected = test_labels[test_index]
        inference_util.display_outputs(layer1_output, expected=expected, actual=actual)

        if actual == expected:
            correct += 1

        if test_index < args.num_images - 1:
            print()

    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
