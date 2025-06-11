import argparse
import time
import shutil

from ai_edge_litert.interpreter import Interpreter
import inference_util
import numpy as np
import numpy_inference
import pyrtl
import tensorflow as tf
from tqdm import tqdm

import matrix
import wire_matrix_2d


def make_outputs(m: wire_matrix_2d.WireMatrix2D):
    num_rows, num_columns = m.shape
    for i in range(num_rows):
        for j in range(num_columns):
            wire = m[i][j]
            output = pyrtl.Output(name="output_" + wire.name, bitwidth=wire.bitwidth)
            output <<= wire


def make_layer(
    layer_num,
    input,
    input_zero,
    weight,
    bias,
    relu: bool,
    output_m0,
    output_n,
    output_zero,
    input_bitwidth,
    accumulator_bitwidth,
):
    layer_name = f"layer{layer_num}"

    # Matrix multiplication with 8-bit inputs and 32-bit outputs.
    product = matrix.make_systolic_array(
        name=layer_name + "_matmul",
        a=weight,
        b=input,
        b_zero=input_zero,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )

    # Add bias (32-bit).
    bias_matrix = wire_matrix_2d.WireMatrix2D(
        values=bias,
        bitwidth=matrix.minimum_bitwidth(bias),
        name=layer_name + "_bias",
        valid=True,
    )
    sum = matrix.make_elementwise_add(
        name=layer_name + "_add",
        a=product,
        b=bias_matrix,
        output_bitwidth=accumulator_bitwidth,
    )

    # ReLU (32-bit).
    if relu:
        relu = matrix.make_elementwise_relu(
            name=layer_name + "_relu", a=sum
        )
    else:
        relu = sum

    # Normalize from 32-bit to 8-bit.
    output = matrix.make_elementwise_normalize(
        name=layer_name,
        a=relu,
        m0=output_m0,
        n=output_n,
        z3=output_zero,
        input_bitwidth=accumulator_bitwidth,
        output_bitwidth=input_bitwidth,
    )
    make_outputs(output)

    valid = pyrtl.Output(name="output_" + layer_name + ".output.valid", bitwidth=1)
    valid <<= output.valid

    return output


def reset_registers(simulation: pyrtl.FastSimulation):
    block = simulation.block
    for reg in block.wirevector_subset(pyrtl.Register):
        reset_value = reg.reset_value if reg.reset_value else 0
        simulation.regs[reg.name] = reset_value


def make_inference(
    flat_image_matrix: wire_matrix_2d.WireMatrix2D,
    make_outputs: bool,
    input_bitwidth: int,
    accumulator_bitwidth: int,
):
    # Load quantized model.
    tflite_file = "quantized.tflite"
    interpreter = Interpreter(model_path=tflite_file)
    tensors = interpreter.get_tensor_details()
    input_scale, input_zero = numpy_inference.get_tensor_scale_zero(
        interpreter=interpreter, tensor_index=2)

    # Extract weights, biases, and quantization metadata from the TFLite model.
    layer0 = numpy_inference.QuantizedLayer(
        interpreter=interpreter,
        input_scale=input_scale,
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
    layer = [layer0, layer1]

    layer0 = make_layer(
        layer_num=0,
        input=flat_image_matrix,
        input_zero=input_zero,
        weight=layer[0].weight,
        bias=layer[0].bias,
        relu=True,
        output_m0=layer[0].m0,
        output_n=layer[0].n,
        output_zero=layer[0].zero,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )

    layer1 = make_layer(
        layer_num=1,
        input=layer0,
        input_zero=layer[0].zero,
        weight=layer[1].weight,
        bias=layer[1].bias,
        relu=False,
        output_m0=layer[1].m0,
        output_n=layer[1].n,
        output_zero=layer[1].zero,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )

    argmax = matrix.make_argmax(a=layer1)

    if make_outputs:
        wire_matrix_2d.make_outputs(layer1)

        num_rows, num_columns = layer1.shape
        assert num_columns == 1

        argmax_output = pyrtl.Output(
            name="argmax", bitwidth=pyrtl.infer_val_and_bitwidth(num_rows).bitwidth
        )
        argmax_output <<= argmax

        valid = pyrtl.Output(name="valid", bitwidth=1)
        valid <<= layer1.valid

    return input_scale, input_zero, layer, layer1.valid, argmax


def main(start_image: int, num_images: int):
    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST data set.
    mnist = tf.keras.datasets.mnist
    _, (test_images, test_labels) = mnist.load_data()

    test_images = inference_util.preprocess_images(test_images)
    flat_shape = (test_images[0].shape[0] * test_images[0].shape[1], 1)
    num_rows, num_inner = (18, flat_shape[0])
    _, num_columns = flat_shape

    done_cycle = num_rows + num_inner + num_columns - 1
    counter_bitwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

    input_bitwidth = 8
    accumulator_bitwidth = 32
    flat_image_memblock = pyrtl.MemBlock(
        addrwidth=counter_bitwidth, bitwidth=input_bitwidth * num_columns
    )
    flat_image_matrix = wire_matrix_2d.WireMatrix2D(
        values=flat_image_memblock,
        shape=flat_shape,
        bitwidth=input_bitwidth,
        name="flat_image",
        valid=True,
    )

    input_scale, input_zero, layer, _, _ = make_inference(
        flat_image_matrix=flat_image_matrix,
        make_outputs=True,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
    )

    pyrtl.passes._remove_wire_nets(pyrtl.working_block())

    def num_cycles(a, b) -> int:
        """Return the number of cycles needed to multiply matrix a and matrix b."""
        num_rows, a_num_inner = a.shape
        b_num_inner, num_columns = b.shape
        assert a_num_inner == b_num_inner
        return num_rows + a_num_inner + num_columns + 2

    correct = 0
    for test_index in range(start_image, start_image + num_images):
        # Run inference on test_index.
        test_image = test_images[test_index]
        print(f"network input (#{test_index}):")
        inference_util.display_image(test_image)

        flat_image = np.reshape(
            (test_image / 255.0 / input_scale) + input_zero, newshape=flat_shape
        ).astype(np.int8)
        data_dict = {
            i: d
            for i, d in enumerate(
                matrix.make_input_romdata(
                    flat_image.transpose(), input_bitwidth, counter_bitwidth
                )
            )
        }
        sim = pyrtl.FastSimulation(memory_value_map={flat_image_memblock: data_dict})
        reset_registers(sim)

        # Simulate layer 0.
        with tqdm(
            total=num_cycles(layer[0].weight, flat_image) + 1,
            desc="Simulating layer0",
            unit="cycle",
        ) as progress:
            done = False
            while not done:
                sim.step()
                progress.update(1)
                done = sim.inspect("output_layer0.output.valid")

        # Check layer 0.
        expected_product0 = numpy_inference.quantized_matmul(
            layer[0].weight, 0, flat_image, input_zero
        )
        # actual_product0 = matrix.inspect_matrix(
        #     sim, "layer0_matmul", expected_product0.shape, bitwidth=32, suffix=".pe"
        # )
        # matrix.verify_tensor(
        #     name="layer0 product0",
        #     expected=expected_product0.T,
        #     actual=actual_product0.T,
        # )
        expected_sum0 = expected_product0 + layer[0].bias
        expected_relu0 = numpy_inference.relu(expected_sum0)
        expected_layer0_output = numpy_inference.normalize(
            expected_relu0, layer[0].m0, layer[0].n, layer[0].zero
        )
        actual_layer0_output = matrix.inspect_matrix(
            sim, "output_layer0", expected_layer0_output.shape, bitwidth=8
        )
        matrix.verify_tensor(
            name="layer0",
            expected=expected_layer0_output.T,
            actual=actual_layer0_output.T,
        )

        # Simulate layer 1.
        print()
        with tqdm(total=num_cycles(layer[1].weight, actual_layer0_output),
                  desc="Simulating layer1",
                  unit="cycle",
                  ) as progress:
            done = False
            while not done:
                sim.step()
                progress.update(1)
                done = sim.inspect("output_layer1.output.valid")

        # Check layer 1.
        expected_product1 = numpy_inference.quantized_matmul(
            layer[1].weight, 0, expected_layer0_output, layer[0].zero
        )
        expected_sum1 = expected_product1 + layer[1].bias
        expected_layer1_output = numpy_inference.normalize(
            expected_sum1, layer[1].m0, layer[1].n, layer[1].zero
        )
        actual_layer1_output = matrix.inspect_matrix(
            sim, "output_layer1", expected_layer1_output.shape, bitwidth=8
        )
        matrix.verify_tensor(
            name="layer1",
            expected=expected_layer1_output.T,
            actual=actual_layer1_output.T,
        )

        print(f"\npyrtl network output (#{test_index}):")
        actual_argmax = sim.inspect("argmax")
        inference_util.display_outputs(
            actual_layer1_output.reshape((10,)),
            expected=test_labels[test_index],
            actual=actual_argmax,
        )

        if actual_argmax == test_labels[test_index]:
            correct += 1

        if test_index < num_images - 1:
            print()

    if num_images > 1:
        print(
            f"{correct}/{num_images} correct predictions, "
            f"{100.0 * correct / num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    main(start_image=args.start_image, num_images=args.num_images)
