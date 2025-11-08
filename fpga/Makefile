# This is a `Makefile` rather than `justfile` because we want incremental
# builds. For example, we only want to re-run `tensorflow_training.py` if the
# training scripts have been modified.

VIVADO_BATCH=vivado -mode batch -nolog -nojournal

all: pyrtlnet.bit pyrtlnet.hwh

# Run tensorflow_training.py to generate quantized weights and save a copy of
# the MNIST test image data.
quantized.npz mnist_test_data.npz: ../tensorflow_training.py ../pyrtlnet/inference_util.py ../pyrtlnet/mnist_util.py ../pyrtlnet/tensorflow_training.py
	uv run ../tensorflow_training.py

# Run pyrtl_inference.py to generate pyrtl_inference_axi.v.
pyrtl_inference_axi.v: ../pyrtl_inference.py ../pyrtlnet/pyrtl_inference.py ../pyrtlnet/pyrtl_matrix.py ../pyrtlnet/pyrtl_axi.py ../pyrtlnet/wire_matrix_2d.py ../pyrtlnet/inference_util.py quantized.npz
	uv run ../pyrtl_inference.py --axi --verilog --initial_delay_cycles=1

# Run Vivado to package pyrtl_inference_axi.v in a pyrtlnet IP block.
pyrtlnet_ip_repo/src/pyrtl_inference_axi.v: pyrtl_inference_axi.v pyrtlnet_ip.tcl
	$(VIVADO_BATCH) -source pyrtlnet_ip.tcl

# Run Vivado to instantiate the pyrtlnet IP block. Connect the IP block to the
# Zynq Processing System (PS) and AXI DMA. Generate a bitstream (.bit) and
# hardware handoff file (.hwh), and copy these files from the project directory
# to this fpga/ directory.
pyrtlnet.bit pyrtlnet.hwh: pyrtlnet_ip_repo/src/pyrtl_inference_axi.v pyrtlnet_pynq.tcl
	$(VIVADO_BATCH) -source pyrtlnet_pynq.tcl
	cp pyrtlnet_pynq/pyrtlnet_pynq.runs/impl_1/design_1_wrapper.bit pyrtlnet.bit
	cp pyrtlnet_pynq/pyrtlnet_pynq.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh pyrtlnet.hwh

# Copy all required assets to the Pynq Z2 board, via rsync over ssh. This
# assumes that the Pynq Z2 is accessable over the network, and has been
# assigned the default hostname `pynq`.
deploy: ../pyrtlnet/inference_util.py ../pyrtlnet/numpy_inference.py pyrtlnet.bit pyrtlnet.hwh fpga_inference.py quantized.npz mnist_test_data.npz
	rsync -av -e ssh ../pyrtlnet pyrtlnet.bit pyrtlnet.hwh fpga_inference.py quantized.npz mnist_test_data.npz xilinx@pynq:

clean:
	rm -rf NA pyrtlnet_ip pyrtlnet_ip_repo pyrtlnet_pynq pyrtl_inference_axi.v pyrtl_inference_axi_test.v mnist_test_data.npz quantized.npz quantized.tflite pyrtlnet.bit pyrtlnet.hwh


