`pyrtlnet` FPGA
===============

Documentation for running `pyrtlnet` inference on a
[Pynq Z2](https://www.amd.com/en/corporate/university-program/aup-boards/pynq-z2.html)
FPGA.

### Requirements

A
[Pynq Z2](https://www.amd.com/en/corporate/university-program/aup-boards/pynq-z2.html)
FPGA is required. With some modifications, these instructions may work with
other FPGAs, but these instructions have only been tested with a Pynq Z2.

The scripts in this directory currently expect to be run on Linux. These
scripts have only been tested on Ubuntu 25.04.

### Pynq Z2 Set Up

Follow the [Pynq Z2 Setup Guide] to configure and boot the Pynq Z2. These
instructions were tested with SD card image v3.1.

Verify that you can
[connect to the board's Jupyter Notebook](https://pynq.readthedocs.io/en/latest/getting_started/pynq_z2_setup.html#connecting-to-jupyter-notebook).

Verify that you can `ssh` to the board:
```
$ ssh xilinx@pynq
...
Welcome to PYNQ Linux, based on Ubuntu 22.04 (GNU/Linux 6.6.10-xilinx-v2024.1-gb36799f4e960 armv7l)

Last login: Fri Oct 24 21:58:37 2025
xilinx@pynq:~$
```

Verify that you can
[connect to the Serial Console](https://pynq.readthedocs.io/en/latest/getting_started/pynq_z2_setup.html#opening-a-usb-serial-terminal).
This is not strictly necessary, but very useful for debugging.

### Install Vivado

Install Vivado. Use the version
[recommended for your SD card image](https://pynq.readthedocs.io/en/latest/pynq_sd_card.html#use-an-existing-ubuntu-os).
These instructions were tested with SD card image v3.1 and Vivado 2024.1.

Ensure that `vivado` is on your `$PATH`:

```
$ vivado -version
vivado v2024.1 (64-bit)
Tool Version Limit: 2024.05
SW Build 5076996 on Wed May 22 18:36:09 MDT 2024
IP Build 5075265 on Wed May 22 21:45:21 MDT 2024
SharedData Build 5076995 on Wed May 22 18:29:18 MDT 2024
Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
```

### Generate Verilog

Generate Verilog for the AXI version of the `pyrtlnet` inference hardware with:

```
$ make pyrtl_inference_axi.v
```

If successful, you should see a generated Verilog file named `pyrtl_inference_axi.v`:

```
$ head pyrtl_inference_axi.v
// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, s0_axi_araddr, s0_axi_aresetn, s0_axi_arvalid, s0_axi_awaddr, s0_axi_awvalid, s0_axi_bready, s0_axi_clk, s0_axi_rready, s0_axi_wdata, s0_axi_wstrb, s0_axi_wvalid, s0_axis_aclk, s0_axis_aresetn, s0_axis_tdata, s0_axis_tlast, s0_axis_tvalid, argmax, s0_axi_arready, s0_axi_awready, s0_axi_bresp, s0_axi_bvalid, s0_axi_rdata, s0_axi_rresp, s0_axi_rvalid, s0_axi_wready, s0_axis_tready, valid);
    input clk;
    input[6:0] s0_axi_araddr;
    input s0_axi_aresetn;
    input s0_axi_arvalid;
    input[6:0] s0_axi_awaddr;
```

The module should have many `axi` inputs and outputs.

### Generate bitstream and hardware handoff file

Generate a bitstream and hardware handoff file from `pyrtl_inference_axi.v`:

```
$ make pyrtlnet.bit pyrtlnet.hwh
```

This step takes some time, about 5 minutes on my computer.

If successful, you should see a bitstream named `pyrtlnet.bit`, and a hardware
handoff file named `pyrtlnet.hwh`:

```
$ ls -l pyrtlnet.bit pyrtlnet.hwh
-rw-rw-r-- 1 lauj lauj 4045676 Oct 27 10:57 pyrtlnet.bit
-rw-rw-r-- 1 lauj lauj  281520 Oct 27 10:57 pyrtlnet.hwh
```

> [!NOTE]
> This step implicitly runs `make pyrtl_inference_axi.v` from the previous
> step, so the previous step is not strictly necessary. For your first time, we
> recommend running the previous step explicitly, to understand where problems
> occur.

### Deploy assets to the Pynq Z2.

Copy:
1. The generated bitstream (`pyrtlnet.bit`)
2. The generated hardware handoff file (`pyrtlnet.hwh`)
3. The `pyrtlnet` FPGA driver script (`pyrtlnet.py`)
4. `pyrtlnet` Python libraries (`pyrtlnet/`)
5. Trained quantized neural network weights (`quantized.npz`)
6. MNIST test data (`mnist_test_data.npz`)

To the Pynq Z2, using [rsync](https://rsync.samba.org/) over `ssh`, by running:

```
$ make deploy
...
sending incremental file list
mnist_test_data.npz
pyrtlnet.bit
pyrtlnet.hwh
pyrtlnet.py
quantized.npz
pyrtlnet/
pyrtlnet/__init__.py
pyrtlnet/inference_util.py
pyrtlnet/litert_inference.py
pyrtlnet/mnist_util.py
pyrtlnet/numpy_inference.py
pyrtlnet/pyrtl_axi.py
pyrtlnet/pyrtl_inference.py
pyrtlnet/pyrtl_matrix.py
pyrtlnet/tensorflow_training.py
pyrtlnet/training_util.py
pyrtlnet/wire_matrix_2d.py

sent 5,725,817 bytes  received 332 bytes  1,272,477.56 bytes/sec
total size is 5,723,218  speedup is 1.00
```

### Prepare the Pynq Z2's Runtime Environment

`ssh` to the Pynq Z2:

```
$ ssh xilinx@pynq
```
Welcome to PYNQ Linux, based on Ubuntu 22.04 (GNU/Linux 6.6.10-xilinx-v2024.1-gb36799f4e960 armv7l)

Last login: Fri Oct 24 21:58:37 2025
xilinx@pynq:~$
```

[Activate the `pynq` virtual environment](https://discuss.pynq.io/t/run-python-scripts-in-pynq-environment/7761/2):

```
$ sudo bash
[sudo] password for xilinx:
root@pynq:/home/xilinx# . /etc/profile.d/pynq_venv.sh
(pynq-venv) root@pynq:/home/xilinx# . /etc/profile.d/xrt_setup.sh
(pynq-venv) root@pynq:/home/xilinx#
```

This lets us use Pynq from the command line, rather than using the slower
Jupyter Notebook interface.

Install required `pip` packages:

```
(pynq-venv) root@pynq:/home/xilinx# pip install pyrtl fxpmath
...
Installing collected packages: pyrtl, fxpmath
Successfully installed fxpmath-0.4.9 pyrtl-0.12
```

> [!NOTE]
> `fxpmath` is not actually required by the `pyrtlnet` FPGA driver script. This
> false dependency should be removed in a future update. `pyrtl` is required to
> convert raw bits to signed integers with
> [`val_to_signed_integer`](https://pyrtl.readthedocs.io/en/latest/helpers.html#pyrtl.val_to_signed_integer).

### Run `pyrtlnet` FPGA Inference

Run the `pyrtlnet.py` FPGA driver script, which loads the `pynq` runtime
environment, copies the `pyrtlnet` bitstream to the FPGA, transmits the MNIST
test image data to the FPGA via DMA, and retrieves the inference results:

![pyrtlnet.py screenshot](https://github.com/UCSBarchlab/pyrtlnet/blob/main/docs/images/pyrtlnet.png?raw=true)
