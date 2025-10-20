from enum import IntEnum

import pyrtl

from pyrtlnet import pyrtl_axi


def main() -> None:
    """AXI-Stream and AXI-Lite demo:

    1. Write data from AXI-Stream to a :class:`~pyrtl.MemBlock`, using
       :func:`make_axi_stream_subordinate`.

    2. Then add up the data in the :class:`~pyrtl.MemBlock`, using a state machine
       defined in this file.

    3. Then read the sum via AXI-Lite, using :func:`make_axi_lite_subordinate`.
    """
    # Make an AXI-Stream subordinate that will write Stream data to `mem`.
    mem = pyrtl.MemBlock(name="mem", addrwidth=2, bitwidth=8)
    sum_done = pyrtl_axi.make_axi_stream_subordinate(mem=mem)

    # Make a state machine for the MemBlock summer.
    class SumState(IntEnum):
        # Loading Stream data into `mem`.
        LOAD = 0
        # Reading `mem` and adding up the read data.
        SUM = 1
        # We need wait one additional cycle for the data to propagate to `total`, due to
        # our synchronous MemBlock read.
        WAIT = 2
        # Addition complete, the sum can be read from `total`.
        DONE = 3

    state = pyrtl.Register(
        name="sum_state",
        bitwidth=pyrtl.infer_val_and_bitwidth(max(SumState)).bitwidth,
    )

    read_addr = pyrtl.Register(name="read_addr", bitwidth=mem.addrwidth)
    read_data = pyrtl.Register(name="read_data", bitwidth=mem.bitwidth)

    # Synchronous MemBlock read: the read's input (`read_addr`) is a Register, and the
    # data read goes directly to a `read_data` Register.
    read_data.next <<= mem[read_addr]

    done = pyrtl.WireVector(name="done", bitwidth=1)

    total = pyrtl.Register(name="total", bitwidth=32)
    with pyrtl.conditional_assignment:
        with state == SumState.LOAD:
            # Transition LOAD -> SUM.
            with sum_done:
                state.next |= SumState.SUM

        with state == SumState.SUM:
            # Transition SUM -> WAIT.
            read_addr.next |= read_addr + 1
            with read_addr != 0:
                total.next |= total + read_data
                with read_addr == 2**mem.addrwidth - 1:
                    state.next |= SumState.WAIT

        with state == SumState.WAIT:
            # Transition WAIT -> DONE.
            total.next |= total + read_data
            state.next |= SumState.DONE

        with state == SumState.DONE:
            done |= True

    # Create an AXI-Lite subordinate that will report the summer's `total`.
    registers = pyrtl_axi.make_axi_lite_subordinate(
        num_registers=2, num_writable_registers=0
    )
    # The `total` will be stored in register 1, which has AXI address 4. AXI-Lite
    # registers are 32-bits, and AXI addresses are byte addresses.
    registers[0].next <<= 9999
    registers[1].next <<= total

    sim = pyrtl.Simulation()
    # `provided_inputs` holds default values for all Inputs. `aresetn` is active low.
    provided_inputs = {
        # AXI-Stream Inputs.
        "s0_axis_aclk": False,
        "s0_axis_aresetn": True,
        "s0_axis_tdata": 0,
        "s0_axis_tvalid": False,
        "s0_axis_tlast": False,
        # AXI-Lite Inputs.
        "s0_axi_clk": False,
        "s0_axi_aresetn": True,
        "s0_axi_araddr": 0,
        "s0_axi_arvalid": False,
        "s0_axi_rready": False,
        "s0_axi_awaddr": 0,
        "s0_axi_awvalid": False,
        "s0_axi_wdata": 0,
        "s0_axi_wvalid": False,
        "s0_axi_wstrb": 0,
        "s0_axi_bready": False,
    }
    # Transmit the MemBlock's data via AXI-Stream.
    data = [i + 10 for i in range(2**mem.addrwidth)]
    print("Calculating sum of:", data, "\n")
    pyrtl_axi.simulate_axi_stream_send(sim, provided_inputs, stream_data=data)

    # Wait for the summation to complete.
    done = False
    while not done:
        sim.step(provided_inputs)
        done = sim.inspect("done")

    # Read the sum via AXI-Lite. The sum is stored in register 1, which has AXI
    # byte-address 4.
    actual_sum = pyrtl_axi.simulate_axi_lite_read(sim, provided_inputs, address=4)

    sim.tracer.render_trace(
        trace_list=[
            # AXI-Stream state.
            "s0_axis_tdata",
            "s0_axis_tready",
            "s0_axis_tvalid",
            "s0_axis_state",
            "s0_axis_write_addr_reg",
            "s0_axis_write_data_reg",
            "s0_axis_write_enable_reg",
            # Summer state.
            "read_addr",
            "read_data",
            "sum_state",
            "total",
            # AXI-Lite state.
            "s0_axi_register1",
            "s0_axi_araddr",
            "s0_axi_arready",
            "s0_axi_arvalid",
            "s0_axi_rdata",
            "s0_axi_rready",
            "s0_axi_rvalid",
            "s0_axi_read_state",
        ],
        repr_func=int,
        repr_per_name={
            "s0_axis_state": pyrtl.enum_name(pyrtl_axi.StreamState),
            "sum_state": pyrtl.enum_name(SumState),
            "s0_axi_read_state": pyrtl.enum_name(pyrtl_axi.ReadState),
        },
    )

    expected_sum = sum(data)
    if actual_sum == expected_sum:
        print("\nReceived correct sum", actual_sum)
    else:
        print("\nReceived INCORRECT sum", actual_sum, "expected sum is", expected_sum)


if __name__ == "__main__":
    main()
