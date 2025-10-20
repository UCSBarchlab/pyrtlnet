"""
Basic hardware implementation of `AXI`_ protocol subordinates, in the `PyRTL`_ hardware
description language.

.. _AXI: https://developer.arm.com/documentation/ihi0022/latest

.. _PyRTL: https://github.com/UCSBarchlab/PyRTL

These subordinates can be useful for integrating ``pyrtlnet``'s hardware with other
systems, for example most `Xilinx IP`_ connects via AXI. The `pyrtl_inference demo`_
uses these subordinates when its ``--axi`` option is enabled.

.. _Xilinx IP: https://docs.amd.com/v/u/en-US/ug1037-vivado-axi-reference-guide

.. _pyrtl_inference demo: https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_inference.py

See the `pyrtl_axi demo`_ for a simple example of how these subordinates work.

.. _pyrtl_axi demo: https://github.com/UCSBarchlab/pyrtlnet/blob/main/pyrtl_axi.py
"""

from enum import IntEnum

import pyrtl


class ReadState(IntEnum):
    """States for `make_axi_lite_subordinate`'s AXI read state machine."""

    # Wait for read address.
    IDLE = 0
    # Send read data and response.
    RESP = 1


class WriteState(IntEnum):
    """States for `make_axi_lite_subordinate`'s the AXI write state machine."""

    # Wait for write address or write data.
    IDLE = 0
    # Have write data, wait for write address.
    ADDR = 1
    # Have write address, wait for write data.
    DATA = 2
    # Send write response.
    RESP = 3


class RResp(IntEnum):
    """Valid values for :func:`make_axi_lite_subordinate`'s ``rresp`` Output.

    See the `AXI Specification`_, Table A4.24: RRESP encodings.

    .. _AXI Specification: https://developer.arm.com/documentation/ihi0022/latest
    """

    OKAY = 0
    EXOKAY = 1
    SLVERR = 2
    DECERR = 3
    PREFETCHED = 4
    TRANSFAULT = 5
    OKAYDIRTY = 6


class BResp(IntEnum):
    """Valid values for :func:`make_axi_lite_subordinate`'s ``bresp`` Output

    See the `AXI Specification`_, Table A4.21: BRESP encodings

    .. _AXI Specification: https://developer.arm.com/documentation/ihi0022/latest
    """

    OKAY = 0
    EXOKAY = 1
    SLVERR = 2
    DECERR = 3
    DEFER = 4
    TRANSFAULT = 5
    UNSUPPORTED = 7


def make_axi_lite_subordinate(
    num_registers: int, num_writable_registers: int | None = None, channel: int = 0
) -> list[pyrtl.Register]:
    """Makes a basic :class:`~pyrtl.Register`-based AXI-Lite subordinate.

    This creates a set of :class:`Registers<pyrtl.Register>` that can be read or written
    via AXI-Lite. Each register has a fixed bitwidth of 32 bits. The registers are
    assigned AXI addresses, which are byte addresses, so register 0 has AXI address 0,
    register 1 has AXI address 4, register 2 has AXI address 8, and so on.

    See the `AXI Specification`_ for details.

    .. _AXI Specification: https://developer.arm.com/documentation/ihi0022/latest

    The generated reset logic only resets :class:`Registers<pyrtl.Register>` that can be
    written via AXI. See the ``num_writable_registers`` argument.

    :param num_registers: The number of :class:`Registers<pyrtl.Register>` managed by
        this AXI-Lite subordinate. ``num_registers`` must be greater than zero. This
        many :class:`Registers<pyrtl.Register>` will be created and returned by
        ``make_axi_lite_subordinate``. Each :class:`~pyrtl.Register` is 32 bits wide.
        AXI addresses are byte addresses, so the :class:`Registers<pyrtl.Register>` will
        be assigned AXI addresses 0, 4, 8, 12, ...
    :param num_writable_registers: Determines which :class:`Registers<pyrtl.Register>`
        may be written via AXI. If ``None``, all :class:`Registers<pyrtl.Register>` may
        be written via AXI. If not ``None``, only the first ``num_writable_registers``
        :class:`Registers<pyrtl.Register>` may be written via AXI. Attempts to write a
        non-writable register will be ignored. ``num_writable_registers`` must be less
        than or equal to ``num_registers``.
    :param channel: Channel number for this AXI connection. Channel numbers are only
        used to name wires. When a module has multiple AXI interfaces, each AXI
        interface must use a different channel number to avoid wire name collisions.

    :returns: A :class:`list` of 32-bit :class:`Registers<pyrtl.Register>` that can be
              read or written via AXI-Lite. These :class:`Registers<pyrtl.Register>` can
              be freely read outside of this function. Any unwritable
              :class:`Registers<pyrtl.Register>` (see the ``num_writable_registers``
              argument) must have their :attr:`~pyrtl.Register.next` attribute set
              outside of this function.
    """
    assert num_registers > 0

    if num_writable_registers is None:
        num_writable_registers = num_registers

    assert num_writable_registers >= 0
    assert num_writable_registers <= num_registers

    # Bitwidth of an AXI byte address. The registers have addresses 0, 4, 8, 12, ... We
    # drop the lowest two bits to convert an `addr` to an `index`, so `addr_bitwidth`
    # must be at least 3 bits wide.
    addr_bitwidth = max(
        3, pyrtl.infer_val_and_bitwidth((num_registers - 1) * 4).bitwidth
    )

    # Bitwidth of a register index. These are indices into the ``registers`` list. The
    # registers have indexes 0, 1, 2, 3, ...
    index_bitwidth = max(1, pyrtl.infer_val_and_bitwidth(num_registers - 1).bitwidth)

    # Bitwidth of each register.
    data_bitwidth = 32

    # Name prefix for all WireVectors.
    prefix = f"s{channel}_axi_"

    registers = [
        pyrtl.Register(name=f"{prefix}register{i}", bitwidth=data_bitwidth)
        for i in range(num_registers)
    ]

    # Define all AXI Inputs and Outputs.

    _ = pyrtl.Input(name=f"{prefix}clk", bitwidth=1)
    # `aresetn` is active low.
    reset = ~pyrtl.Input(name=f"{prefix}aresetn", bitwidth=1)

    # Read channel address.
    read_addr = pyrtl.Input(name=f"{prefix}araddr", bitwidth=addr_bitwidth)
    # Indicates that we are ready to receive ``read_addr``.
    read_addr_ready = pyrtl.Output(name=f"{prefix}arready", bitwidth=1)
    # Indicates that the manager is ready to send ``read_addr``.
    read_addr_valid = pyrtl.Input(name=f"{prefix}arvalid", bitwidth=1)

    # Read channel data.
    read_data = pyrtl.Output(name=f"{prefix}rdata", bitwidth=data_bitwidth)
    # Status of the last read. See the `RResp` `IntEnum`.
    read_resp = pyrtl.Output(name=f"{prefix}rresp", bitwidth=2)
    # Indicates that the manager is ready to receive ``read_data`` and ``read_resp``.
    read_data_ready = pyrtl.Input(name=f"{prefix}rready", bitwidth=1)
    # Indicates that we are ready to send ``read_data`` and ``read_resp``.
    read_data_valid = pyrtl.Output(name=f"{prefix}rvalid", bitwidth=1)

    # Write channel address.
    write_addr = pyrtl.Input(name=f"{prefix}awaddr", bitwidth=addr_bitwidth)
    # Indicates that we are ready to receive ``write_addr``.
    write_addr_ready = pyrtl.Output(name=f"{prefix}awready", bitwidth=1)
    # Indicates that the manager is ready to send ``write_addr``.
    write_addr_valid = pyrtl.Input(name=f"{prefix}awvalid", bitwidth=1)

    # Write channel data.
    write_data = pyrtl.Input(name=f"{prefix}wdata", bitwidth=data_bitwidth)
    # Indicates that we are ready to receive ``write_data``.
    write_data_ready = pyrtl.Output(name=f"{prefix}wready", bitwidth=1)
    # Indicates that the manager is ready to send ``write_data``.
    write_data_valid = pyrtl.Input(name=f"{prefix}wvalid", bitwidth=1)
    # Bitfield indicating which of the 4 ``write_data`` bytes are valid. This
    # implementation assumes all ``write_data`` bytes are always valid.
    _ = pyrtl.Input(name=f"{prefix}wstrb", bitwidth=4)

    # Status of the last write.  See the `BResp` `IntEnum`.
    write_resp = pyrtl.Output(name=f"{prefix}bresp", bitwidth=2)
    # Indicates that the manager is ready to receive ``write_resp``.
    write_resp_ready = pyrtl.Input(name=f"{prefix}bready", bitwidth=1)
    # Indicates that we are ready to send ``write_resp``.
    write_resp_valid = pyrtl.Output(name=f"{prefix}bvalid", bitwidth=1)

    # For now, read and write responses are always OKAY.
    # TODO: Return RResp.DECERR or BResp.DECERR for invalid addresses.
    read_resp <<= RResp.OKAY
    write_resp <<= BResp.OKAY

    # Make the read state machine.
    read_state = pyrtl.Register(
        name=f"{prefix}read_state",
        bitwidth=pyrtl.infer_val_and_bitwidth(max(ReadState)).bitwidth,
    )

    # Index of the Register to read.
    read_index = pyrtl.Register(name=f"{prefix}read_index", bitwidth=index_bitwidth)

    with pyrtl.conditional_assignment:
        with reset:
            read_state.next |= ReadState.IDLE
            read_index.next |= 0

        with read_state == ReadState.IDLE:
            # Save the selected register's index in the ``read_index`` Register.
            read_addr_ready |= True
            with read_addr_valid:
                # Transition IDLE -> RESP.
                read_state.next |= ReadState.RESP
                # ``read_addr`` is a byte address, so drop the two lowest bits to
                # convert to a register index.
                read_index.next |= read_addr[2:]

        with read_state == ReadState.RESP:
            # Return the data in the selected register as ``read_data``.
            read_data_valid |= True
            if len(registers) == 1:
                read_data |= registers[0]
            else:
                read_data |= pyrtl.mux(read_index, *registers, default=0)

            with read_data_ready:
                # Transition RESP -> IDLE.
                read_state.next |= ReadState.IDLE

    # Make the write state machine.
    write_state = pyrtl.Register(
        name=f"{prefix}write_state",
        bitwidth=pyrtl.infer_val_and_bitwidth(max(WriteState)).bitwidth,
    )

    # Index of the Register to write.
    write_index_reg = pyrtl.Register(
        name=f"{prefix}write_index_reg", bitwidth=index_bitwidth
    )

    # Data to write to the selected Register.
    write_data_reg = pyrtl.Register(
        name=f"{prefix}write_data_reg", bitwidth=data_bitwidth
    )

    with pyrtl.conditional_assignment:
        with reset:
            write_state.next |= WriteState.IDLE
            write_index_reg.next |= 0
            write_data_reg.next |= 0
            for register in registers[:num_writable_registers]:
                register.next |= 0

        with write_state == WriteState.IDLE:
            write_addr_ready |= True
            write_data_ready |= True

            with write_addr_valid & write_data_valid:
                # Transition IDLE -> RESP.
                write_index_reg.next |= write_addr[2:]
                write_data_reg.next |= write_data
                write_state.next |= WriteState.RESP

            with write_addr_valid:
                # Transition IDLE -> DATA.
                write_index_reg.next |= write_addr[2:]
                write_state.next |= WriteState.DATA

            with write_data_valid:
                # Transition IDLE -> ADDR.
                write_data_reg.next |= write_data
                write_state.next |= WriteState.ADDR

        with write_state == WriteState.ADDR:
            write_addr_ready |= True
            with write_addr_valid:
                # Transition ADDR -> RESP.
                write_index_reg.next |= write_addr[2:]
                write_state.next |= WriteState.RESP

        with write_state == WriteState.DATA:
            write_data_ready |= True
            with write_data_valid:
                # Transition DATA -> RESP.
                write_data_reg.next |= write_data
                write_state.next |= WriteState.RESP

        with write_state == WriteState.RESP:
            write_resp_valid |= True
            with write_resp_ready:
                # Transition RESP -> IDLE.
                write_state.next |= WriteState.IDLE
                # Write ``write_data_reg`` to the register selected by
                # ``write_index_reg``.
                for i, register in enumerate(registers[:num_writable_registers]):
                    with i == write_index_reg:
                        register.next |= write_data_reg

    return registers


def simulate_axi_lite_read(
    sim: pyrtl.Simulation, provided_inputs: dict, address: int, channel: int = 0
) -> None:
    """Simulate reading an AXI-Lite register.

    :param sim: The PyRTL :class:`~pyrtl.Simulation` to read the AXI-Lite register from.
    :param provided_inputs: Additional :class:`~pyrtl.Input` values for the PyRTL
        :class:`~pyrtl.Simulation`.
    :param address: Address of the AXI-Lite register to read. AXI-Lite addresses are
        byte addresses, and each register is 32 bits wide, so ``address`` must be an
        even multiple of 4.
    :param channel: Channel number for this AXI connection. Channel numbers are only
        used to name wires. When a module has multiple AXI interfaces, each AXI
        interface must use a different channel number to avoid wire name collisions.

    :returns: The contents of the requested AXI-Lite register.
    """
    # All AXI-Lite addresses are byte addresses, and each register is 32 bits wide, so
    # `address` must be an even multiple of 4.
    assert address % 4 == 0

    # Name prefix for all WireVectors.
    prefix = f"s{channel}_axi_"

    # Send the AXI address.
    ready = False
    while not ready:
        sim.step(
            provided_inputs
            | {
                f"{prefix}araddr": address,
                f"{prefix}arvalid": True,
            }
        )
        ready = sim.inspect(f"{prefix}arready")

    # Receive the data.
    valid = False
    while not valid:
        sim.step(provided_inputs | {f"{prefix}rready": True})
        valid = sim.inspect(f"{prefix}rvalid")

    # Read status should be OKAY (0).
    assert sim.inspect(f"{prefix}rresp") == RResp.OKAY

    return sim.inspect(f"{prefix}rdata")


class StreamState(IntEnum):
    """States for `make_axi_stream_subordinate`'s state machine."""

    # Loading data into the MemBlock.
    LOAD = 0
    # Finished loading data into the MemBlock.
    DONE = 1


def make_axi_stream_subordinate(
    mem: pyrtl.MemBlock, channel: int = 0
) -> pyrtl.WireVector:
    """Makes a basic :class:`~pyrtl.MemBlock`-based AXI-Stream subordinate.

    The Stream's data will be written to the :class:`~pyrtl.MemBlock`. The
    :class:`~pyrtl.MemBlock` will be completely overwritten with the Stream's data,
    starting from address ``0``. The :class:`~pyrtl.MemBlock`'s
    :attr:`~pyrtl.MemBlock.addrwidth` determines the number of data items to write, and
    the :class:`~pyrtl.MemBlock`'s :attr:`~pyrtl.MemBlock.bitwidth` determines the size
    of each data item.

    See the `AXI-Stream Spec`_ for details.

    .. _AXI-Stream Spec: https://developer.arm.com/documentation/ihi0051/latest/

    The generated reset logic does not reset the :class:`~pyrtl.MemBlock`'s contents.

    :param mem: The :class:`~pyrtl.MemBlock` to fill with the Stream's data. The
        :class:`~pyrtl.MemBlock`'s :attr:`~pyrtl.MemBlock.bitwidth` must be an even
        multiple of 8.
    :param channel: Channel number for this AXI connection. Channel numbers are only
        used to name wires. When a module has multiple AXI interfaces, each AXI
        interface must use a different channel number to avoid wire name collisions.

    :returns: A 1-bit :class:`~pyrtl.WireVector` that indicates when loading is
              complete, and ``mem`` has been completely overwritten with Stream data.
    """
    addr_bitwidth = mem.addrwidth
    data_bitwidth = mem.bitwidth

    # The transmitted data must be an integer number of bytes. See the AXI-Stream
    # specification.
    assert data_bitwidth % 8 == 0

    # Name prefix for all WireVectors.
    prefix = f"s{channel}_axis_"

    # Define all AXI-Stream Inputs and Outputs.

    _ = pyrtl.Input(name=f"{prefix}aclk", bitwidth=1)
    # `aresetn` is active low.
    reset = ~pyrtl.Input(name=f"{prefix}aresetn", bitwidth=1)

    data = pyrtl.Input(name=f"{prefix}tdata", bitwidth=data_bitwidth)
    ready = pyrtl.Output(name=f"{prefix}tready", bitwidth=1)
    valid = pyrtl.Input(name=f"{prefix}tvalid", bitwidth=1)
    # `tlast` indicates a packet boundary. This basic implementation ignores packet
    # boundaries, and just keeps loading data from the stream until the `MemBlock` is
    # full.
    _ = pyrtl.Input(name=f"{prefix}tlast", bitwidth=1)

    # Current address to store data. Increments by one when `data` is `valid`.
    # `addr` is an index into `mem`.
    addr = pyrtl.Register(name=f"{prefix}addr", bitwidth=addr_bitwidth)

    # `addr` delayed by one cycle. Needed for synchronous MemBlock writes.
    write_addr_reg = pyrtl.Register(
        name=f"{prefix}write_addr_reg", bitwidth=addr_bitwidth
    )
    # `data` delayed by one cycle. Needed for synchronous MemBlock writes.
    write_data_reg = pyrtl.Register(
        name=f"{prefix}write_data_reg", bitwidth=data_bitwidth
    )
    # Write enable logic, delayed by one cycle. Needed for synchronous MemBlock writes.
    write_enable_reg = pyrtl.Register(name=f"{prefix}write_enable_reg", bitwidth=1)

    # Synchronous MemBlock write: All write inputs (address, data, write enable) are
    # Registers.
    mem[write_addr_reg] <<= pyrtl.MemBlock.EnabledWrite(
        data=write_data_reg, enable=write_enable_reg
    )

    # Make the state machine.
    state = pyrtl.Register(
        name=f"{prefix}state",
        bitwidth=pyrtl.infer_val_and_bitwidth(max(StreamState)).bitwidth,
    )

    with pyrtl.conditional_assignment:
        with reset:
            addr.next |= 0
            state.next |= StreamState.LOAD
            write_addr_reg.next |= 0
            write_data_reg.next |= 0
            write_enable_reg.next |= False

        with state == StreamState.LOAD:
            ready |= True
            with valid:
                # Received valid data. Write the received data to the MemBlock and
                # advance to the next write address.
                write_addr_reg.next |= addr
                write_data_reg.next |= data
                write_enable_reg.next |= True
                with addr == 2**addr_bitwidth - 1:
                    state.next |= StreamState.DONE
                with pyrtl.otherwise:
                    addr.next |= addr + 1

        with state == StreamState.DONE:
            write_enable_reg.next |= False

    done = pyrtl.WireVector(name=f"{prefix}done", bitwidth=1)
    done <<= state == StreamState.DONE
    return done


def simulate_axi_stream_send(
    sim: pyrtl.Simulation,
    provided_inputs: dict,
    stream_data: list[int],
    channel: int = 0,
) -> None:
    """Simulate sending data to an AXI-Stream.

    :param sim: The PyRTL :class:`~pyrtl.Simulation` to read the AXI-Lite register from.
    :param provided_inputs: Additional :class:`~pyrtl.Input` values for the PyRTL
        :class:`~pyrtl.Simulation`.
    :param stream_data: Data to send to the AXI-Stream. At most one element from this
        :class:`list` will be sent each cycle. Each :class:`list` element must fit in
        the stream's bitwidth.
    :param channel: Channel number for this AXI connection. Channel numbers are only
        used to name wires. When a module has multiple AXI interfaces, each AXI
        interface must use a different channel number to avoid wire name collisions.
    """
    # Name prefix for all WireVectors.
    prefix = f"s{channel}_axis_"

    # Transmit the memblock_data via AXI-Stream.
    for i, data in enumerate(stream_data):
        ready = False
        while not ready:
            sim.step(
                provided_inputs
                | {
                    f"{prefix}tdata": data,
                    f"{prefix}tvalid": True,
                    f"{prefix}tlast": i == len(stream_data) - 1,
                }
            )
            ready = sim.inspect(f"{prefix}tready")
