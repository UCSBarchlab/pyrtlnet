import unittest

import pyrtl

import pyrtlnet.pyrtl_axi as pyrtl_axi


class TestPyrtlAxiLite(unittest.TestCase):
    def setUp(self) -> None:
        pyrtl.reset_working_block()

        # default values for all Inputs. `aresetn` is active low.
        self.provided_inputs = {
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

    def simulate_axi_lite_read(self, sim: pyrtl.Simulation, read_addr: int) -> int:
        """Simulate an AXI register read."""
        # read_addr is a byte address, and each register is 32 bits wide, so read_addr
        # must be an even multiple of 4.
        self.assertEqual(read_addr % 4, 0)
        sim.step(
            self.provided_inputs | {"s0_axi_araddr": read_addr, "s0_axi_arvalid": True}
        )
        # Subordinate should be ready to receive read address.
        self.assertTrue(sim.inspect("s0_axi_arready"))

        # Wait until the read completes.
        read_done = False
        while not read_done:
            sim.step(self.provided_inputs | {"s0_axi_rready": True})
            read_done = sim.inspect("s0_axi_rvalid")
        # Read status should be OKAY (0).
        self.assertEqual(sim.inspect("s0_axi_rresp"), pyrtl_axi.RResp.OKAY)
        return sim.inspect("s0_axi_rdata")

    def test_axi_lite_subordinate_write(self) -> None:
        """Test writing and reading an AXI-Lite register."""
        _ = pyrtl_axi.make_axi_lite_subordinate(num_registers=1)

        sim = pyrtl.Simulation()

        # Write the register via AXI.
        written_data = 42
        sim.step(
            self.provided_inputs
            | {
                "s0_axi_awvalid": True,
                "s0_axi_wdata": written_data,
                "s0_axi_wvalid": True,
            }
        )
        # Subordinate should be ready to receive write address and write data.
        self.assertTrue(sim.inspect("s0_axi_awready"))
        self.assertTrue(sim.inspect("s0_axi_wready"))

        # Wait until the write completes.
        write_done = False
        while not write_done:
            sim.step(self.provided_inputs | {"s0_axi_bready": True})
            write_done = sim.inspect("s0_axi_bvalid")
        # Write status should be OKAY (0).
        self.assertEqual(sim.inspect("s0_axi_bresp"), pyrtl_axi.BResp.OKAY)

        # Read the register via AXI.
        read_data = self.simulate_axi_lite_read(sim, read_addr=0)
        self.assertEqual(read_data, written_data)

    def test_axi_lite_subordinate_write_addr_first(self) -> None:
        """Test writing and reading an AXI-Lite register.

        This is like `test_axi_lite_subordinate_write`, except the write address is
        transmitted first.
        """
        _ = pyrtl_axi.make_axi_lite_subordinate(num_registers=1)

        sim = pyrtl.Simulation()

        # Transmit the write address.
        sim.step(
            self.provided_inputs
            | {
                "s0_axi_awaddr": 0,
                "s0_axi_awvalid": True,
            }
        )
        # Subordinate should be ready to receive write address and write data.
        self.assertTrue(sim.inspect("s0_axi_awready"))
        self.assertTrue(sim.inspect("s0_axi_wready"))

        # Transmit the write data.
        written_data = 42
        sim.step(
            self.provided_inputs | {"s0_axi_wdata": written_data, "s0_axi_wvalid": True}
        )
        # Subordinate should be ready to receive write data, and not ready to receive
        # the write address.
        self.assertTrue(sim.inspect("s0_axi_wready"))
        self.assertFalse(sim.inspect("s0_axi_awready"))

        # Wait until the write completes.
        write_done = False
        while not write_done:
            sim.step(self.provided_inputs | {"s0_axi_bready": True})
            write_done = sim.inspect("s0_axi_bvalid")
        # Write status should be OKAY (0).
        self.assertEqual(sim.inspect("s0_axi_bresp"), pyrtl_axi.BResp.OKAY)

        # Read the register via AXI.
        read_data = self.simulate_axi_lite_read(sim, read_addr=0)
        self.assertEqual(read_data, written_data)

    def test_axi_lite_subordinate_write_data_first(self) -> None:
        """Test writing and reading an AXI-Lite register.

        This is like `test_axi_lite_subordinate_write`, except the write data is
        transmitted first.
        """
        _ = pyrtl_axi.make_axi_lite_subordinate(num_registers=1)

        sim = pyrtl.Simulation()

        # Transmit the write data.
        written_data = 42
        sim.step(
            self.provided_inputs | {"s0_axi_wdata": written_data, "s0_axi_wvalid": True}
        )
        # Subordinate should be ready to receive write address and write data.
        self.assertTrue(sim.inspect("s0_axi_awready"))
        self.assertTrue(sim.inspect("s0_axi_wready"))

        # Transmit the write address.
        sim.step(
            self.provided_inputs
            | {
                "s0_axi_awaddr": 0,
                "s0_axi_awvalid": True,
            }
        )
        # Subordinate should be ready to receive write address, and not ready to receive
        # the write data.
        self.assertTrue(sim.inspect("s0_axi_awready"))
        self.assertFalse(sim.inspect("s0_axi_wready"))

        # Wait until the write completes.
        write_done = False
        while not write_done:
            sim.step(self.provided_inputs | {"s0_axi_bready": True})
            write_done = sim.inspect("s0_axi_bvalid")
        # Write status should be OKAY (0).
        self.assertEqual(sim.inspect("s0_axi_bresp"), pyrtl_axi.BResp.OKAY)

        # Read the register via AXI.
        read_data = self.simulate_axi_lite_read(sim, read_addr=0)
        self.assertEqual(read_data, written_data)

    def test_axi_lite_subordinate_unwritable(self) -> None:
        """Test reading an externally managed AXI-Lite register."""
        registers = pyrtl_axi.make_axi_lite_subordinate(
            num_registers=2, num_writable_registers=0
        )

        self.assertEqual(len(registers), 2)
        # AXI-Lite registers are always 32 bits wide.
        self.assertEqual(registers[0].bitwidth, 32)
        self.assertEqual(registers[1].bitwidth, 32)

        # Set register values.
        first_register_value = 9999
        registers[0].next <<= first_register_value
        second_register_value = 0xFF
        registers[1].next <<= second_register_value

        sim = pyrtl.Simulation()

        # Read the first register.
        read_value = self.simulate_axi_lite_read(sim, read_addr=0)
        self.assertEqual(read_value, first_register_value)

        # Read the second register.
        counter_value = self.simulate_axi_lite_read(sim, read_addr=4)
        self.assertEqual(counter_value, second_register_value)


class TestPyrtlAxiStream(unittest.TestCase):
    def setUp(self) -> None:
        pyrtl.reset_working_block()

        # default values for all Inputs. `aresetn` is active low.
        self.provided_inputs = {
            "s0_axis_aclk": False,
            "s0_axis_aresetn": True,
            "s0_axis_tdata": 0,
            "s0_axis_tvalid": False,
            "s0_axis_tlast": False,
        }

    def test_axi_stream_subordinate(self) -> None:
        """Test writing a MemBlock from an AXI-Stream."""
        mem = pyrtl.MemBlock(name="mem", addrwidth=2, bitwidth=8)
        write_done = pyrtl_axi.make_axi_stream_subordinate(mem=mem)

        sim = pyrtl.Simulation()

        data = [i + 10 for i in range(2**mem.addrwidth)]

        # Transmit the `data`.
        for i, d in enumerate(data):
            ready = False
            while not ready:
                sim.step(
                    self.provided_inputs
                    | {
                        "s0_axis_tdata": d,
                        "s0_axis_tvalid": True,
                        "s0_axis_tlast": i == len(data) - 1,
                    }
                )
                ready = sim.inspect("s0_axis_tready")

        # Wait for the last write to finish.
        done = False
        while not done:
            sim.step(self.provided_inputs)
            done = sim.inspect(write_done.name)

        # Check that the memblock contains all the transmitted `data`.
        actual_data = sim.inspect_mem(mem)
        actual_data = [pair[1] for pair in sorted(actual_data.items())]
        self.assertEqual(data, actual_data)


if __name__ == "__main__":
    unittest.main()
