# Link the pyrtlnet IP created by pyrtlnet_ip.tcl with the Zynq Processing
# System, via AXI DMA.
#
# Tested with Vivado 2024.1

set project_dir [file normalize "pyrtlnet_pynq"]
# This `ip_repo_dir` must be consistent with `pyrtlnet_ip.tcl`'s `ip_repo_dir`.
set ip_repo_dir [file normalize "pyrtlnet_ip_repo"]

# Configure the project, and add the pyrtlnet IP repo. The IP repo is created
# by `pyrtlnet_ip.tcl`.
set_param board.repoPaths [get_property LOCAL_ROOT_DIR [xhub::get_xstores xilinx_board_store]]
create_project pyrtlnet_pynq "$project_dir" -part xc7z020clg400-1 -force
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property  ip_repo_paths  "$ip_repo_dir" [current_project]
update_ip_catalog

# Create a block design.
create_bd_design "design_1"
open_bd_design {"${project_dir}/pyrtlnet_pynq.srcs/sources_1/bd/design_1/design_1.bd"}
update_compile_order -fileset sources_1

# Configure Zynq Processing System.
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
# Enable High Performance AXI port, needed by AXI DMA.
set_property CONFIG.PCW_USE_S_AXI_HP0 {1} [get_bd_cells processing_system7_0]
# Reduce FPGA clock frequency, to make timing.
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {25} [get_bd_cells processing_system7_0]

# Configure AXI DMA.
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma
endgroup
# Disable scatter-gather and set data_width to 64. These are required by the
# Pynq Z2.
#
# Set tdata_width to 8, since our image data is 8 bits wide. Disable the write
# port since we don't use it.
set_property -dict [list \
  CONFIG.c_include_s2mm {0} \
  CONFIG.c_include_sg {0} \
  CONFIG.c_m_axi_mm2s_data_width {64} \
  CONFIG.c_m_axis_mm2s_tdata_width {8} \
  CONFIG.c_mm2s_burst_size {16} \
  CONFIG.c_sg_length_width {26} \
] [get_bd_cells dma]

startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/dma/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins dma/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/dma/M_AXI_MM2S} Slave {/processing_system7_0/S_AXI_HP0} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
endgroup

# Configure pyrtlnet.
startgroup
create_bd_cell -type ip -vlnv ucsb.edu:ucsbarchlab:pyrtlnet:1.0 pyrtlnet
endgroup

startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/processing_system7_0/FCLK_CLK0 (25 MHz)} Clk_slave {Auto} Clk_xbar {/processing_system7_0/FCLK_CLK0 (25 MHz)} Master {/processing_system7_0/M_AXI_GP0} Slave {/pyrtlnet/s0_axi} ddr_seg {Auto} intc_ip {/ps7_0_axi_periph} master_apm {0}}  [get_bd_intf_pins pyrtlnet/s0_axi]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (25 MHz)} Freq {25} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins pyrtlnet/s0_axi_clk]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (25 MHz)} Freq {25} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins pyrtlnet/s0_axis_aclk]
endgroup

connect_bd_intf_net [get_bd_intf_pins pyrtlnet/s0_axis] [get_bd_intf_pins dma/M_AXIS_MM2S]

regenerate_bd_layout
save_bd_design

# Make a HDL wrapper for the block design.
make_wrapper -files [get_files "${project_dir}/pyrtlnet_pynq.srcs/sources_1/bd/design_1/design_1.bd"] -top
add_files -norecurse "${project_dir}/pyrtlnet_pynq.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v"

# Run synthesis, implementation, and create a bitstream file.
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
