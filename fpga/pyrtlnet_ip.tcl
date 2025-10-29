# Create a Xilinx IP for the pyrtlnet Verilog module.
#
# Tested with Vivado 2024.1

set ip_project_dir [file normalize "pyrtlnet_ip"]
# This `ip_repo_dir` must be consistent with `pyrtlnet_pynq.tcl`'s
# `ip_repo_dir`.
set ip_repo_dir [file normalize "pyrtlnet_ip_repo"]
set verilog_source [file normalize "pyrtl_inference_axi.v"]

# Configure the project, and add the Verilog source code.
create_project pyrtlnet_ip "$ip_project_dir" -part xc7z020clg400-1 -force
add_files -norecurse "$verilog_source"
import_files -force -norecurse
update_compile_order -fileset sources_1

# Set IP metadata.
ipx::package_project -root_dir "$ip_repo_dir" -vendor ucsb.edu -library ucsbarchlab -taxonomy /UserIP -import_files
set_property name pyrtlnet [ipx::current_core]
set_property display_name pyrtlnet [ipx::current_core]
set_property description {pyrtlnet quantized inference} [ipx::current_core]

# Create a named register for `argmax`.
ipx::add_register argmax [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s0_axi -of_objects [ipx::current_core]]]
set_property size 32 [ipx::get_registers argmax -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s0_axi -of_objects [ipx::current_core]]]]
set_property display_name argmax [ipx::get_registers argmax -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s0_axi -of_objects [ipx::current_core]]]]
set_property description {Predicted digit, 0-9} [ipx::get_registers argmax -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s0_axi -of_objects [ipx::current_core]]]]

# Save the IP.
set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
set_property  ip_repo_paths "$ip_repo_dir" [current_project]
update_ip_catalog
