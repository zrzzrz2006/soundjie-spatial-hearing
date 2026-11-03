# Vivado 2017.4 create-only integration script for the staged CH24 design.
# Contract names are repeated here for static audit:
# display_vdma -> S_AXI_HP0; dpu_data0 -> S_AXI_HP1;
# dpu_data1 -> S_AXI_HP2; dpu_instr -> S_AXI_HP3.

proc fail_closed {message} {
    puts stderr "SOUNDJIE_D4_FAIL_CLOSED: $message"
    return -code error $message
}

if {[version -short] ne "2017.4"} {
    fail_closed "REFUSE_IP_UPGRADE: Vivado 2017.4 is required"
}
if {$argc < 2} {
    fail_closed "usage: build_soundjie_d4_combined.tcl STAGED_ROOT OUTPUT_DIR ?preflight|build|resume|report|export?"
}
set staged_root [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
set mode [expr {$argc >= 3 ? [lindex $argv 2] : "preflight"}]
if {[file exists $output_dir]} {
    fail_closed "create-only output directory already exists: $output_dir"
}
file mkdir $output_dir

set project_file [file join $staged_root sources ch24 CH24_LCD_Tourch 02_fpga fpga system system.xpr]
set dpu_repo [file join $staged_root sources dpu ip_repo dpu]
if {![file isfile $project_file]} { fail_closed "missing staged CH24 project" }
if {![file isdirectory $dpu_repo]} { fail_closed "missing staged DPU IP repository" }

open_project $project_file
if {[get_property PART [current_project]] ne "xc7z100ffg900-2"} {
    fail_closed "PART_MISMATCH: expected xc7z100ffg900-2"
}
set_property ip_repo_paths [list $dpu_repo [file join $staged_root sources ch24 CH24_LCD_Tourch 02_fpga fpga ip_repo]] [current_project]
update_ip_catalog
if {![llength [get_ipdefs -all xilinx.com:ip:dpu_eu:2.0]]} {
    fail_closed "REFUSE_IP_UPGRADE: xilinx.com:ip:dpu_eu:2.0 is unavailable"
}
set locked_ips [get_ips -quiet -filter {IS_LOCKED == 1}]
if {[llength $locked_ips]} {
    fail_closed "REFUSE_IP_UPGRADE: locked IP detected: $locked_ips"
}

set bd_file [get_files -quiet */system.bd]
if {![llength $bd_file]} { fail_closed "missing CH24 system.bd" }
open_bd_design [lindex $bd_file 0]
if {![llength [get_bd_cells -quiet -hier -filter {VLNV =~ "xilinx.com:ip:axi_vdma:*"}]]} {
    fail_closed "missing display axi_vdma"
}
set ps [lindex [get_bd_cells -quiet -hier -filter {VLNV =~ "xilinx.com:ip:processing_system7:*"}] 0]
if {$ps eq ""} { fail_closed "missing processing_system7" }

# Preflight intentionally ends before mutation. The integration is performed only in build mode.
if {$mode eq "preflight"} {
    validate_bd_design
    puts "SOUNDJIE_D4_PREFLIGHT=pass"
    close_project
    exit 0
}
if {$mode eq "report" || $mode eq "export"} {
    if {![llength [get_runs -quiet impl_1]] || [get_property PROGRESS [get_runs impl_1]] ne "100%"} {
        fail_closed "report mode requires a completed impl_1 run"
    }
    open_run impl_1
    source [file join [file dirname [info script]] report_soundjie_d4_combined.tcl]
    if {$mode eq "export"} {
        write_bitstream -force [file join $output_dir soundjie_d4_combined.bit]
        write_hwdef -force -file [file join $output_dir soundjie_d4_combined.hdf]
        puts "SOUNDJIE_D4_EXPORT=pass"
    } else {
        puts "SOUNDJIE_D4_REPORT_ONLY=pass"
    }
    close_project
    exit 0
}
if {$mode ne "build" && $mode ne "resume"} { fail_closed "unknown mode: $mode" }

if {$mode eq "build"} {
if {![llength [get_bd_cells -quiet soundjie_hdmi_tx]]} {
    source [file join [file dirname [info script]] integrate_soundjie_d4_hdmi.tcl]
}
source [file join [file dirname [info script]] .. .. tests vivado test_soundjie_d4_hdmi_bd.tcl]
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_USE_S_AXI_HP2 {1} \
    CONFIG.PCW_USE_S_AXI_HP3 {1} \
    CONFIG.PCW_S_AXI_HP3_DATA_WIDTH {32} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1}] $ps

if {[llength [get_bd_cells -quiet dpu_eu_0]]} {
    fail_closed "DPU already exists; refusing an ambiguous rebuild"
}
set dpu [create_bd_cell -type ip -vlnv xilinx.com:ip:dpu_eu:2.0 dpu_eu_0]

# DPU HP contract: display stays on HP0, DPU owns HP1/HP2/HP3.
connect_bd_intf_net [get_bd_intf_pins $dpu/DPU0_M_AXI_DATA0] [get_bd_intf_pins $ps/S_AXI_HP1]
connect_bd_intf_net [get_bd_intf_pins $dpu/DPU0_M_AXI_DATA1] [get_bd_intf_pins $ps/S_AXI_HP2]
connect_bd_intf_net [get_bd_intf_pins $dpu/DPU0_M_AXI_INSTR] [get_bd_intf_pins $ps/S_AXI_HP3]

# Use Vivado connection automation only for the DPU AXI-Lite slave. It extends the
# existing GP0 control interconnect and does not own a DDR HP port.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Master {/processing_system7_0/M_AXI_GP0} Clk {/processing_system7_0/FCLK_CLK0}] \
    [get_bd_intf_pins $dpu/S_AXI]

set clk_wiz [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:5.4 soundjie_dpu_clk_wiz]
set fclk_hz [get_property CONFIG.FREQ_HZ [get_bd_pins $ps/FCLK_CLK0]]
set fclk_mhz [format "%.6f" [expr {$fclk_hz / 1000000.0}]]
set_property -dict [list CONFIG.PRIM_IN_FREQ $fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {150.000} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {300.000} \
    CONFIG.CLKOUT2_USED {true} CONFIG.NUM_OUT_CLKS {2} \
    CONFIG.USE_LOCKED {false} CONFIG.USE_RESET {false}] $clk_wiz
connect_bd_net [get_bd_pins $ps/FCLK_CLK0] [get_bd_pins $clk_wiz/clk_in1]
connect_bd_net [get_bd_pins $clk_wiz/clk_out1] [get_bd_pins $dpu/m_axi_dpu_aclk] \
    [get_bd_pins $ps/S_AXI_HP1_ACLK] [get_bd_pins $ps/S_AXI_HP2_ACLK] [get_bd_pins $ps/S_AXI_HP3_ACLK]
connect_bd_net [get_bd_pins $clk_wiz/clk_out2] [get_bd_pins $dpu/dpu_2x_clk]

set rst_axi [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 soundjie_dpu_axi_reset]
set rst_2x [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 soundjie_dpu_2x_reset]
connect_bd_net [get_bd_pins $clk_wiz/clk_out1] [get_bd_pins $rst_axi/slowest_sync_clk]
connect_bd_net [get_bd_pins $clk_wiz/clk_out2] [get_bd_pins $rst_2x/slowest_sync_clk]
connect_bd_net [get_bd_pins $ps/FCLK_RESET0_N] [get_bd_pins $rst_axi/ext_reset_in] [get_bd_pins $rst_2x/ext_reset_in]
connect_bd_net [get_bd_pins $rst_axi/peripheral_aresetn] [get_bd_pins $dpu/m_axi_dpu_aresetn]
connect_bd_net [get_bd_pins $rst_2x/peripheral_aresetn] [get_bd_pins $dpu/dpu_2x_resetn]

set irq_concat [lindex [get_bd_cells -quiet xlconcat_0] 0]
if {$irq_concat eq ""} { fail_closed "missing CH24 interrupt concat" }
# Preserve CH24 In0=LCD touch and In1=VDMA; append DPU at In2.
set_property -dict [list CONFIG.NUM_PORTS {3}] $irq_concat
connect_bd_net [get_bd_pins $dpu/dpu_interrupt] [get_bd_pins $irq_concat/In2]

assign_bd_address
set_property offset 0x4F000000 [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces $ps/Data] -filter {NAME =~ "*dpu_eu_0*"}]
validate_bd_design
save_bd_design
} elseif {$mode eq "resume"} {
    set dpu [lindex [get_bd_cells -quiet dpu_eu_0] 0]
    if {$dpu eq ""} { fail_closed "resume requires an existing DPU" }
    if {[get_property VLNV $dpu] ne "xilinx.com:ip:dpu_eu:2.0"} {
        fail_closed "resume found an unexpected DPU version"
    }
    validate_bd_design
    puts "SOUNDJIE_D4_RESUME_AUDIT=pass"
}

set cdc_xdc [file normalize [file join [file dirname [info script]] .. .. constraints soundjie_d4_axis2video_cdc.xdc]]
if {![file isfile $cdc_xdc]} { fail_closed "missing axis2video CDC constraint" }
if {![llength [get_files -quiet $cdc_xdc]]} {
    add_files -fileset constrs_1 $cdc_xdc
}
set_property USED_IN_SYNTHESIS true [get_files $cdc_xdc]
set_property USED_IN_IMPLEMENTATION true [get_files $cdc_xdc]

set hdmi_xdc [file normalize [file join [file dirname [info script]] .. .. constraints soundjie_d4_hdmi.xdc]]
if {![llength [get_files -quiet $hdmi_xdc]]} { add_files -fileset constrs_1 $hdmi_xdc }
source [file join [file dirname [info script]] .. .. tests vivado test_soundjie_d4_hdmi_bd.tcl]

generate_target all $bd_file
set wrapper_files [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_files
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1
reset_run impl_1
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    fail_closed "synthesis did not complete"
}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
source [file join [file dirname [info script]] report_soundjie_d4_combined.tcl]
write_bitstream -force [file join $output_dir soundjie_d4_combined.bit]
write_hwdef -force -file [file join $output_dir soundjie_d4_combined.hdf]
close_project
