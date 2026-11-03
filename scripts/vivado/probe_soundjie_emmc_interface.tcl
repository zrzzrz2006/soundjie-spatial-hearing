# Offline interface characterization only. Never program a device or open a vendor project.
if {[version -short] ne "2017.4"} { error "Vivado 2017.4 required" }
if {$argc != 1} { error "usage: probe_soundjie_emmc_interface.tcl NEW_OUTPUT_DIR" }
set out [file normalize [lindex $argv 0]]
if {![string match "/mnt/hgfs/Soundjie_E/d4/artifacts/emmc_interface_probe_*" $out]} {
    error "output must be a new approved E-drive probe directory"
}
if {[file exists $out]} { error "refusing existing output directory" }
create_project emmc_probe $out -part xc7z100ffg900-2
create_bd_design emmc_probe
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
set_property -dict [list CONFIG.PCW_SD1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD1_SD1_IO {EMIO} \
    CONFIG.PCW_SD1_GRP_CD_ENABLE {1} CONFIG.PCW_SD1_GRP_CD_IO {EMIO} \
    CONFIG.PCW_SD1_GRP_WP_ENABLE {1} CONFIG.PCW_SD1_GRP_WP_IO {EMIO} \
    CONFIG.PCW_USE_M_AXI_GP0 {0}] $ps
make_bd_intf_pins_external [get_bd_intf_pins ps7/DDR]
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]
set sd [get_bd_intf_pins ps7/SDIO_1]
if {[llength $sd] != 1} { error "SDIO_1 did not enable" }
make_bd_intf_pins_external $sd
validate_bd_design
save_bd_design
set bd [get_files */emmc_probe.bd]
generate_target all $bd
set wrappers [make_wrapper -files $bd -top]
puts "EMMC_PROBE_WRAPPERS=$wrappers"
write_bd_tcl [file join $out emmc_probe_bd.tcl]
puts "EMMC_INTERFACE_PROBE_GENERATED_NOT_BOARD_VALIDATED"
close_project
exit 0
