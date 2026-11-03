# Source only in the approved isolated candidate, never the previous build.
set emmc_root [file normalize [file join [file dirname [info script]] ../..]]
source [file join [file dirname [info script]] emmc_candidate_guard.tcl]
set emmc_expected [file join $emmc_root artifacts pl_rebuild_20261024_emmc_a sources ch24 CH24_LCD_Tourch 02_fpga fpga system system.xpr]
set emmc_project [file join [get_property DIRECTORY [current_project]] "[get_property NAME [current_project]].xpr"]
set emmc_ps_cells [get_bd_cells -hier -filter {VLNV =~ "xilinx.com:ip:processing_system7:*"}]
emmc_candidate_guard $emmc_project $emmc_expected [get_property PART [current_project]] $emmc_ps_cells
set emmc_ps [lindex $emmc_ps_cells 0]
if {[llength [get_bd_cells -quiet soundjie_emmc_io_0]]} { error "eMMC boundary already exists" }
set_property -dict [list CONFIG.PCW_SD1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD1_SD1_IO {EMIO} \
    CONFIG.PCW_SD1_GRP_CD_ENABLE {1} CONFIG.PCW_SD1_GRP_CD_IO {EMIO} \
    CONFIG.PCW_SD1_GRP_WP_ENABLE {1} CONFIG.PCW_SD1_GRP_WP_IO {EMIO}] $emmc_ps
add_files -norecurse [file join $emmc_root rtl soundjie_emmc_io.v]
update_compile_order -fileset sources_1
create_bd_cell -type module -reference soundjie_emmc_io soundjie_emmc_io_0
foreach {ps_pin io_pin} {
    SDIO1_CLK ps_clk SDIO1_CLK_FB ps_clk_fb
    SDIO1_CMD_O ps_cmd_o SDIO1_CMD_T ps_cmd_t SDIO1_CMD_I ps_cmd_i
    SDIO1_DATA_O ps_data_o SDIO1_DATA_T ps_data_t SDIO1_DATA_I ps_data_i
    SDIO1_CDN ps_cd_n SDIO1_WP ps_wp
} {
    connect_bd_net [get_bd_pins $emmc_ps/$ps_pin] [get_bd_pins soundjie_emmc_io_0/$io_pin]
}
foreach {p dir} {emmc_clk O emmc_cmd IO emmc_cd_n I emmc_wp I} {
    create_bd_port -dir $dir $p
    connect_bd_net [get_bd_ports $p] [get_bd_pins soundjie_emmc_io_0/$p]
}
create_bd_port -dir IO -from 3 -to 0 emmc_data
connect_bd_net [get_bd_ports emmc_data] [get_bd_pins soundjie_emmc_io_0/emmc_data]
validate_bd_design
source [file join $emmc_root tests vivado test_soundjie_emmc_bd.tcl]
save_bd_design
