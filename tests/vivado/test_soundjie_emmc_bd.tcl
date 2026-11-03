# Run against the real BD, not a mocked Vivado command surface.
proc emmc_assert {ok message} {
    if {!$ok} { error "EMMC_CONTRACT_FAIL: $message" }
}
set emmc_ps [lindex [get_bd_cells -hier -filter {VLNV =~ "xilinx.com:ip:processing_system7:*"}] 0]
set emmc_io [get_bd_cells -quiet soundjie_emmc_io_0]
emmc_assert [expr {[llength $emmc_io] == 1}] "missing physical EMIO boundary"
foreach {ps_pin io_pin} {
    SDIO1_CLK ps_clk SDIO1_CLK_FB ps_clk_fb
    SDIO1_CMD_O ps_cmd_o SDIO1_CMD_T ps_cmd_t SDIO1_CMD_I ps_cmd_i
    SDIO1_DATA_O ps_data_o SDIO1_DATA_T ps_data_t SDIO1_DATA_I ps_data_i
    SDIO1_CDN ps_cd_n SDIO1_WP ps_wp
} {
    set a [get_bd_nets -quiet -of_objects [get_bd_pins $emmc_ps/$ps_pin]]
    set b [get_bd_nets -quiet -of_objects [get_bd_pins $emmc_io/$io_pin]]
    emmc_assert [expr {[llength $a] == 1 && $a eq $b}] "wrong connection $ps_pin"
}
foreach p {emmc_clk emmc_cmd emmc_data emmc_cd_n emmc_wp} {
    set a [get_bd_nets -quiet -of_objects [get_bd_ports -quiet $p]]
    set b [get_bd_nets -quiet -of_objects [get_bd_pins $emmc_io/$p]]
    emmc_assert [expr {[llength $a] == 1 && $a eq $b}] "missing physical port $p"
}
puts "EMMC_BD_CONTRACT_PASS_NOT_BOARD_VALIDATED"
