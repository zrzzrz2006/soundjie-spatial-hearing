# Integration assertions against the actual open Vivado block design.
proc hdmi_assert {ok message} {
    if {!$ok} { error "HDMI_CONTRACT_FAIL: $message" }
}
proc hdmi_same_net {a b} {
    set na [get_bd_nets -quiet -of_objects [get_bd_pins $a]]
    set nb [get_bd_nets -quiet -of_objects [get_bd_pins $b]]
    hdmi_assert [expr {[llength $na] == 1 && $na eq $nb}] "$a not connected to $b"
}
set tx [get_bd_cells -quiet soundjie_hdmi_tx]
hdmi_assert [expr {[llength $tx] == 1}] "actual HDMI transmitter is absent"
hdmi_assert [expr {[get_property VLNV $tx] eq "xilinx.com:user:HDMI_FPGA_ML:1.0"}] "wrong transmitter"
foreach p {HDMI_CLK_P HDMI_CLK_N HDMI_D0_P HDMI_D0_N HDMI_D1_P HDMI_D1_N HDMI_D2_P HDMI_D2_N} {
    set port [get_bd_ports -quiet $p]
    hdmi_assert [expr {[llength $port] == 1}] "missing output $p"
    set net [get_bd_nets -quiet -of_objects $port]
    hdmi_assert [expr {[llength $net] == 1 && $net eq [get_bd_nets -of_objects [get_bd_pins $tx/$p]]}] "undriven $p"
}
foreach {dst src} {VGA_RGB video_data VGA_HS video_hsync_o VGA_VS video_vsync_o VGA_DE video_de_o} {
    hdmi_same_net $tx/$dst axis2video_0/$src
}
hdmi_same_net $tx/PXLCLK_I clk_wiz_0/clk_out1
hdmi_same_net $tx/PXLCLK_5X_I clk_wiz_0/clk_out2
hdmi_same_net $tx/LOCKED_I clk_wiz_0/locked
hdmi_same_net $tx/RST_N soundjie_video_reset/peripheral_aresetn
hdmi_same_net axis2video_0/video_rst soundjie_video_reset/peripheral_reset
hdmi_same_net v_tc_0/resetn soundjie_video_reset/peripheral_aresetn
foreach {prop want} {
    GEN_HACTIVE_SIZE 1280 GEN_VACTIVE_SIZE 720 GEN_HFRAME_SIZE 1650
    GEN_HSYNC_START 1390 GEN_HSYNC_END 1430 GEN_F0_VFRAME_SIZE 750
    GEN_F0_VSYNC_VSTART 725 GEN_F0_VSYNC_VEND 730
} {
    hdmi_assert [expr {[get_property CONFIG.$prop [get_bd_cells v_tc_0]] == $want}] "wrong timing $prop"
}
set c [get_bd_cells clk_wiz_0]
set fin [get_property CONFIG.PRIM_IN_FREQ $c]
set vco [expr {$fin * [get_property CONFIG.MMCM_CLKFBOUT_MULT_F $c] / [get_property CONFIG.MMCM_DIVCLK_DIVIDE $c]}]
set pixel [expr {$vco / [get_property CONFIG.MMCM_CLKOUT0_DIVIDE_F $c]}]
set serial [expr {$vco / [get_property CONFIG.MMCM_CLKOUT1_DIVIDE $c]}]
hdmi_assert [expr {abs($serial / $pixel - 5.0) < 1e-6}] "serial/pixel ratio must be exactly 5"
hdmi_assert [expr {abs($pixel / 74.25 - 1.0) < 0.001}] "pixel clock outside 0.1 percent budget"
puts "HDMI_BD_CLOCKS pixel_mhz=$pixel serial_mhz=$serial"
puts "HDMI_BD_CONTRACT=pass_not_implemented_or_board_tested"
