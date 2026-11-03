# Source only into a fresh, staged CH24 block design. Never touches vendor files.
if {[llength [get_bd_cells -quiet soundjie_hdmi_tx]]} { error "HDMI already integrated" }
set pixel_clock [get_bd_cells clk_wiz_0]
set fclk_mhz [expr {[get_property CONFIG.FREQ_HZ [get_bd_pins processing_system7_0/FCLK_CLK0]] / 1e6}]
# Preserve PS/DPU clocks. Integer output dividers keep the serializer exactly 5x.
# 142.857132 MHz * 26 / 5 / 10 = 74.28570864 MHz (about +481 ppm vs 74.25).
set pixel_target [expr {$fclk_mhz * 26.0 / 50.0}]
set serial_target [expr {$pixel_target * 5.0}]
if {abs($pixel_target / 74.25 - 1.0) >= 0.001} { error "unexpected PS clock for HDMI divider plan" }
set_property -dict [list CONFIG.PRIM_IN_FREQ $fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $pixel_target \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $serial_target CONFIG.CLKOUT2_USED true \
    CONFIG.NUM_OUT_CLKS 2 CONFIG.USE_LOCKED true CONFIG.USE_RESET false] $pixel_clock

set_property -dict [list CONFIG.VIDEO_MODE Custom \
    CONFIG.GEN_HACTIVE_SIZE 1280 CONFIG.GEN_VACTIVE_SIZE 720 \
    CONFIG.GEN_HFRAME_SIZE 1650 CONFIG.GEN_HSYNC_START 1390 CONFIG.GEN_HSYNC_END 1430 \
    CONFIG.GEN_F0_VFRAME_SIZE 750 CONFIG.GEN_F0_VSYNC_VSTART 725 CONFIG.GEN_F0_VSYNC_VEND 730 \
    CONFIG.GEN_F0_VBLANK_HSTART 1280 CONFIG.GEN_F0_VBLANK_HEND 1280 \
    CONFIG.GEN_F0_VSYNC_HSTART 1280 CONFIG.GEN_F0_VSYNC_HEND 1280 \
    CONFIG.GEN_HSYNC_POLARITY High CONFIG.GEN_VSYNC_POLARITY High \
    CONFIG.GEN_INTERLACED false] [get_bd_cells v_tc_0]

set tx [create_bd_cell -type ip -vlnv xilinx.com:user:HDMI_FPGA_ML:1.0 soundjie_hdmi_tx]
foreach {dst src} {VGA_RGB video_data VGA_HS video_hsync_o VGA_VS video_vsync_o VGA_DE video_de_o} {
    connect_bd_net [get_bd_pins axis2video_0/$src] [get_bd_pins $tx/$dst]
}
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins $tx/PXLCLK_I]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins $tx/PXLCLK_5X_I]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins $tx/LOCKED_I]

set vr [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 soundjie_video_reset]
set_property CONFIG.C_EXT_RESET_HIGH 0 $vr
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins $vr/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins $vr/dcm_locked]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins $vr/ext_reset_in]
foreach {target source} {v_tc_0/resetn peripheral_aresetn axis2video_0/video_rst peripheral_reset} {
    set pin [get_bd_pins $target]
    set old [get_bd_nets -quiet -of_objects $pin]
    if {[llength $old]} { disconnect_bd_net $old $pin }
    connect_bd_net [get_bd_pins $vr/$source] $pin
}
connect_bd_net [get_bd_pins $vr/peripheral_aresetn] [get_bd_pins $tx/RST_N]
foreach p {HDMI_CLK_P HDMI_CLK_N HDMI_D0_P HDMI_D0_N HDMI_D1_P HDMI_D1_N HDMI_D2_P HDMI_D2_N} {
    create_bd_port -dir O $p
    connect_bd_net [get_bd_ports $p] [get_bd_pins $tx/$p]
}
