# CH24 axis2video.v implements video_vsync_i -> axis_clk as a three-stage
# synchronizer named axis_vsync_reg[0:2]. Only the asynchronous path into
# stage 0 is false; later stages and every other cross-domain path stay timed.
set sj_axis_vsync_regs [get_cells -quiet -hier -regexp {.*axis2video_0.*/axis_vsync_reg\[[0-2]\]}]
set_property ASYNC_REG TRUE $sj_axis_vsync_regs

set sj_axis_vsync_stage0_d [get_pins -quiet -hier -regexp {.*axis2video_0.*/axis_vsync_reg\[0\]/D}]
set_false_path -to $sj_axis_vsync_stage0_d
