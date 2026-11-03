# Fail-closed post-implementation report for the combined DPU + HDMI design.
set FAIL_CLOSED 0
if {![info exists output_dir]} { set output_dir [pwd] }
set report_path [file join $output_dir pl-feasibility-report.txt]
set fp [open $report_path w]
proc audit_emit {fp key value} { puts $fp "$key=$value" }
proc audit_fail {fp message} {
    global FAIL_CLOSED
    set FAIL_CLOSED 1
    audit_emit $fp FAILURE $message
}

set dpu_cells [get_cells -quiet -hier -filter {REF_NAME =~ "dpu_eu*" || NAME =~ "*dpu_eu*"}]
set vdma_cells [get_cells -quiet -hier -filter {REF_NAME =~ "axi_vdma*" || NAME =~ "*axi_vdma*"}]
set framebuffer_cells [get_cells -quiet -hier -filter {NAME =~ "*axis2video*" || NAME =~ "*framebuffer*"}]
audit_emit $fp DPU_IMPLEMENTED [expr {[llength $dpu_cells] > 0 ? 1 : 0}]
audit_emit $fp VDMA_IMPLEMENTED [expr {[llength $vdma_cells] > 0 ? 1 : 0}]
audit_emit $fp FRAMEBUFFER_IMPLEMENTED [expr {[llength $framebuffer_cells] > 0 ? 1 : 0}]
if {![llength $dpu_cells]} { audit_fail $fp "missing dpu_eu" }
if {![llength $vdma_cells]} { audit_fail $fp "missing axi_vdma" }
if {![llength $framebuffer_cells]} { audit_fail $fp "missing framebuffer display pipeline" }
source [file join [file dirname [info script]] report_soundjie_d4_hdmi.tcl]

set cdc_axis_vsync_regs [get_cells -quiet -hier -regexp {.*axis2video_0.*/axis_vsync_reg\[[0-2]\]}]
set cdc_axis_vsync_stage0_pins [get_pins -quiet -hier -regexp {.*axis2video_0.*/axis_vsync_reg\[0\]/D}]
audit_emit $fp CDC_AXIS_VSYNC_REGS [llength $cdc_axis_vsync_regs]
audit_emit $fp CDC_AXIS_VSYNC_STAGE0_PINS [llength $cdc_axis_vsync_stage0_pins]
if {[llength $cdc_axis_vsync_regs] != 3} { audit_fail $fp "axis_vsync synchronizer register count" }
if {[llength $cdc_axis_vsync_stage0_pins] != 1} { audit_fail $fp "axis_vsync stage-0 pin count" }
foreach cdc_reg $cdc_axis_vsync_regs {
    set cdc_async_value [get_property ASYNC_REG $cdc_reg]
    audit_emit $fp CDC_ASYNC_REG $cdc_async_value
    if {![string equal -nocase $cdc_async_value "true"] && $cdc_async_value ne "1"} {
        audit_fail $fp "axis_vsync ASYNC_REG missing"
    }
}

report_utilization -file [file join $output_dir utilization.rpt]
report_drc -file [file join $output_dir drc.rpt]
report_timing_summary -delay_type max -max_paths 20 -file [file join $output_dir timing_summary.rpt]
set timing_paths [get_timing_paths -quiet -delay_type max -max_paths 1]
if {![llength $timing_paths]} {
    set WNS "unavailable"
    audit_fail $fp "timing path unavailable"
} else {
    set WNS [get_property SLACK [lindex $timing_paths 0]]
    if {$WNS < 0.0} { audit_fail $fp "negative WNS" }
}
audit_emit $fp WNS $WNS
audit_emit $fp RESULT [expr {$FAIL_CLOSED ? "FAIL_CLOSED" : "PASS"}]
close $fp
if {$FAIL_CLOSED} { return -code error "combined PL report failed closed" }
