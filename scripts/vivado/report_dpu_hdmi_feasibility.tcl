# Vivado 2017.4 read-only feasibility reporter for an already-open copied project.
# This script reports conflicts and never modifies the block design.
if {[info exists ::env(SOUNDJIE_HDMI_REPORT)]} {
    set report_path $::env(SOUNDJIE_HDMI_REPORT)
} else {
    set report_path "dpu-hdmi-vivado-report.txt"
}
set fp [open $report_path w]
proc emit {fp key value} { puts $fp "$key=$value" }

set dpu_cells [get_bd_cells -quiet -hier -filter {VLNV =~ "*:*:dpu*:*" || NAME =~ "*dpu*"}]
set vdma_cells [get_bd_cells -quiet -hier -filter {VLNV =~ "xilinx.com:ip:axi_vdma:*" || NAME =~ "*vdma*"}]
emit $fp DPU_CELL [expr {[llength $dpu_cells] ? [join $dpu_cells ,] : "none"}]
emit $fp VDMA_CELL [expr {[llength $vdma_cells] ? [join $vdma_cells ,] : "none"}]

set dpu_hp {}
set display_hp {}
foreach net [get_bd_intf_nets -quiet -hier] {
    set pins [join [get_bd_intf_pins -quiet -of_objects $net] " "]
    foreach hp {HP0 HP1 HP2 HP3} {
        if {[string match -nocase "*S_AXI_${hp}*" $pins]} {
            if {[string match -nocase "*dpu*" $pins]} { lappend dpu_hp $hp }
            if {[string match -nocase "*vdma*" $pins]} { lappend display_hp $hp }
        }
    }
}
set dpu_hp [lsort -unique $dpu_hp]
set display_hp [lsort -unique $display_hp]
set hp_conflicts {}
foreach hp $dpu_hp { if {[lsearch -exact $display_hp $hp] >= 0} { lappend hp_conflicts $hp } }
emit $fp DPU_HP_PORTS [expr {[llength $dpu_hp] ? [join $dpu_hp ,] : "none"}]
emit $fp DISPLAY_HP_PORTS [expr {[llength $display_hp] ? [join $display_hp ,] : "none"}]
emit $fp ADDRESS_CONFLICTS [expr {[llength $hp_conflicts] ? [join $hp_conflicts ,] : "none_detected"}]
emit $fp IRQ_CONFLICTS manual_review_required
emit $fp CLOCK_CONFLICTS manual_review_required

set lut_percent unavailable_until_synthesis
set bram_percent unavailable_until_synthesis
set dsp_percent unavailable_until_synthesis
if {[llength [get_runs -quiet impl_1]] && [get_property PROGRESS [get_runs impl_1]] eq "100%"} {
    open_run impl_1
    foreach {key site_type var_name} {LUT_PERCENT LUT lut_percent BRAM_PERCENT RAMB36 bram_percent DSP_PERCENT DSP48 dsp_percent} {
        set used [llength [get_cells -quiet -hier -filter "PRIMITIVE_TYPE =~ *$site_type*"]]
        set sites [llength [get_sites -quiet -filter "SITE_TYPE =~ *$site_type*"]]
        if {$sites > 0} { set $var_name [format "%.2f" [expr {100.0 * $used / $sites}]] }
    }
}
emit $fp LUT_PERCENT $lut_percent
emit $fp BRAM_PERCENT $bram_percent
emit $fp DSP_PERCENT $dsp_percent
if {![llength $dpu_cells]} {
    set result "blocked_missing_dpu"
} elseif {![llength $vdma_cells]} {
    set result "blocked_missing_display_source"
} elseif {[llength $hp_conflicts]} {
    set result "blocked_hp_conflict"
} else {
    set result "pass_static_needs_synthesis"
}
emit $fp RESULT $result
close $fp
puts "SOUNDJIE_DPU_HDMI_REPORT=$report_path"
puts "RESULT=$result"
