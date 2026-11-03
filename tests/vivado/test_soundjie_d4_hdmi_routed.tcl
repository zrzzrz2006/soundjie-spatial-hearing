# Real routed-design regression: valid hierarchical HDMI must pass; disconnect
# a port in memory and require the SAME production audit to reject it.
if {$argc != 2 || [version -short] ne "2017.4"} { error "requires CHECKPOINT REPORT" }
set report_path [lindex $argv 1]
if {[file exists $report_path]} { error "refusing existing report" }
open_checkpoint [lindex $argv 0]
set fp [open $report_path w]
proc audit_emit {fp key value} { puts $fp "$key=$value" }
proc audit_fail {fp message} { global errors; lappend errors $message; puts $fp "FAILURE=$message" }
set production_report [file normalize [file join [file dirname [info script]] .. .. scripts vivado report_soundjie_d4_hdmi.tcl]]
set errors {}
source $production_report
flush $fp
if {[llength $errors]} { close $fp; error "valid routed HDMI rejected: $errors" }
puts $fp "POSITIVE_AUDIT=pass"
set port [get_ports HDMI_D0_P]
set port_net [get_nets -of_objects $port]
disconnect_net -net $port_net -objects $port
set errors {}
source $production_report
flush $fp
if {[lsearch -exact $errors "HDMI output not driven by differential buffer HDMI_D0_P"] < 0} {
    close $fp
    error "disconnected HDMI output was not rejected"
}
puts $fp "DISCONNECTED_PORT_REJECTED=pass"
close $fp
puts "HDMI_ROUTED_REGRESSION=pass_not_board_tested"
exit 0
