if {[version -short] ne "2017.4" || $argc < 1} { error "2017.4 and staged root required" }
set root [file normalize [lindex $argv 0]]
set repo [file normalize [file join [file dirname [info script]] .. ..]]
open_project [file join $root sources ch24 CH24_LCD_Tourch 02_fpga fpga system system.xpr]
if {[get_property PART [current_project]] ne "xc7z100ffg900-2"} { error "PART_MISMATCH" }
set_property ip_repo_paths [list [file join $root sources dpu ip_repo dpu] \
    [file join $root sources ch24 CH24_LCD_Tourch 02_fpga fpga ip_repo]] [current_project]
update_ip_catalog
open_bd_design [lindex [get_files */system.bd] 0]
if {$argc == 2 && [lindex $argv 1] eq "apply"} {
    source [file join $repo scripts vivado integrate_soundjie_d4_hdmi.tcl]
    validate_bd_design
}
source [file join $repo tests vivado test_soundjie_d4_hdmi_bd.tcl]
if {$argc == 2 && [lindex $argv 1] eq "apply"} { save_bd_design }
close_project
exit 0
