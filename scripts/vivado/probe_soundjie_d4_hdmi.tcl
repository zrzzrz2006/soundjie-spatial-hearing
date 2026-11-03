# Read the fresh staged design and device pin database; never export a BIT.
if {[version -short] ne "2017.4" || $argc != 1} {
    error "Vivado 2017.4 and one staged-root argument required"
}
set root [file normalize [lindex $argv 0]]
set project [file join $root sources ch24 CH24_LCD_Tourch 02_fpga fpga system system.xpr]
open_project $project
if {[get_property PART [current_project]] ne "xc7z100ffg900-2"} {
    error "PART_MISMATCH"
}
set_property ip_repo_paths [list \
    [file join $root sources dpu ip_repo dpu] \
    [file join $root sources ch24 CH24_LCD_Tourch 02_fpga fpga ip_repo]] [current_project]
update_ip_catalog
open_bd_design [lindex [get_files */system.bd] 0]
puts "=== HDMI IP DEFINITIONS ==="
puts [get_ipdefs -all *HDMI*]
puts "=== DISPLAY CONFIGURATION ==="
foreach cell [get_bd_cells] {
    if {[regexp -nocase {clk_wiz|v_tc|axis2video|vdma|HDMI} $cell]} {
        report_property $cell
        foreach pin [get_bd_pins -of_objects $cell] {
            puts "PIN $pin NET [get_bd_nets -quiet -of_objects $pin]"
        }
    }
}
puts "=== EXTERNAL PORTS ==="
foreach port [get_bd_ports] {
    puts "PORT $port NET [get_bd_nets -quiet -of_objects $port]"
}
puts "=== DEVICE PACKAGE PINS ==="
close_project
create_project -in_memory -part xc7z100ffg900-2 pin_audit
set_property design_mode PinPlanning [current_fileset]
open_io_design -name pin_audit
foreach pin {D15 D14 G15 G14 C14 C13 F15 F14} {
    report_property [get_package_pins $pin]
}
puts "SOUNDJIE_HDMI_PROBE=complete_no_hardware_validation"
close_project
exit 0
