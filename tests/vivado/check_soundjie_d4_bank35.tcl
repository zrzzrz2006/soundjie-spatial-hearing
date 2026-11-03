# Run on an actual optimized checkpoint. Optional XDC is evaluated before DRC.
if {[version -short] ne "2017.4" || $argc < 2} { error "requires Vivado 2017.4: CHECKPOINT REPORT ?XDC?" }
open_checkpoint [lindex $argv 0]
if {$argc == 3} { read_xdc [lindex $argv 2] }
report_drc -checks [get_drc_checks BIVC-1] -file [lindex $argv 1]
set conflicts [get_drc_violations -quiet -filter {NAME =~ "BIVC-1*"}]
puts "BANK35_CONFLICTS=[llength $conflicts]"
foreach p [get_ports {lcd_gpio_tri_io[1] HDMI_CLK_P}] {
    puts "BANK35_PORT=$p,[get_property PACKAGE_PIN $p],[get_property IOSTANDARD $p]"
}
if {[llength $conflicts]} { error "bank voltage conflict remains" }
if {[get_property IOSTANDARD [get_ports {lcd_gpio_tri_io[1]}]] ne "LVCMOS18"} { error "E12 must match actual 1.8V bank35" }
puts "BANK35_DRC=pass_not_board_tested"
exit 0
