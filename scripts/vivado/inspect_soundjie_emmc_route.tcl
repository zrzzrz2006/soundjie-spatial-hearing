# Read-only inspection of an explicitly supplied routed DCP. No write_bitstream.
if {$argc != 2} { error "usage: routed.dcp new-output-directory" }
set checkpoint [lindex $argv 0]
set out [lindex $argv 1]
if {[file exists $out]} { error "refuse to overwrite inspection directory" }
file mkdir $out
open_checkpoint $checkpoint
set fp [open [file join $out emmc-route.txt] {WRONLY CREAT EXCL}]
puts $fp "PART=[get_property PART [current_design]]"
puts $fp "UNUSEDPIN=[get_property BITSTREAM.CONFIG.UNUSEDPIN [current_design]]"
set failures 0
foreach {name pin direction} {
    emmc_clk A15 OUT emmc_cmd A17 INOUT
    emmc_data[0] B12 INOUT emmc_data[1] C12 INOUT
    emmc_data[2] C11 INOUT emmc_data[3] A14 INOUT
    emmc_cd_n L14 IN emmc_wp L15 IN
} {
    set port [get_ports -quiet $name]
    if {[llength $port] != 1} {
        puts $fp "FAIL=missing port $name"
        incr failures
        continue
    }
    set actual_pin [get_property PACKAGE_PIN $port]
    set actual_standard [get_property IOSTANDARD $port]
    set actual_direction [get_property DIRECTION $port]
    puts $fp "PORT=$name PIN=$actual_pin STANDARD=$actual_standard DIRECTION=$actual_direction PULLUP=[get_property PULLUP $port]"
    if {$actual_pin ne $pin || $actual_standard ne "LVCMOS18" || $actual_direction ne $direction} {
        puts $fp "FAIL=port contract $name"
        incr failures
    }
}
set buffers [get_cells -quiet -hier -filter {REF_NAME == IOBUF && NAME =~ *soundjie_emmc_io*}]
puts $fp "IOBUF_COUNT=[llength $buffers]"
if {[llength $buffers] != 5} { incr failures; puts $fp "FAIL=IOBUF count" }
foreach cell $buffers {
    puts $fp "BUFFER=$cell"
    foreach pin [get_pins -of_objects $cell] {
        puts $fp "  PIN=$pin NET=[get_nets -segments -of_objects $pin]"
    }
}
foreach ps [get_cells -quiet -hier -filter {REF_NAME == PS7}] {
    foreach pin [get_pins -of_objects $ps -filter {REF_PIN_NAME =~ *SDIO1*}] {
        puts $fp "PS_PIN=$pin NET=[get_nets -segments -of_objects $pin]"
    }
}
foreach cell [get_cells -quiet -hier -filter {NAME =~ *soundjie_emmc_io* && IS_PRIMITIVE == 1}] {
    set ref [get_property REF_NAME $cell]
    puts $fp "PRIMITIVE=$cell REF=$ref"
    if {$ref eq "LUT1"} { puts $fp "  INIT=[get_property INIT $cell]" }
    foreach pin [get_pins -of_objects $cell] {
        puts $fp "  PIN=$pin NET=[get_nets -segments -of_objects $pin]"
    }
}
puts $fp "B17_PORTS=[get_ports -quiet -filter {PACKAGE_PIN == B17}]"
puts $fp "STRUCTURAL_FAILURES=$failures"
puts $fp "BOARD_VALIDATED=0"
close $fp
report_io -file [file join $out io.rpt]
report_drc -file [file join $out drc.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $out timing.rpt]
check_timing -verbose -file [file join $out check_timing.rpt]
close_design
if {$failures} { error "eMMC route structural contract failed" }
puts "EMMC_ROUTE_INSPECTED_NOT_BOARD_VALIDATED"
exit 0
