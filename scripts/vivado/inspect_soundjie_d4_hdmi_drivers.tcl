# Read-only inspection of a completed checkpoint; no design mutations or export.
if {$argc != 1 || [version -short] ne "2017.4"} { error "requires Vivado 2017.4 and CHECKPOINT" }
open_checkpoint [lindex $argv 0]
foreach p [get_ports HDMI_*] {
    puts "INSPECT_PORT=$p"
    set nets [get_nets -of_objects $p]
    puts "TOP_NETS=$nets"
    set direct [get_pins -quiet -of_objects $nets -filter {DIRECTION == OUT}]
    puts "DIRECT_DRIVERS=$direct"
    set segments [get_nets -segments -of_objects $p]
    puts "SEGMENTS=$segments"
    set leaf [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
    foreach pin $leaf {
        set cell [get_cells -of_objects $pin]
        puts "LEAF_DRIVER=$pin;CELL=$cell;REF=[get_property REF_NAME $cell]"
    }
}
exit 0
