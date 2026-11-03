# Sourced after implementation; audit_emit/audit_fail/fp are defined by caller.
set hdmi_ser [get_cells -quiet -hier -filter {NAME =~ "*soundjie_hdmi_tx*" && REF_NAME == OSERDESE2}]
set hdmi_buf [get_cells -quiet -hier -filter {NAME =~ "*soundjie_hdmi_tx*" && REF_NAME == OBUFDS}]
audit_emit $fp HDMI_OSERDESE2_COUNT [llength $hdmi_ser]
audit_emit $fp HDMI_OBUFDS_COUNT [llength $hdmi_buf]
if {[llength $hdmi_ser] != 8 || [llength $hdmi_buf] != 4} { audit_fail $fp "HDMI serialization primitives missing" }
foreach {name pin} {HDMI_D0_P D15 HDMI_D0_N D14 HDMI_D1_P G15 HDMI_D1_N G14 HDMI_D2_P C14 HDMI_D2_N C13 HDMI_CLK_P F15 HDMI_CLK_N F14} {
    set p [get_ports -quiet $name]
    if {[llength $p] != 1} { audit_fail $fp "missing HDMI port $name"; continue }
    set actual [get_property PACKAGE_PIN $p]
    set standard [get_property IOSTANDARD $p]
    audit_emit $fp $name "$actual,$standard"
    if {$actual ne $pin || $standard ne "LVDS"} { audit_fail $fp "HDMI pin/electrical mismatch $name" }
    # A top-level net ends at hierarchy boundary pins. Follow all connected
    # segments to leaf outputs; a boundary cell is not the physical driver.
    set segments [get_nets -quiet -segments -of_objects $p]
    set drivers [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
    set buffers [get_cells -quiet -of_objects $drivers -filter {REF_NAME == OBUFDS}]
    audit_emit $fp HDMI_OUTPUT_DRIVER "$name,$drivers"
    if {[llength $buffers] != 1} { audit_fail $fp "HDMI output not driven by differential buffer $name" }
}
foreach ser $hdmi_ser {
    set pixel [get_clocks -quiet -of_objects [get_pins $ser/CLKDIV]]
    set serial [get_clocks -quiet -of_objects [get_pins $ser/CLK]]
    if {[llength $pixel] != 1 || [llength $serial] != 1} {
        audit_fail $fp "missing serializer timing clocks $ser"; continue
    }
    set pp [get_property PERIOD $pixel]
    set sp [get_property PERIOD $serial]
    if {abs($pp / $sp - 5.0) > 0.005 || abs((1000.0/$pp)/74.25-1.0) > 0.001} {
        audit_fail $fp "HDMI implemented clock mismatch $ser"
    }
    audit_emit $fp HDMI_SERIALIZER_CLOCKS "$ser,pixel_ns=$pp,serial_ns=$sp"
}
