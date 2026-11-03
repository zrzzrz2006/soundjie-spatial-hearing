if {[version -short] ne "2017.4"} { error "Vivado2017.4 required" }
set root /mnt/hgfs/Soundjie_E/d4
set stage $root/artifacts/pl_rebuild_20261024_emmc_a
set project $stage/sources/ch24/CH24_LCD_Tourch/02_fpga/fpga/system/system.xpr
open_project $project
if {[get_property PART [current_project]] ne "xc7z100ffg900-2"} { error "wrong part" }
set_property ip_repo_paths [list $stage/sources/dpu/ip_repo/dpu \
    $stage/sources/ch24/CH24_LCD_Tourch/02_fpga/fpga/ip_repo] [current_project]
update_ip_catalog
set bd [get_files */system.bd]
open_bd_design $bd
source $root/tests/vivado/test_soundjie_d4_hdmi_bd.tcl
source $root/scripts/vivado/integrate_soundjie_emmc.tcl
add_files -fileset constrs_1 $root/constraints/soundjie_emmc.xdc
generate_target all $bd
set wrappers [make_wrapper -files $bd -top]
add_files -norecurse $wrappers
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1
source $root/tests/vivado/test_soundjie_emmc_bd.tcl
source $root/tests/vivado/test_soundjie_d4_hdmi_bd.tcl
write_bd_tcl $stage/emmc_candidate_bd.tcl
puts "EMMC_CANDIDATE_PREPARED_NOT_IMPLEMENTED"
close_project
exit 0
