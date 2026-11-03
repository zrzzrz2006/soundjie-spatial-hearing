# Pure scope predicate, tested in Tcl independently of Vivado.
proc emmc_candidate_guard {project expected part ps_cells} {
    if {[file normalize $project] ne [file normalize $expected]} {
        error "CANDIDATE_SCOPE: refusing non-candidate project"
    }
    if {$part ne "xc7z100ffg900-2" || [llength $ps_cells] != 1} {
        error "CANDIDATE_SCOPE: wrong device or ambiguous PS"
    }
    return candidate_scope_verified
}
