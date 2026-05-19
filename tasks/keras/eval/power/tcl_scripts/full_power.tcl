# Idiomatic power estimation flow for hls4ml IP blocks (non-project, OOC).
#
# This is the ORCHESTRATOR. It dispatches each stage to its own fresh
# Vivado process. Each child invocation starts clean (no accumulated heap
# state, no fragmented allocations), reads in only the checkpoint it
# needs, does its work, and exits. This avoids the realloc/heap-corruption
# class of bugs in long-lived Vivado sessions and keeps peak memory per
# process tight enough for an 8GB machine.
#
set_param general.maxThreads 4
catch { config_webtalk -user off }

if { $argc != 6 } {
    puts "Usage: vivado -mode batch -source full_power.tcl -tclargs \\"
    puts "         <verilog_dir> <tb_file> <output_dir> <fpga_part> <top_module> <clock_period_ns>"
    exit 1
}

set verilog_dir   [file normalize [lindex $argv 0]]
set tb_file       [file normalize [lindex $argv 1]]
set output_dir    [file normalize [lindex $argv 2]]
set fpga_part     [lindex $argv 3]
set top_module    [lindex $argv 4]
set clock_period  [lindex $argv 5]

set tb_top       "power_tb"
set saif_scope   "/${tb_top}/dut"
set strip_path   "${tb_top}/dut"
set vectors_file "$output_dir/input_vectors.dat"
set script_dir   [file dirname [file normalize [info script]]]

file mkdir $output_dir

puts "================================================================"
puts "  Power estimation orchestrator (multi-process, OOC)"
puts "================================================================"
puts "  Verilog dir:    $verilog_dir"
puts "  TB file:        $tb_file"
puts "  Output dir:     $output_dir"
puts "  FPGA part:      $fpga_part"
puts "  Top module:     $top_module"
puts "  Clock period:   $clock_period ns"
puts "================================================================"

# Locate glbl.v (sanity check before spawning children)
set glbl_v "$::env(XILINX_VIVADO)/data/verilog/src/glbl.v"
if {![file exists $glbl_v]} {
    puts "ERROR: cannot find glbl.v at $glbl_v"
    exit 1
}

# Helper: run a child Vivado in a fresh process, fail loudly on crash
proc run_child_vivado {tag tcl_path args_list} {
    global output_dir
    set log_file "$output_dir/child_${tag}.log"
    set cmd [list vivado -mode batch -nojournal -nolog -source $tcl_path -tclargs {*}$args_list]
    puts "    spawning: vivado -mode batch -source [file tail $tcl_path]"
    puts "    log:      $log_file"
    set rc [catch {
        exec sh -c "[join $cmd { }] > $log_file 2>&1"
    } msg]
    if {$rc != 0} {
        puts "    ERROR: child '$tag' failed (rc=$rc); tail of $log_file:"
        catch {exec tail -n 60 $log_file} tail
        puts $tail
        return 0
    }
    return 1
}

# ================================================================
# STAGE 1: Synthesis (OOC) -> post_synth.dcp + funcsim netlist
# ================================================================
puts "\n>>> STAGE 1: synthesis (child process)"
if {![run_child_vivado "synth" "$script_dir/stage_synth.tcl" \
        [list $verilog_dir $output_dir $fpga_part $top_module $clock_period]]} {
    puts "FATAL: synthesis stage failed"
    exit 1
}


# ================================================================
# STAGE 2: xsim functional + timing -> SAIFs
# (xsim already runs as separate child processes, no change needed)
# ================================================================
proc run_xsim_stage {stage_dir netlist_v tb_file glbl_v sdf_arg saif_path saif_scope vectors_src} {
    file mkdir $stage_dir
    file copy -force $vectors_src "$stage_dir/input_vectors.dat"

    set run_tcl "$stage_dir/run.tcl"
    set fh [open $run_tcl w]
    puts $fh "open_saif \"$saif_path\""
    puts $fh "log_saif \[get_objects -r $saif_scope/* \]"
    puts $fh "run all"
    puts $fh "close_saif"
    puts $fh "quit"
    close $fh

    set snapshot "power_tb_snap"
    set xvlog_log "$stage_dir/xvlog.log"
    set xelab_log "$stage_dir/xelab.log"
    set xsim_log  "$stage_dir/xsim.log"

    puts "    xvlog (compiling)..."
    set xvlog_cmd "cd $stage_dir && xvlog -work xil_defaultlib $netlist_v $tb_file $glbl_v"
    set rc [catch { exec sh -c "$xvlog_cmd > $xvlog_log 2>&1" } msg]
    if {$rc != 0} {
        puts "    ERROR: xvlog failed (rc=$rc); see $xvlog_log"
        catch {exec tail -n 40 $xvlog_log} tail
        puts $tail
        return 0
    }

    puts "    xelab (elaborating)..."
    set xelab_flags "-L simprims_ver -L secureip -L unisims_ver --debug typical --relax -s $snapshot"
    if {$sdf_arg ne ""} {
        set xelab_flags "$xelab_flags --transport_int_delays --pulse_r 100 --pulse_int_r 100 --pulse_e 100 --pulse_int_e 100 $sdf_arg"
    }
    set xelab_cmd "cd $stage_dir && xelab $xelab_flags xil_defaultlib.power_tb xil_defaultlib.glbl"
    set rc [catch { exec sh -c "$xelab_cmd > $xelab_log 2>&1" } msg]
    if {$rc != 0} {
        puts "    ERROR: xelab failed (rc=$rc); see $xelab_log"
        catch {exec tail -n 40 $xelab_log} tail
        puts $tail
        return 0
    }

    puts "    xsim (running)..."
    set xsim_cmd "cd $stage_dir && xsim $snapshot -tclbatch run.tcl"
    set rc [catch { exec sh -c "$xsim_cmd > $xsim_log 2>&1" } msg]
    if {$rc != 0} {
        puts "    ERROR: xsim failed (rc=$rc); see $xsim_log"
        catch {exec tail -n 40 $xsim_log} tail
        puts $tail
        return 0
    }

    if {[file exists $saif_path]} {
        set sz [expr {[file size $saif_path] / 1024}]
        puts "    SAIF written: $saif_path (${sz} KB)"
        return 1
    } else {
        puts "    ERROR: SAIF not produced at $saif_path"
        return 0
    }
}

set saif_synth "$output_dir/switching_synth.saif"
set orig_pwd [pwd]

# ================================================================
# STAGE 2: post-synth functional sim + SAIF — PRIMARY activity source.
# Drives xsim on the post-synth funcsim netlist (no SDF, no timing
# annotation) and logs SAIF for STAGE 4's report_power. Faster than
# the impl-side timing SAIF since we skip place+route and SDF
# annotation, at the cost of zero-delay activity estimates.
# ================================================================
puts "\n>>> STAGE 2: post-synth functional sim + SAIF"
set synth_sim_dir "$output_dir/sim_synth"
run_xsim_stage \
    $synth_sim_dir \
    "$output_dir/post_synth_funcsim.v" \
    $tb_file \
    $glbl_v \
    "" \
    $saif_synth \
    $saif_scope \
    $vectors_file
cd $orig_pwd


# ================================================================
# STAGE 3: Sanity check impl captures + SAIF header
# ================================================================
puts "\n>>> STAGE 3: sanity check"

proc capture_sanity {capture_path label} {
    if {![file exists $capture_path]} {
        puts "  \[$label\] CAPTURE MISSING: $capture_path"
        return 0
    }
    set fh [open $capture_path r]
    set count 0
    while {[gets $fh line] >= 0} {
        if {[string length [string trim $line]] > 0} { incr count }
    }
    close $fh
    puts "  \[$label\] captured samples: $count"
    return $count
}

proc saif_header {saif_path label} {
    if {![file exists $saif_path]} {
        puts "  \[$label\] SAIF MISSING"
        return
    }
    set sz [expr {[file size $saif_path] / 1024}]
    puts "  \[$label\] size: ${sz} KB"
    set fh [open $saif_path r]
    set lines 0
    set printed 0
    while {[gets $fh line] >= 0} {
        incr lines
        if {[string match "*INSTANCE*" $line]} {
            if {$printed < 8} {
                puts "    $line"
                incr printed
            }
        }
        if {$lines > 2000} break
    }
    close $fh
    puts "  \[$label\] (showed first $printed of many INSTANCE lines)"
}

capture_sanity "$synth_sim_dir/output_captured.dat" "synth"

foreach {src dst} [list \
    "$synth_sim_dir/output_captured.dat"  "$output_dir/output_captured_synth.dat" \
] {
    if {[file exists $src]} { file copy -force $src $dst }
}

puts ">>> SAIF headers:"
saif_header $saif_synth "synth"

# ================================================================
# STAGE 4: report_power (post-synth + synth SAIF) — child process
# Consumes the functional SAIF from STAGE 2 against the post-synth
# checkpoint. Produces power_synth.{xml,verbose.txt,hierarchical.txt}
# plus switching_synth_breakdown.txt.
# ================================================================
puts "\n>>> STAGE 4: report_power (post-synth + synth SAIF) (child process)"
if {![run_child_vivado "power_synth" "$script_dir/stage_power.tcl" \
        [list $output_dir post_synth.dcp $saif_synth $strip_path power_synth]]} {
    puts "WARN: post-synth power report failed (continuing)"
}

# ================================================================
# STAGE 5: Utilization + timing reports — child process
# ================================================================
puts "\n>>> STAGE 5: utilization + timing reports (child process)"
if {![run_child_vivado "reports" "$script_dir/stage_reports.tcl" \
        [list $output_dir]]} {
    puts "WARN: utilization/timing reports failed"
}

puts "\n================================================================"
puts "  All stages complete"
puts "================================================================"
puts "  Synth capture:        $output_dir/output_captured_synth.dat"
puts "  Synth SAIF:           $saif_synth"
puts "  Synth power XML:      $output_dir/power_synth.xml"
puts "  Synth power verbose:  $output_dir/power_synth_verbose.txt"
puts "  Utilization XML:      $output_dir/utilization.xml"
puts "  Timing summary:       $output_dir/timing_summary.txt"
puts "================================================================"

exit 0