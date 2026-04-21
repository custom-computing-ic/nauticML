#!/usr/bin/tclsh
# Power Estimation TCL Script for hls4ml IP Blocks
# Supports vectorless, post-synth SAIF, and post-impl SAIF modes

if { $argc < 3 || $argc > 6 } {
    puts "Error: Invalid number of arguments"
    puts "Usage: vivado -mode batch -source power_estimation_ip.tcl -tclargs \\"
    puts "         <verilog_dir> <output_dir> <fpga_part> \[top_module\] \[power_mode\] \[tb_file\]"
    puts ""
    puts "  power_mode: vectorless (default) | saif_synth | saif_impl"
    puts "  tb_file:    required for saif_synth and saif_impl"
    exit 1
}

set verilog_dir [file normalize [lindex $argv 0]]
set output_dir  [file normalize [lindex $argv 1]]
set fpga_part   [lindex $argv 2]

set top_module ""
set power_mode "vectorless"
set tb_file ""

if { $argc >= 4 } { set top_module [lindex $argv 3] }
if { $argc >= 5 } { set power_mode [lindex $argv 4] }
if { $argc >= 6 } { set tb_file    [file normalize [lindex $argv 5]] }

if { $power_mode ni {vectorless saif_synth saif_impl} } {
    puts "Error: power_mode must be one of: vectorless, saif_synth, saif_impl"
    exit 1
}

if { $power_mode ne "vectorless" && $tb_file eq "" } {
    puts "Error: tb_file is required for SAIF modes"
    exit 1
}

if { $power_mode ne "vectorless" && ![file exists $tb_file] } {
    puts "Error: testbench not found: $tb_file"
    exit 1
}

file mkdir $output_dir

puts "========================================="
puts "HLS4ML IP Power Estimation Script"
puts "========================================="
puts "Verilog Directory: $verilog_dir"
puts "Output Directory:  $output_dir"
puts "Target Device:     $fpga_part"
puts "Power Mode:        $power_mode"
if { $tb_file ne "" } { puts "Testbench:         $tb_file" }
puts "========================================="

proc find_top_module {verilog_dir} {
    set candidates {}
    foreach vfile [glob -nocomplain "$verilog_dir/*.v"] {
        set basename [file rootname [file tail $vfile]]
        if { ![string match "*_ap_*" $basename] &&
             ![string match "*_mul_*" $basename] &&
             ![string match "*_add_*" $basename] &&
             ![string match "*_sub_*" $basename] &&
             ![string match "*_mac_*" $basename] &&
             ![string match "*_mux_*" $basename] } {
            lappend candidates $basename
        }
    }
    if { [llength $candidates] > 0 } {
        return [lindex $candidates 0]
    }
    return ""
}

if { ![file exists $verilog_dir] } {
    puts "Error: Verilog directory not found: $verilog_dir"
    exit 1
}

# ============================================================================
# PROJECT SETUP
# ============================================================================
set project_name "power_est_[clock seconds]"
create_project $project_name "$output_dir/$project_name" -part $fpga_part -force

puts "Adding design files..."
set v_files  [glob -nocomplain "$verilog_dir/*.v"]
set sv_files [glob -nocomplain "$verilog_dir/*.sv"]
set vh_files [glob -nocomplain "$verilog_dir/*.vh"]

if { [llength $v_files] == 0 && [llength $sv_files] == 0 } {
    puts "Error: No Verilog files found in $verilog_dir"
    close_project
    exit 1
}

if { [llength $v_files]  > 0 } { add_files $v_files }
if { [llength $sv_files] > 0 } { add_files $sv_files }
if { [llength $vh_files] > 0 } { add_files -fileset [current_fileset] $vh_files }

if { $top_module eq "" } {
    set top_module [find_top_module $verilog_dir]
}
if { $top_module eq "" } {
    puts "Error: Could not determine top module"
    close_project
    exit 1
}
puts "Setting top module: $top_module"
set_property top $top_module [current_fileset]
update_compile_order -fileset sources_1

# Clock constraint
set xdc_file "$output_dir/clock_constraint.xdc"
set xdc_fh [open $xdc_file w]
puts $xdc_fh "create_clock -period 5.000 -name ap_clk \[get_ports ap_clk\]"
close $xdc_fh
add_files -fileset constrs_1 $xdc_file

# ============================================================================
# SYNTHESIS (all modes)
# ============================================================================
puts "Running out-of-context synthesis..."
synth_design -top $top_module -part $fpga_part -mode out_of_context \
    -resource_sharing auto -no_timing_driven
write_checkpoint -force "$output_dir/post_synth.dcp"
puts "Synthesis completed"

# ============================================================================
# IMPLEMENTATION (only for saif_impl)
# ============================================================================
if { $power_mode eq "saif_impl" } {
    puts "Running implementation (opt -> place -> route)..."
    opt_design
    puts "  opt_design done"
    place_design
    puts "  place_design done"
    route_design
    puts "  route_design done"
    write_checkpoint -force "$output_dir/post_route.dcp"
    puts "Implementation completed"
}

# ============================================================================
# SAIF GENERATION via SIMULATION
# ============================================================================
if { $power_mode ne "vectorless" } {
    puts "Setting up simulation for SAIF generation..."

    # Add testbench to sim fileset only
    add_files -fileset sim_1 $tb_file
    set tb_top [file rootname [file tail $tb_file]]
    set_property top $tb_top [get_filesets sim_1]
    set_property top_lib xil_defaultlib [get_filesets sim_1]
    update_compile_order -fileset sim_1

    set saif_file "$output_dir/switching.saif"

    if { $power_mode eq "saif_synth" } {
        # Post-synthesis functional simulation
        set_property -name {xsim.simulate.runtime} -value {all} \
            -objects [get_filesets sim_1]
        set_property -name {xsim.simulate.saif_scope} -value "$tb_top/uut" \
            -objects [get_filesets sim_1]
        set_property -name {xsim.simulate.saif} -value $saif_file \
            -objects [get_filesets sim_1]

        puts "Launching post-synth functional simulation..."
        launch_simulation -mode post-synthesis -type functional
    } else {
        # Post-implementation timing simulation
        set_property -name {xsim.simulate.runtime} -value {all} \
            -objects [get_filesets sim_1]
        set_property -name {xsim.simulate.saif_scope} -value "$tb_top/uut" \
            -objects [get_filesets sim_1]
        set_property -name {xsim.simulate.saif} -value $saif_file \
            -objects [get_filesets sim_1]

        puts "Launching post-impl timing simulation (slow)..."
        launch_simulation -mode post-implementation -type timing
    }

    close_sim
    puts "SAIF generated: $saif_file"

    # Reload the appropriate checkpoint for power reporting
    close_design
    if { $power_mode eq "saif_synth" } {
        open_checkpoint "$output_dir/post_synth.dcp"
    } else {
        open_checkpoint "$output_dir/post_route.dcp"
    }

    # Feed SAIF into power analysis
    read_saif $saif_file
    puts "SAIF loaded for power analysis"
}

# ============================================================================
# POWER REPORTS
# ============================================================================
report_power                         -file "$output_dir/power_summary.txt"
report_power -format xml             -file "$output_dir/power_summary.xml"
report_power -hierarchical_depth 3   -file "$output_dir/power_hierarchical.txt"

# ============================================================================
# AREA / UTILIZATION REPORTS
# ============================================================================
report_utilization                   -file "$output_dir/utilization.txt"
report_utilization -hierarchical     -file "$output_dir/utilization_hierarchical.txt"
report_utilization -format xml       -file "$output_dir/utilization.xml"

# ============================================================================
# TIMING REPORTS
# ============================================================================
report_timing_summary                -file "$output_dir/timing_summary.txt"

# ============================================================================
# FINAL CHECKPOINT
# ============================================================================
write_checkpoint -force "$output_dir/design_checkpoint.dcp"

puts "========================================="
puts "Power & Area estimation complete ($power_mode)"
puts "========================================="
puts "Output directory: $output_dir"
puts "Key reports:"
puts "  Power:  $output_dir/power_summary.txt"
puts "  Area:   $output_dir/utilization.txt"
puts "  Timing: $output_dir/timing_summary.txt"
if { $power_mode ne "vectorless" } {
    puts "  SAIF:   $output_dir/switching.saif"
}
puts "========================================="

close_project
exit 0