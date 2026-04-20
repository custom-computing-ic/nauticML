#!/usr/bin/tclsh
# Power Estimation TCL Script for hls4ml IP Blocks

if { $argc < 3 || $argc > 4 } {
    puts "Error: Invalid number of arguments"
    puts "Usage: vivado -mode batch -source power_estimation_ip.tcl -tclargs <verilog_dir> <output_dir> <fpga_part> \[top_module\]"
    exit 1
}

set verilog_dir [lindex $argv 0]
set output_dir  [lindex $argv 1]
set fpga_part   [lindex $argv 2]

set verilog_dir [file normalize $verilog_dir]
set output_dir  [file normalize $output_dir]

set top_module ""
if { $argc == 4 } {
    set top_module [lindex $argv 3]
}

file mkdir $output_dir

puts "========================================="
puts "HLS4ML IP Power Estimation Script"
puts "========================================="
puts "Verilog Directory: $verilog_dir"
puts "Output Directory: $output_dir"
puts "Target Device: xcu250-figd2104-2l-e"
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

set project_name "power_est_[clock seconds]"
create_project $project_name "$output_dir/$project_name" -part $fpga_part -force

puts "Adding design files..."
set v_files [glob -nocomplain "$verilog_dir/*.v"]
set sv_files [glob -nocomplain "$verilog_dir/*.sv"]
set vh_files [glob -nocomplain "$verilog_dir/*.vh"]

if { [llength $v_files] == 0 && [llength $sv_files] == 0 } {
    puts "Error: No Verilog files found in $verilog_dir"
    close_project
    exit 1
}

if { [llength $v_files] > 0 } {
    add_files $v_files
    puts "Added [llength $v_files] Verilog files"
}
if { [llength $sv_files] > 0 } {
    add_files $sv_files
    puts "Added [llength $sv_files] SystemVerilog files"
}
if { [llength $vh_files] > 0 } {
    add_files -fileset [current_fileset] $vh_files
    puts "Added [llength $vh_files] Verilog header files"
}

if { $top_module == "" } {
    set top_module [find_top_module $verilog_dir]
}

if { $top_module != "" } {
    puts "Setting top module: $top_module"
    set_property top $top_module [current_fileset]
} else {
    puts "Error: Could not determine top module"
    close_project
    exit 1
}

update_compile_order -fileset sources_1

# Clock constraint
set xdc_file "$output_dir/clock_constraint.xdc"
set xdc_fh [open $xdc_file w]
puts $xdc_fh "create_clock -period 5.000 -name ap_clk \[get_ports ap_clk\]"
close $xdc_fh
add_files -fileset constrs_1 $xdc_file

# Run synthesis in-process (no child processes — stable over NFS/Docker)
puts "Running out-of-context synthesis..."
synth_design -top $top_module -part $fpga_part -mode out_of_context -resource_sharing auto -no_timing_driven
puts "Synthesis completed successfully"

puts "Running vectorless power estimation..."

# ============================================================================
# POWER REPORTS
# ============================================================================
report_power -file "$output_dir/power_summary.txt"
report_power -format xml -file "$output_dir/power_summary.xml"
report_power -hierarchical_depth 3 -file "$output_dir/power_hierarchical.txt"

# ============================================================================
# AREA/UTILIZATION REPORTS
# ============================================================================
report_utilization -file "$output_dir/utilization.txt"
report_utilization -hierarchical -file "$output_dir/utilization_hierarchical.txt"
report_utilization -format xml -file "$output_dir/utilization.xml"

# ============================================================================
# TIMING REPORTS
# ============================================================================
report_timing_summary -file "$output_dir/timing_summary.txt"

# ============================================================================
# CHECKPOINT
# ============================================================================
write_checkpoint -force "$output_dir/design_checkpoint.dcp"

puts "========================================="
puts "Power & Area estimation complete!"
puts "========================================="
puts "Output directory: $output_dir"
puts ""
puts "Key reports:"
puts "  Power:  $output_dir/power_summary.txt"
puts "  Area:   $output_dir/utilization.txt"
puts "  Timing: $output_dir/timing_summary.txt"
puts "========================================="

close_project
exit 0