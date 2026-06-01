# Stage 1: synthesis only, then exit. Fresh Vivado process.
#
#   - maxThreads 4 — post-synth optimization & Unisim transformation phases
#     can use multiple threads; serializing them makes the slow box appear
#     hung.
#   - -flatten_hierarchy none (was rebuilt) — rebuilt forces a flatten-then-
#     rebuild pass that is very expensive on conv designs with many DSPs.
#     For OOC synth into a checkpoint we don't need rebuilt.
#   - -directive RuntimeOptimized — skip QoR passes we don't need for
#     power estimation.
#   - write_verilog -mode funcsim is in stage_funcsim.tcl so the dcp is
#     saved BEFORE the expensive netlist export. If funcsim write hangs
#     or OOMs, we still have post_synth.dcp on disk.

set_param general.maxThreads 4
catch { config_webtalk -user off }

if { $argc != 5 } {
    puts "Usage: vivado -mode batch -source stage_synth.tcl -tclargs \\"
    puts "         <verilog_dir> <output_dir> <fpga_part> <top_module> <clock_period_ns>"
    exit 1
}

set verilog_dir  [lindex $argv 0]
set output_dir   [lindex $argv 1]
set fpga_part    [lindex $argv 2]
set top_module   [lindex $argv 3]
set clock_period [lindex $argv 4]

puts "stage_synth: verilog_dir=$verilog_dir"
puts "stage_synth: output_dir=$output_dir"
puts "stage_synth: part=$fpga_part top=$top_module period=$clock_period"

set v_files  [glob -nocomplain "$verilog_dir/*.v"]
set sv_files [glob -nocomplain "$verilog_dir/*.sv"]

if {[llength $v_files] == 0 && [llength $sv_files] == 0} {
    puts "ERROR: no .v or .sv files found in $verilog_dir"
    exit 1
}

foreach vfile $v_files {
    read_verilog $vfile
}
foreach sfile $sv_files {
    read_verilog -sv $sfile
}

set xdc_file "$output_dir/clock.xdc"
set xdc_fh [open $xdc_file w]
puts $xdc_fh "create_clock -period $clock_period -name ap_clk \[get_ports ap_clk\]"
close $xdc_fh
read_xdc $xdc_file

synth_design -top $top_module -part $fpga_part \
    -mode out_of_context \
    -flatten_hierarchy none \
    -directive RuntimeOptimized

# Save the checkpoint FIRST. If anything downstream hangs/crashes,
# we still have this and stage_funcsim can be retried independently.
write_checkpoint -force "$output_dir/post_synth.dcp"

puts "stage_synth: complete"
exit 0
