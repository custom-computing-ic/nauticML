# Stage 1b: open the synth checkpoint and write the funcsim netlist.
#
# Separated from stage_synth.tcl so that:
#   (a) post_synth.dcp is on disk before this potentially-expensive step
#   (b) if write_verilog hangs or crashes we don't lose the synth result
#   (c) we can re-run funcsim independently (e.g. on a faster machine)

set_param general.maxThreads 8
catch { config_webtalk -user off }

if { $argc != 1 } {
    puts "Usage: vivado -mode batch -source stage_funcsim.tcl -tclargs <output_dir>"
    exit 1
}

set output_dir [lindex $argv 0]
set dcp        "$output_dir/post_synth.dcp"
set out_v      "$output_dir/post_synth_funcsim.v"

if {![file exists $dcp]} {
    puts "ERROR: checkpoint not found: $dcp"
    exit 1
}

puts "stage_funcsim: opening $dcp"
open_checkpoint $dcp

puts "stage_funcsim: writing $out_v"
write_verilog -mode funcsim -force -nolib $out_v

if {![file exists $out_v]} {
    puts "ERROR: funcsim netlist not produced at $out_v"
    exit 1
}

set sz [expr {[file size $out_v] / 1024}]
puts "stage_funcsim: complete (${sz} KB)"
exit 0
