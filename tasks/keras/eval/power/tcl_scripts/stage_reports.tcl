set_param general.maxThreads 4

set output_dir [lindex $argv 0]

open_checkpoint "$output_dir/post_synth.dcp"
report_utilization               -file "$output_dir/utilization.txt"
report_utilization -format xml   -file "$output_dir/utilization.xml"
report_utilization -hierarchical -file "$output_dir/utilization_hierarchical.txt"
report_timing_summary            -file "$output_dir/timing_summary.txt"
write_checkpoint -force "$output_dir/design_final.dcp"

close_design
exit 0