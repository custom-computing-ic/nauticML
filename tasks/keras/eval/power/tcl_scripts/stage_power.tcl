# Single power-report run. args:
#   <output_dir> <dcp_filename> <saif_path_or_empty> <strip_path_or_empty> <out_basename>
set output_dir   [lindex $argv 0]
set dcp_name     [lindex $argv 1]
set saif_path    [lindex $argv 2]
set strip_path   [lindex $argv 3]
set out_basename [lindex $argv 4]

open_checkpoint "$output_dir/$dcp_name"

if {$saif_path ne "" && [file exists $saif_path]} {
    read_saif -strip_path $strip_path $saif_path
}

report_power -format xml  -file "$output_dir/${out_basename}.xml"
report_power -verbose     -file "$output_dir/${out_basename}_verbose.txt"
report_power -hierarchical_depth 100 -xpe power.xpe -file  "$output_dir/${out_basename}_hierarchical.txt"

if {$saif_path ne ""} {
    set tag [string map {power_ ""} $out_basename]
    report_switching_activity -hier -file "$output_dir/switching_${tag}_breakdown.txt" [get_cells]
}

close_design
exit 0