# Stage 1: synthesis only, then exit. Fresh Vivado process.

set verilog_dir  [lindex $argv 0]
set output_dir   [lindex $argv 1]
set fpga_part    [lindex $argv 2]
set top_module   [lindex $argv 3]
set clock_period [lindex $argv 4]

foreach vfile [glob -nocomplain "$verilog_dir/*.v"] {
    read_verilog $vfile
}
foreach sfile [glob -nocomplain "$verilog_dir/*.sv"] {
    read_verilog -sv $sfile
}

set xdc_file "$output_dir/clock.xdc"
set xdc_fh [open $xdc_file w]
puts $xdc_fh "create_clock -period $clock_period -name ap_clk \[get_ports ap_clk\]"
close $xdc_fh
read_xdc $xdc_file

synth_design -top $top_module -part $fpga_part -mode out_of_context -flatten_hierarchy none
write_checkpoint -force "$output_dir/post_synth.dcp"
write_verilog -mode funcsim -force "$output_dir/post_synth_funcsim.v"
exit 0