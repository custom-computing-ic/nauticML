// Power-estimation testbench for hls4ml io_stream DUT.
//
// Stream protocol: TDATA / TVALID / TREADY handshake. Per Vitis HLS UG1399,
// a beat transfers on the rising clock edge when both TVALID and TREADY high.

`timescale 1ns / 1ps

module power_tb;
    parameter CLK_PERIOD       = {clk_period};
    parameter IN_TDATA_WIDTH   = {in_tdata_width};
    parameter OUT_TDATA_WIDTH  = {out_tdata_width};
    parameter N_SAMPLES        = {n_samples};
    parameter BEATS_PER_SAMPLE = {beats_per_sample};
    parameter TOTAL_IN_BEATS   = N_SAMPLES * BEATS_PER_SAMPLE;
    parameter RESET_CYCLES     = 60;
    parameter TAIL_CYCLES      = 200;
    parameter TIMEOUT_CYCLES   = {timeout_cycles};

    reg  ap_clk     = 1'b0;
    reg  {reset_port} = {reset_active};
    reg  ap_start   = 1'b0;
    wire ap_done, ap_idle, ap_ready;
    wire reset_done = ({reset_port} == {reset_inactive});

    reg  [IN_TDATA_WIDTH-1:0] in_tdata  = {{IN_TDATA_WIDTH{{1'b0}}}};
    reg                       in_tvalid = 1'b0;
    wire                      in_tready;

    wire [OUT_TDATA_WIDTH-1:0] out_tdata;
    wire                       out_tvalid;
    reg                        out_tready = 1'b1;

    {dut_name} dut (
        .ap_clk   (ap_clk),
        .{reset_port}(   {reset_port}),
        .ap_start (ap_start),
        .ap_done  (ap_done),
        .ap_idle  (ap_idle),
        .ap_ready (ap_ready),{ap_continue_binding}
        .{in_port}_TDATA (in_tdata),
        .{in_port}_TVALID(in_tvalid),
        .{in_port}_TREADY(in_tready),
        .{out_port}_TDATA (out_tdata),
        .{out_port}_TVALID(out_tvalid),
        .{out_port}_TREADY(out_tready)
    );

    always #(CLK_PERIOD / 2.0) ap_clk = ~ap_clk;

    reg [IN_TDATA_WIDTH-1:0] vectors [0:TOTAL_IN_BEATS-1];
    integer fed         = 0;
    integer captured    = 0;
    integer skipped_x   = 0;
    integer fd;
    integer cycle_count = 0;
    reg     run         = 1'b0;
    reg     done_flag   = 1'b0;
    reg     first_valid_seen = 1'b0;

    // ----------------------------------------------------------------
    // Reset + memory load. No data-path assignments here.
    // ----------------------------------------------------------------
    initial begin
        $readmemh("input_vectors.dat", vectors);
        fd = $fopen("output_captured.dat", "w");
        if (fd == 0) begin
            $display("ERROR: cannot open output_captured.dat");
            $finish;
        end

        repeat (RESET_CYCLES) @(posedge ap_clk);
        @(negedge ap_clk);
        {reset_port} <= {reset_inactive};
        run          <= 1'b1;
    end

    // ----------------------------------------------------------------
    // Stimulus: AXI-Stream input feed, saturating throughput.
    // - Wraps around the vector array once exhausted, so long runs stay
    //   driven with real data instead of falling off the end into X's.
    // - Hold the current beat unchanged when vld is high but rdy is low
    //   (AXI-Stream backpressure rule).
    // ----------------------------------------------------------------
    always @(posedge ap_clk) begin
        if (!reset_done) begin
            fed       <= 0;
            in_tdata  <= {{IN_TDATA_WIDTH{{1'b0}}}};
            in_tvalid <= 1'b0;
            ap_start  <= 1'b0;
        end else if (run) begin
            ap_start <= 1'b1;

            if ((in_tvalid && in_tready) || !in_tvalid) begin
                // Advance: either the previous beat was just accepted, or
                // we haven't started presenting yet.
                in_tdata  <= vectors[fed % TOTAL_IN_BEATS];
                in_tvalid <= 1'b1;
                fed       <= fed + 1;
            end
            // else: tvalid high, tready low — hold the current beat.
        end
    end

    // ----------------------------------------------------------------
    // Capture + timeout.
    // - Skip X-beats silently (DUT pipeline flush during startup).
    // - Don't start counting "captured" until the first non-X output is
    //   seen, then collect N_SAMPLES of real data.
    // ----------------------------------------------------------------
    always @(posedge ap_clk) begin
        if (!reset_done) begin
            captured         <= 0;
            cycle_count      <= 0;
            done_flag        <= 1'b0;
            skipped_x        <= 0;
            first_valid_seen <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (cycle_count > TIMEOUT_CYCLES) begin
                $display("ERROR: timeout after %0d cycles, fed %0d, captured %0d/%0d, skipped_x %0d, first_valid_seen=%0b",
                         cycle_count, fed, captured, N_SAMPLES, skipped_x, first_valid_seen);
                $fclose(fd);
                $finish;
            end

            if (out_tvalid && out_tready) begin
                if ((^out_tdata) === 1'bx) begin
                    skipped_x <= skipped_x + 1;
                end else begin
                    first_valid_seen <= 1'b1;
                    $fwrite(fd, "%h\n", out_tdata);
                    captured <= captured + 1;
                    if (captured + 1 >= N_SAMPLES) begin
                        done_flag <= 1'b1;
                    end
                end
            end
        end
    end

    // ----------------------------------------------------------------
    // Tail + finish, fully decoupled.
    // ----------------------------------------------------------------
    initial begin
        wait (done_flag === 1'b1);
        repeat (TAIL_CYCLES) @(posedge ap_clk);
        $fclose(fd);
        $display("SUCCESS: captured %0d samples, skipped %0d X-beats",
                 N_SAMPLES, skipped_x);
        $finish;
    end

    // Progress heartbeat — prints every 5000 cycles so a slow sim shows
    // life in xsim.log instead of looking hung.
    always @(posedge ap_clk) begin
        if (reset_done && (cycle_count % 5000 == 0)) begin
            $display("[%0t] cycle=%0d fed=%0d cap=%0d skipped_x=%0d in_tvalid=%b in_tready=%b out_tvalid=%b",
                     $time, cycle_count, fed, captured, skipped_x, in_tvalid, in_tready, out_tvalid);
        end
    end
endmodule