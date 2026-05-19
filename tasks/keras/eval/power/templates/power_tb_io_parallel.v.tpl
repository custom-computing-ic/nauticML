// Power-estimation testbench for hls4ml io_parallel DUT.
//
// Reset duration is RESET_CYCLES (60 @ 5ns = 300ns) to clear Vivado's GSR
// pulse, which holds all FFs in reset for ~100ns in post-synth/post-impl
// sims. See UG900 "Post-Synthesis Simulation".
//
// Important: reset port name and polarity come from PortParser, not
// hardcoded — Vitis HLS emits `ap_rst_n` (active-low) on the top function;
// older Vivado HLS used `ap_rst` (active-high). A hardcoded `ap_rst`
// silently no-connects against `ap_rst_n` (Vivado treats unmatched named
// connections as a warning), leaves the real reset floating at Z, and the
// FSM never properly releases — ap_idle/ap_done/ap_ready never fire even
// though the internal dataflow still clocks (producing X capture pulses).
//
// ap_continue is bound to 1'b1 whenever the DUT exposes it (ap_ctrl_chain),
// so each ap_done is acknowledged immediately and the FSM is free to start
// the next call. With ap_ctrl_hs the binding line is empty.
//
// Stimulus: one sample per DUT call, driven by the ap_idle → ap_done
// handshake. Earlier revisions held ap_start + input_vld high continuously
// and advanced `fed` every clock; with II ≈ ReuseFactor (often hundreds of
// cycles per call) that meant inputs cycled at ~512× the DUT's actual
// sampling rate, so captured[k] aligned to vectors[k·II mod N_SAMPLES]
// rather than vectors[k] — only ~N_SAMPLES/II distinct inputs ever reached
// the DUT and the argmax-vs-reference comparison flipped on ~half the
// samples. The handshake-gated pattern below feeds exactly one new input
// per ap_done so captured[k] == output(vectors[k]).

`timescale 1ns / 1ps

module power_tb;
    parameter CLK_PERIOD      = {clk_period};
    parameter IN_WIDTH_TOTAL  = {in_width_total};
    parameter OUT_WIDTH_TOTAL = {out_width_total};
    parameter N_SAMPLES       = {n_samples};
    parameter RESET_CYCLES    = 60;
    parameter TAIL_CYCLES     = 200;
    parameter TIMEOUT_CYCLES  = {timeout_cycles};

    reg  ap_clk         = 1'b0;
    reg  {reset_port}   = {reset_active};
    reg  ap_start       = 1'b0;
    wire ap_done, ap_idle, ap_ready;
    wire reset_done     = ({reset_port} == {reset_inactive});

    reg  [IN_WIDTH_TOTAL-1:0] input_bus = {{IN_WIDTH_TOTAL{{1'b0}}}};
    reg                       input_vld = 1'b0;

{output_wire_decls}

    wire capture_vld = {capture_vld_expr};

    {dut_name} dut (
        .ap_clk        (ap_clk),
        .{reset_port}  ({reset_port}),
        .ap_start      (ap_start),
        .ap_done       (ap_done),
        .ap_idle       (ap_idle),
        .ap_ready      (ap_ready),{ap_continue_binding}
        .{input_port_name}       (input_bus),
        .{input_port_name}_ap_vld(input_vld){output_port_connections}
    );

    wire [OUT_WIDTH_TOTAL-1:0] output_bus = {{{output_concat}}};

    always #(CLK_PERIOD / 2.0) ap_clk = ~ap_clk;

    reg [IN_WIDTH_TOTAL-1:0] vectors [0:N_SAMPLES-1];
    integer fed              = 0;
    integer captured         = 0;
    integer skipped_x        = 0;
    integer fd;
    integer cycle_count      = 0;
    reg     run              = 1'b0;
    reg     done_flag        = 1'b0;
    reg     first_valid_seen = 1'b0;

    // ----------------------------------------------------------------
    // Reset + memory load. Single driver for `{reset_port}` and `run`.
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
    // Stimulus: handshake-gated, one sample per call. Two transitions:
    //   1. DUT idle and we haven't started yet → present vectors[fed],
    //      raise ap_start + input_vld. DUT will sample in[] on the next
    //      clock and enter its run state.
    //   2. ap_done pulses → drop ap_start + input_vld, advance fed. DUT
    //      returns to idle, and condition (1) fires again the next cycle.
    // Total overhead is ~2 cycles per call vs the back-to-back chained
    // pattern, negligible compared to II=ReuseFactor cycles of work.
    // ----------------------------------------------------------------
    always @(posedge ap_clk) begin
        if (!reset_done) begin
            fed       <= 0;
            input_bus <= {{IN_WIDTH_TOTAL{{1'b0}}}};
            input_vld <= 1'b0;
            ap_start  <= 1'b0;
        end else if (run && !done_flag) begin
            if (ap_idle && !ap_start) begin
                input_bus <= vectors[fed % N_SAMPLES];
                input_vld <= 1'b1;
                ap_start  <= 1'b1;
            end else if (ap_done) begin
                ap_start  <= 1'b0;
                input_vld <= 1'b0;
                fed       <= fed + 1;
            end
        end else if (done_flag) begin
            ap_start  <= 1'b0;
            input_vld <= 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // Capture + timeout. Skip X-beats during pipeline fill (matches the
    // io_stream TB) so output_captured.dat is clean and the SAIF run is
    // pinned on real switching activity, not Z/X glitches.
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

            if (capture_vld) begin
                if ((^output_bus) === 1'bx) begin
                    skipped_x <= skipped_x + 1;
                end else begin
                    first_valid_seen <= 1'b1;
                    $fwrite(fd, "%h\n", output_bus);
                    captured <= captured + 1;
                    if (captured + 1 >= N_SAMPLES) begin
                        done_flag <= 1'b1;
                    end
                end
            end
        end
    end

    // ----------------------------------------------------------------
    // Tail + finish, fully decoupled from the capture block so the
    // capture process never blocks on `repeat`.
    // ----------------------------------------------------------------
    initial begin
        wait (done_flag === 1'b1);
        repeat (TAIL_CYCLES) @(posedge ap_clk);
        $fclose(fd);
        $display("SUCCESS: captured %0d samples, skipped %0d X-beats",
                 N_SAMPLES, skipped_x);
        $finish;
    end

    // Periodic progress logging — leave commented out for SAIF runs.
    // Uncomment to debug a hang (handshake stuck, capture_vld silent, etc.).
    // always @(posedge ap_clk) begin
    //     if (cycle_count % 5000 == 0) begin
    //         $display("[%0t] cycle=%0d reset_done=%b run=%b done=%b start=%b ready=%b idle=%b done_dut=%b vld=%b fed=%0d cap=%0d skipped_x=%0d",
    //                  $time, cycle_count, reset_done, run, done_flag, ap_start, ap_ready, ap_idle, ap_done, input_vld, fed, captured, skipped_x);
    //     end
    // end

endmodule
