`timescale 1ns/1ps
`default_nettype none

module tb_stream_increment;
    reg clk = 0;
    always #5 clk = !clk;  // Simulation time only; this is not a timing result.
    reg rst = 1, s_valid = 0, m_ready = 0;
    reg [7:0] s_data = 0;
    wire s_ready, m_valid, overflow_seen;
    wire [7:0] m_data;
    stream_increment dut(.*);

    integer accepted = 0, delivered = 0, flushed = 0;
    integer stalls = 0, replacements = 0, saturated = 0;
    integer occupancy = 0, cycles = 0, j, seed = 1;
    integer expected = 0;  // Independent mathematical result for one slot.
    reg sticky_expected = 0, held = 0, source_held = 0;
    reg [7:0] held_data = 0, source_held_data = 0;
    reg input_taken = 0;
    reg [31:0] rng = 1;
    reg next_valid = 0;
    reg [7:0] next_data = 0;
    reg [1023:0] waveform;

    // Sample BEFORE the DUT's nonblocking updates. Inputs change at negedge.
    // This checks accepted transactions, not merely intended stimulus.
    always @(posedge clk) begin
        cycles = cycles + 1;
        input_taken = 0;
        if (rst) begin
            if (s_ready !== 1'b0 || m_valid !== 1'b0)
                $fatal(1, "RESET: transfers must be suppressed");
            flushed = flushed + occupancy;
            occupancy = 0;
            held = 0;
            source_held = 0;
            sticky_expected = 0;
        end else begin
            if (held && (m_valid !== 1'b1 || m_data !== held_data))
                $fatal(1, "STALL: output changed before acceptance");
            if (overflow_seen !== sticky_expected)
                $fatal(1, "OVERFLOW: sticky flag differs from accepted-input reference");
            if (source_held && (s_valid !== 1'b1 || s_data !== source_held_data))
                $fatal(1, "SOURCE: testbench violated its own valid/ready contract");
            if (m_valid !== (occupancy == 1))
                $fatal(1, "VALID: output valid disagrees with reference occupancy");
            if (s_ready !== ((occupancy == 0) || m_ready))
                $fatal(1, "READY: stage cannot accept exactly when expected");
            if (m_valid && m_data !== expected[7:0])
                $fatal(1, "DATA: expected %0d got %0d", expected, m_data);

            if (m_valid && m_ready) begin
                delivered = delivered + 1;
                occupancy = occupancy - 1;
                if (s_valid && s_ready) replacements = replacements + 1;
            end
            if (s_valid && s_ready) begin
                accepted = accepted + 1;
                input_taken = 1;
                occupancy = occupancy + 1;
                // Widen first, then clamp: independent of the RTL expression.
                expected = $unsigned(s_data) + 1;
                if (expected > 255) begin
                    expected = 255;
                    sticky_expected = 1;
                    saturated = saturated + 1;
                end
            end
            if (occupancy < 0 || occupancy > 1)
                $fatal(1, "CAPACITY: reference occupancy is outside [0,1]");
            held = m_valid && !m_ready;
            held_data = m_data;
            source_held = s_valid && !s_ready;
            source_held_data = s_data;
            if (held) stalls = stalls + 1;
        end
    end

    task tick(input reg reset_now, input reg valid_now,
              input reg [7:0] data_now, input reg ready_now);
        begin
            @(negedge clk);
            rst = reset_now;
            s_valid = valid_now;
            s_data = data_now;
            m_ready = ready_now;
            @(posedge clk);
            #1; // Scoreboard and nonblocking assignments have completed.
        end
    endtask

    initial begin
        if ($value$plusargs("SEED=%d", seed)) begin end
        rng = seed;
        if ($value$plusargs("VCD=%s", waveform)) begin
            $dumpfile(waveform);
            $dumpvars(0, tb_stream_increment);
        end
        tick(1, 0, 0, 0);
        tick(0, 1, 10, 0);     // Empty stage accepts even with sink stalled.
        tick(0, 1, 255, 0);    // Hold old output AND pending new input.
        tick(0, 1, 255, 0);
        tick(0, 1, 255, 1);    // Simultaneously deliver 11 and accept 255.
        tick(0, 0, 0, 1);      // Saturation must produce 255, not wrap to 0.
        tick(0, 1, 44, 0);     // Queue one item that reset will discard.
        tick(1, 0, 0, 0);
        tick(1, 0, 0, 0);
        tick(0, 0, 0, 1);      // No old result may appear after reset.

        for (j = 0; j < 1000; j = j + 1) begin
            rng = rng ^ (rng << 13);
            rng = rng ^ (rng >> 17);
            rng = rng ^ (rng << 5);
            // A producer must retain a pending item through backpressure.
            if (!next_valid || input_taken) begin
                next_valid = rng[0] | rng[1];
                next_data = rng[15:8];
            end
            tick(0, next_valid, next_data, rng[2]);
        end
        // Finish any pending source transfer before withdrawing valid.
        if (next_valid && !input_taken) tick(0, 1, next_data, 1);
        tick(0, 0, 0, 1);
        tick(0, 0, 0, 1);
        if (occupancy != 0 || accepted != delivered + flushed)
            $fatal(1, "ACCOUNTING: accepted != delivered + reset-flushed");
        if (stalls == 0 || replacements == 0 || saturated == 0 || flushed != 1)
            $fatal(1, "COVERAGE: required scenarios did not occur");
        $display("RESULT accepted=%0d delivered=%0d flushed=%0d stalls=%0d replacements=%0d saturated=%0d cycles=%0d",
                 accepted, delivered, flushed, stalls, replacements, saturated, cycles);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "TIMEOUT: simulation did not finish");
    end
endmodule

`default_nettype wire
