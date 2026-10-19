`timescale 1ns/1ps
module tb_slot_gate;
    parameter integer SLOT_CYCLES = 10;
    parameter integer NUM_SLOTS = 4;
    parameter integer OWN_SLOT = 1;
    parameter integer GUARD_CYCLES = 2;
    reg clk = 0;
    reg rst = 1;
    wire enable;
    integer phase;
    integer index;
    integer open_count;
    integer epoch;
    reg expected;
    slot_gate #(.SLOT_CYCLES(SLOT_CYCLES), .NUM_SLOTS(NUM_SLOTS),
                .OWN_SLOT(OWN_SLOT), .GUARD_CYCLES(GUARD_CYCLES)) dut(clk,rst,enable);
    always #5 clk = ~clk;

    task check_phase;
        input integer p;
        begin
            // Independent absolute phase oracle: open interval within each frame.
            expected = p >= OWN_SLOT*SLOT_CYCLES+GUARD_CYCLES &&
                       p < (OWN_SLOT+1)*SLOT_CYCLES-GUARD_CYCLES;
            if (enable !== expected)
                $fatal(1, "phase mismatch phase=%0d actual=%b expected=%b",p,enable,expected);
        end
    endtask

    initial begin
        // Two epochs, with reset asserted at a different point in the frame.
        for (epoch=0; epoch<2; epoch=epoch+1) begin
            @(negedge clk); rst=1;
            @(posedge clk); #1;
            if (enable !== 0) $fatal(1,"enabled during reset");
            @(negedge clk); rst=0; #1;
            check_phase(0);
            open_count=enable;
            for (index=1; index<SLOT_CYCLES*NUM_SLOTS*3; index=index+1) begin
                @(posedge clk); #1;
                phase=index % (SLOT_CYCLES*NUM_SLOTS);
                check_phase(phase);
                open_count=open_count+enable;
            end
            if (open_count != 3*(SLOT_CYCLES-2*GUARD_CYCLES))
                $fatal(1,"wrong active interval count");
            // Advance by a non-frame interval before resetting again.
            repeat(OWN_SLOT*SLOT_CYCLES+GUARD_CYCLES+1) @(posedge clk);
        end
        $display("PASS: slot phase, guard windows, wrap and reset");
        $finish;
    end
    initial begin
        #1000000;
        $fatal(1,"testbench timeout");
    end
endmodule
