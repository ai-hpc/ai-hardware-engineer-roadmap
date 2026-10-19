// Local single-clock scheduling primitive, not a synchronized wireless MAC.
module slot_gate #(
    parameter integer SLOT_CYCLES = 10,
    parameter integer NUM_SLOTS = 4,
    parameter integer OWN_SLOT = 1,
    parameter integer GUARD_CYCLES = 2
)(
    input wire clk,
    input wire rst,
    output wire tx_enable
);
    localparam integer TICK_BITS = SLOT_CYCLES > 1 ? $clog2(SLOT_CYCLES) : 1;
    localparam integer SLOT_BITS = NUM_SLOTS > 1 ? $clog2(NUM_SLOTS) : 1;
    reg [TICK_BITS-1:0] tick_q;
    reg [SLOT_BITS-1:0] slot_q;

    initial begin
        if (SLOT_CYCLES <= 0 || NUM_SLOTS <= 0 || OWN_SLOT < 0 ||
            OWN_SLOT >= NUM_SLOTS || GUARD_CYCLES < 0 ||
            GUARD_CYCLES >= SLOT_CYCLES || GUARD_CYCLES >= SLOT_CYCLES-GUARD_CYCLES)
            $fatal(1, "invalid slot configuration");
    end

    assign tx_enable = !rst && (slot_q == OWN_SLOT) &&
                       (tick_q >= GUARD_CYCLES) &&
                       (tick_q < SLOT_CYCLES-GUARD_CYCLES);

    // Synchronous reset; output is also suppressed while rst is asserted.
    always @(posedge clk) begin
        if (rst) begin
            tick_q <= 0;
            slot_q <= 0;
        end else begin
`ifdef COURSE_MUTATE_LONG_SLOT
            if (tick_q == SLOT_CYCLES) begin
`else
            if (tick_q == SLOT_CYCLES-1) begin
`endif
                tick_q <= 0;
                if (slot_q == NUM_SLOTS-1) slot_q <= 0;
                else slot_q <= slot_q + 1'b1;
            end else tick_q <= tick_q + 1'b1;
        end
    end
endmodule
