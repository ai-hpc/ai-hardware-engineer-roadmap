`timescale 1ns/1ps
`default_nettype none

// One-slot elastic stage: unsigned byte -> saturating byte + 1.
// Both interfaces share clk. rst is active high and must span a rising edge.
module stream_increment (
    input  wire       clk,
    input  wire       rst,
    input  wire       s_valid,
    output wire       s_ready,
    input  wire [7:0] s_data,
    output wire       m_valid,
    input  wire       m_ready,
    output wire [7:0] m_data,
    output reg        overflow_seen
);
    reg       full;
    reg [7:0] payload;

    // A departing item can be replaced at the same rising edge.
    // Reset suppresses interface transfers and flushes storage at the edge.
    assign s_ready = !rst && (!full || m_ready);
    assign m_valid = !rst && full;
    assign m_data = payload;

    always @(posedge clk) begin
        if (rst) begin
            full <= 1'b0;
            payload <= 8'd0;
            overflow_seen <= 1'b0;
        end else if (s_ready) begin
            full <= s_valid;
            if (s_valid) begin
                payload <= (s_data == 8'hff) ? 8'hff : s_data + 8'd1;
                if (s_data == 8'hff)
                    overflow_seen <= 1'b1;
            end
        end
        // When full && !m_ready, preserve valid, payload, and overflow flag.
    end
endmodule

`default_nettype wire
