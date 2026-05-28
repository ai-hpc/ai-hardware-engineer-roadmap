# Hardware Description Languages — Verilog Focus

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">HDLV</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Specialized course identity for Hardware Description Languages — Verilog Focus.</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


Verilog is how you turn the digital design concepts from the previous guide into real hardware. You write behavioral or structural descriptions; synthesis tools convert them into gates; place-and-route tools turn those gates into physical layout. This guide teaches Verilog through progressively larger designs — from wires and gates to a pipelined MIPS core — with testbenches at every step.

> **Why Verilog over VHDL?** Both are IEEE standards and fully capable. Verilog dominates in US industry and all major AI chip companies (NVIDIA, AMD, Google, Apple). VHDL is stronger in European defense/aerospace. SystemVerilog extends Verilog with verification features (UVM, assertions) and is covered in Phase 2. Start with Verilog — you can read VHDL in a weekend once you know one HDL.

---

## 1. Verilog Fundamentals

### 1.1 Modules and Ports

A **module** is the basic building block — it has a name, ports, and internal logic. Think of it as a chip with labeled pins.

```verilog
module adder (
    input  wire [3:0] a,      // 4-bit input
    input  wire [3:0] b,      // 4-bit input
    input  wire       cin,    // carry in
    output wire [3:0] sum,    // 4-bit sum
    output wire       cout    // carry out
);
    assign {cout, sum} = a + b + cin;
endmodule
```

**Port directions:** `input`, `output`, `inout` (bidirectional — rare, mostly for pads).

**Port types:** `wire` (default for inputs/outputs) or `reg` (required for outputs driven inside `always` blocks — despite the name, `reg` does NOT always mean a register).

### 1.2 Data Types

| Type        | What it represents                           | Driven by                      |
|-------------|----------------------------------------------|--------------------------------|
| `wire`      | Physical connection (combinational)          | `assign`, module output, gate  |
| `reg`       | Storage element OR combinational variable    | `always`, `initial`            |
| `integer`   | 32-bit signed, for loops/testbenches         | Not synthesizable as hardware  |
| `parameter` | Named constant                               | Set at compile/instantiation   |
| `localparam`| Named constant, cannot be overridden         | Internal to module             |

**Vectors** — multi-bit signals:

```verilog
wire [7:0]  data_bus;        // 8-bit bus, data_bus[7] is MSB
reg  [31:0] instruction;     // 32-bit register
wire [0:7]  reversed;        // bit 0 is MSB (uncommon, avoid)
```

**Bit-select and part-select:**

```verilog
wire [31:0] instr;
wire [5:0]  opcode = instr[31:26];   // bits 31 down to 26
wire [4:0]  rs     = instr[25:21];
wire [4:0]  rt     = instr[20:16];
```

**Memory (array of registers):**

```verilog
reg [7:0] mem [0:255];       // 256 bytes of memory
reg [31:0] reg_file [0:31];  // 32-entry × 32-bit register file
```

### 1.3 Number Literals

Format: `<width>'<radix><value>`

```verilog
8'b1010_0011     // 8-bit binary = 0xA3 (underscores for readability)
8'hA3            // 8-bit hex = 163
8'd163           // 8-bit decimal
32'hDEAD_BEEF    // 32-bit hex
4'b1             // 4-bit = 4'b0001 (zero-padded)
-8'd5            // 8-bit negative = two's complement of 5
```

**Four-value logic:** every bit can be `0`, `1`, `x` (unknown), or `z` (high-impedance/undriven).

```verilog
4'bxx01          // upper 2 bits unknown
8'bzzzz_zzzz     // tri-state (all high-Z)
```

`x` propagation is the #1 source of simulation bugs. If you see `x` in your waveform, something is undriven or uninitialized — track it down.

### 1.4 Operators

| Category     | Operators                             | Notes                          |
|-------------|---------------------------------------|--------------------------------|
| Arithmetic  | `+`, `-`, `*`, `/`, `%`              | `/` and `%` may not synthesize |
| Relational  | `<`, `>`, `<=`, `>=`                 | Result is 1-bit               |
| Equality    | `==`, `!=`                            | `x` in either operand → `x` result |
| Case equal  | `===`, `!==`                         | Compares x/z literally (sim only) |
| Logical     | `&&`, `\|\|`, `!`                    | Result is 1-bit               |
| Bitwise     | `&`, `\|`, `^`, `~`, `~^`           | Bit-by-bit                    |
| Reduction   | `&a`, `\|a`, `^a`, `~&a`, `~\|a`    | Reduce vector to 1 bit        |
| Shift       | `<<`, `>>`, `<<<`, `>>>`            | `>>>` is arithmetic (sign-extend) |
| Concatenation | `{a, b}`, `{4{a}}`                | `{4{1'b0}}` = `4'b0000`      |
| Conditional | `cond ? a : b`                       | Synthesizes to MUX             |

**Reduction operator example** — parity:

```verilog
wire [7:0] data;
wire parity = ^data;    // XOR all 8 bits → 1 if odd number of 1s
```

**Concatenation and replication — sign extension:**

```verilog
wire [15:0] imm16;
wire [31:0] sign_ext = {{16{imm16[15]}}, imm16};  // replicate sign bit 16 times
```

---

## 2. Modeling Styles

Verilog gives you three ways to describe hardware. All three can be mixed in one module.

### 2.1 Dataflow Modeling (assign)

Continuous assignments describe combinational logic — the output updates whenever any input changes.

```verilog
module mux2 #(parameter WIDTH = 8) (
    input  wire [WIDTH-1:0] d0, d1,
    input  wire             sel,
    output wire [WIDTH-1:0] y
);
    assign y = sel ? d1 : d0;
endmodule
```

The `?:` operator maps directly to a 2:1 MUX in hardware. Chained `?:` becomes a priority MUX chain.

**Full adder in dataflow:**

```verilog
module full_adder (
    input  wire a, b, cin,
    output wire sum, cout
);
    assign sum  = a ^ b ^ cin;
    assign cout = (a & b) | (cin & (a ^ b));
endmodule
```

### 2.2 Behavioral Modeling (always blocks)

`always` blocks describe what happens on events — clock edges for sequential, signal changes for combinational.

**Combinational logic — `always @(*)`:**

```verilog
module alu (
    input  wire [31:0] a, b,
    input  wire [2:0]  op,
    output reg  [31:0] result,
    output reg         zero
);
    always @(*) begin
        case (op)
            3'b000: result = a & b;       // AND
            3'b001: result = a | b;       // OR
            3'b010: result = a + b;       // ADD
            3'b110: result = a - b;       // SUB
            3'b111: result = (a < b) ? 32'd1 : 32'd0;  // SLT
            default: result = 32'bx;      // undefined
        endcase
        zero = (result == 32'd0);
    end
endmodule
```

**Key rules for combinational `always @(*)`:**
- Use `always @(*)` — the `*` auto-includes all read signals in the sensitivity list
- Assign EVERY output in EVERY branch (or use `default`) — incomplete assignment creates a latch (unintended memory)
- Use blocking assignment `=` (executes in order, like software)

**Sequential logic — `always @(posedge clk)`:**

```verilog
module register #(parameter WIDTH = 32) (
    input  wire             clk,
    input  wire             rst,
    input  wire             en,
    input  wire [WIDTH-1:0] d,
    output reg  [WIDTH-1:0] q
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            q <= {WIDTH{1'b0}};      // async reset
        else if (en)
            q <= d;                   // load on enable
        // else: q holds (implicit — this is a register)
    end
endmodule
```

**Key rules for sequential `always @(posedge clk)`:**
- Use non-blocking assignment `<=` (all right-hand sides evaluated first, then all left-hand sides updated — models real flip-flop behavior)
- Reset comes first (`if (rst)`) — can be async (`posedge rst` in sensitivity) or sync (no `rst` in sensitivity)
- Omitting an else branch is fine — it means "hold the value" (register)

**Blocking (`=`) vs non-blocking (`<=`) — the critical distinction:**

```verilog
// WRONG — blocking in sequential logic:
always @(posedge clk) begin
    b = a;     // b gets a immediately
    c = b;     // c gets the NEW b (= a)  → b and c both get a!
end

// CORRECT — non-blocking in sequential logic:
always @(posedge clk) begin
    b <= a;    // scheduled: b will get old a
    c <= b;    // scheduled: c will get old b  → shift register!
end
```

Rule: **use `<=` in clocked blocks, `=` in combinational blocks.** Never mix them.

### 2.3 Structural Modeling (instantiation)

Build hierarchy by connecting modules together:

```verilog
module ripple_carry_4 (
    input  wire [3:0] a, b,
    input  wire       cin,
    output wire [3:0] sum,
    output wire       cout
);
    wire c1, c2, c3;

    full_adder fa0 (.a(a[0]), .b(b[0]), .cin(cin),  .sum(sum[0]), .cout(c1));
    full_adder fa1 (.a(a[1]), .b(b[1]), .cin(c1),   .sum(sum[1]), .cout(c2));
    full_adder fa2 (.a(a[2]), .b(b[2]), .cin(c2),   .sum(sum[2]), .cout(c3));
    full_adder fa3 (.a(a[3]), .b(b[3]), .cin(c3),   .sum(sum[3]), .cout(cout));
endmodule
```

**Always use named port connections** (`.port(signal)`) — positional connections are error-prone and unreadable.

### 2.4 Generate Blocks

`generate` creates hardware at elaboration time — parameterized, scalable structures:

```verilog
module ripple_carry #(parameter N = 32) (
    input  wire [N-1:0] a, b,
    input  wire         cin,
    output wire [N-1:0] sum,
    output wire         cout
);
    wire [N:0] carry;
    assign carry[0] = cin;
    assign cout = carry[N];

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : fa_stage
            full_adder fa (
                .a(a[i]), .b(b[i]), .cin(carry[i]),
                .sum(sum[i]), .cout(carry[i+1])
            );
        end
    endgenerate
endmodule
```

The `generate for` unrolls at compile time — it creates N separate `full_adder` instances, not a loop that runs at runtime.

---

## 3. Combinational Design Patterns

### 3.1 Decoder

```verilog
module decoder_2to4 (
    input  wire [1:0] in,
    input  wire       en,
    output reg  [3:0] out
);
    always @(*) begin
        out = 4'b0000;
        if (en)
            case (in)
                2'b00: out = 4'b0001;
                2'b01: out = 4'b0010;
                2'b10: out = 4'b0100;
                2'b11: out = 4'b1000;
            endcase
    end
endmodule
```

### 3.2 Priority Encoder

```verilog
module priority_enc (
    input  wire [7:0] req,
    output reg  [2:0] grant,
    output reg        valid
);
    always @(*) begin
        valid = 1'b1;
        casez (req)                         // casez treats z/? as don't-care
            8'b1???_????: grant = 3'd7;
            8'b01??_????: grant = 3'd6;
            8'b001?_????: grant = 3'd5;
            8'b0001_????: grant = 3'd4;
            8'b0000_1???: grant = 3'd3;
            8'b0000_01??: grant = 3'd2;
            8'b0000_001?: grant = 3'd1;
            8'b0000_0001: grant = 3'd0;
            default: begin grant = 3'd0; valid = 1'b0; end
        endcase
    end
endmodule
```

### 3.3 Parameterized MUX

```verilog
module mux4 #(parameter W = 32) (
    input  wire [W-1:0] d0, d1, d2, d3,
    input  wire [1:0]   sel,
    output reg  [W-1:0] y
);
    always @(*) begin
        case (sel)
            2'b00: y = d0;
            2'b01: y = d1;
            2'b10: y = d2;
            2'b11: y = d3;
        endcase
    end
endmodule
```

### 3.4 ALU with Flags

A complete ALU as you'd find in a MIPS datapath:

```verilog
module alu_32 (
    input  wire [31:0] a, b,
    input  wire [3:0]  alu_ctrl,
    output reg  [31:0] result,
    output wire        zero,
    output wire        negative,
    output wire        overflow,
    output wire        carry_out
);
    reg [32:0] tmp;    // 33-bit for carry detection

    always @(*) begin
        tmp = 33'd0;
        case (alu_ctrl)
            4'b0000: tmp = {1'b0, a & b};                    // AND
            4'b0001: tmp = {1'b0, a | b};                    // OR
            4'b0010: tmp = {1'b0, a} + {1'b0, b};           // ADD
            4'b0110: tmp = {1'b0, a} + {1'b0, ~b} + 33'd1;  // SUB
            4'b0111: begin                                    // SLT
                tmp = {1'b0, a} + {1'b0, ~b} + 33'd1;
                tmp = {32'd0, tmp[32] ^ overflow};  // signed less than
            end
            4'b1100: tmp = {1'b0, ~(a | b)};                 // NOR
            4'b1000: tmp = {1'b0, a ^ b};                    // XOR
            default: tmp = 33'd0;
        endcase
        result = tmp[31:0];
    end

    assign zero     = (result == 32'd0);
    assign negative = result[31];
    assign carry_out = tmp[32];
    assign overflow = (alu_ctrl == 4'b0010 || alu_ctrl == 4'b0110) ?
                      (a[31] ^ result[31]) & ~(a[31] ^ b[31] ^ alu_ctrl[2]) :
                      1'b0;
endmodule
```

---

## 4. Sequential Design Patterns

### 4.1 D Flip-Flop Variants

```verilog
// Basic DFF with async reset
module dff (
    input  wire clk, rst, d,
    output reg  q
);
    always @(posedge clk or posedge rst)
        if (rst) q <= 1'b0;
        else     q <= d;
endmodule

// DFF with sync reset and enable
module dff_en (
    input  wire clk, rst, en, d,
    output reg  q
);
    always @(posedge clk)
        if (rst)      q <= 1'b0;
        else if (en)  q <= d;
endmodule
```

### 4.2 Shift Register and LFSR

```verilog
module shift_reg #(parameter N = 8) (
    input  wire       clk, rst, si,    // serial in
    output wire       so,               // serial out
    output wire [N-1:0] q               // parallel out
);
    reg [N-1:0] sr;

    always @(posedge clk or posedge rst)
        if (rst) sr <= {N{1'b0}};
        else     sr <= {sr[N-2:0], si};   // shift left, insert si at LSB

    assign so = sr[N-1];
    assign q  = sr;
endmodule
```

**LFSR (maximal-length, 8-bit):**

```verilog
module lfsr8 (
    input  wire       clk, rst,
    output reg  [7:0] q
);
    // Polynomial: x^8 + x^6 + x^5 + x^4 + 1 (taps at 8,6,5,4)
    wire feedback = q[7] ^ q[5] ^ q[4] ^ q[3];

    always @(posedge clk or posedge rst)
        if (rst) q <= 8'h01;                       // seed (must be non-zero)
        else     q <= {q[6:0], feedback};           // shift + feedback
endmodule
// Period: 2^8 - 1 = 255 states before repeating
```

### 4.3 Counter with Enable and Load

```verilog
module counter #(parameter WIDTH = 8) (
    input  wire              clk, rst, en, load,
    input  wire [WIDTH-1:0]  d,
    output reg  [WIDTH-1:0]  count,
    output wire              tc           // terminal count
);
    always @(posedge clk or posedge rst) begin
        if (rst)       count <= {WIDTH{1'b0}};
        else if (load) count <= d;
        else if (en)   count <= count + 1'b1;
    end

    assign tc = &count;   // all 1s → terminal count (reduction AND)
endmodule
```

### 4.4 Finite State Machine — Moore Style

**Example: SPI master controller** (simplified — 8-bit transmit)

```
States: IDLE → LOAD → SHIFT (×8) → DONE → IDLE

         start                  bit_cnt==7
  IDLE ─────────► LOAD ──► SHIFT ─────────► DONE ──► IDLE
                            │  ↑
                            └──┘ bit_cnt < 7
```

```verilog
module spi_tx (
    input  wire       clk, rst, start,
    input  wire [7:0] data_in,
    output reg        sclk, mosi, cs_n, done
);
    // State encoding
    localparam IDLE  = 2'b00,
               LOAD  = 2'b01,
               SHIFT = 2'b10,
               DONE  = 2'b11;

    reg [1:0] state, next_state;
    reg [7:0] shift_reg;
    reg [2:0] bit_cnt;

    // State register
    always @(posedge clk or posedge rst)
        if (rst) state <= IDLE;
        else     state <= next_state;

    // Next-state logic (combinational)
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:  if (start)         next_state = LOAD;
            LOAD:                     next_state = SHIFT;
            SHIFT: if (bit_cnt == 7)  next_state = DONE;
            DONE:                     next_state = IDLE;
        endcase
    end

    // Datapath (sequential)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            shift_reg <= 8'd0;
            bit_cnt   <= 3'd0;
            sclk      <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    sclk    <= 1'b0;
                    bit_cnt <= 3'd0;
                end
                LOAD: begin
                    shift_reg <= data_in;
                end
                SHIFT: begin
                    sclk      <= ~sclk;
                    if (sclk) begin                    // shift on falling edge
                        shift_reg <= {shift_reg[6:0], 1'b0};
                        bit_cnt   <= bit_cnt + 1'b1;
                    end
                end
                DONE: begin
                    sclk <= 1'b0;
                end
            endcase
        end
    end

    // Output logic (Moore — depends only on state and datapath)
    assign mosi = shift_reg[7];
    assign cs_n = (state == IDLE);
    assign done = (state == DONE);
endmodule
```

**FSM coding checklist:**
1. Separate state register (`always @(posedge clk)`) from next-state logic (`always @(*)`)
2. Always have a `default` case (or assign default values before the `case`)
3. One-hot encoding for FPGAs, binary for ASICs (tools usually choose)
4. Name states with `localparam` — never use raw numbers

### 4.5 Synchronous FIFO

FIFOs are everywhere in hardware — between pipeline stages, clock domains, producer-consumer buffers:

```verilog
module sync_fifo #(
    parameter DEPTH = 16,
    parameter WIDTH = 8
) (
    input  wire             clk, rst,
    input  wire             wr_en, rd_en,
    input  wire [WIDTH-1:0] wr_data,
    output wire [WIDTH-1:0] rd_data,
    output wire             full, empty
);
    localparam ADDR_W = $clog2(DEPTH);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_W:0]  wr_ptr, rd_ptr;     // extra bit for full/empty detection

    // Write
    always @(posedge clk)
        if (wr_en && !full)
            mem[wr_ptr[ADDR_W-1:0]] <= wr_data;

    // Read
    assign rd_data = mem[rd_ptr[ADDR_W-1:0]];

    // Pointer update
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
        end else begin
            if (wr_en && !full)  wr_ptr <= wr_ptr + 1;
            if (rd_en && !empty) rd_ptr <= rd_ptr + 1;
        end
    end

    // Full: pointers equal but MSBs differ (wrapped around)
    assign full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&
                   (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);
    assign empty = (wr_ptr == rd_ptr);
endmodule
```

The extra MSB in the pointers distinguishes "full" (same address, different MSB) from "empty" (identical pointers). This is the standard trick.

---

## 5. Testbenches and Simulation

### 5.1 Testbench Structure

A testbench is a **non-synthesizable** module that drives inputs and checks outputs:

```verilog
`timescale 1ns / 1ps    // time unit / precision

module adder_tb;
    // 1. Declare signals
    reg  [3:0] a, b;
    reg        cin;
    wire [3:0] sum;
    wire       cout;

    // 2. Instantiate DUT (Device Under Test)
    adder uut (
        .a(a), .b(b), .cin(cin),
        .sum(sum), .cout(cout)
    );

    // 3. Drive stimulus
    initial begin
        // Test vector 1: 3 + 4 + 0 = 7
        a = 4'd3; b = 4'd4; cin = 0;
        #10;   // wait 10ns
        if ({cout, sum} !== 5'd7)
            $display("FAIL: 3+4+0 = %0d, expected 7", {cout, sum});

        // Test vector 2: 15 + 1 + 0 = 16 (overflow to cout)
        a = 4'd15; b = 4'd1; cin = 0;
        #10;
        if ({cout, sum} !== 5'd16)
            $display("FAIL: 15+1+0 = %0d, expected 16", {cout, sum});

        // Test vector 3: 15 + 15 + 1 = 31
        a = 4'd15; b = 4'd15; cin = 1;
        #10;
        if ({cout, sum} !== 5'd31)
            $display("FAIL: 15+15+1 = %0d, expected 31", {cout, sum});

        $display("All tests passed");
        $finish;
    end

    // 4. Dump waveforms (for viewer)
    initial begin
        $dumpfile("adder_tb.vcd");
        $dumpvars(0, adder_tb);
    end
endmodule
```

### 5.2 Clock and Reset Generation

```verilog
// Clock: 100 MHz (10ns period)
reg clk;
initial clk = 0;
always #5 clk = ~clk;    // toggle every 5ns

// Reset: assert for 20ns, then deassert
reg rst;
initial begin
    rst = 1;
    #20;
    rst = 0;
end
```

### 5.3 Self-Checking Testbench with Task

```verilog
module alu_tb;
    reg  [31:0] a, b;
    reg  [3:0]  op;
    wire [31:0] result;
    wire        zero;

    alu_32 uut (.a(a), .b(b), .alu_ctrl(op), .result(result),
                .zero(zero), .negative(), .overflow(), .carry_out());

    integer pass_count = 0;
    integer fail_count = 0;

    task check;
        input [31:0] expected;
        input [255:0] name;       // string
    begin
        #1;
        if (result !== expected) begin
            $display("FAIL %0s: got %h, expected %h", name, result, expected);
            fail_count = fail_count + 1;
        end else begin
            pass_count = pass_count + 1;
        end
    end
    endtask

    initial begin
        // AND
        a = 32'hFF00_FF00; b = 32'h0F0F_0F0F; op = 4'b0000;
        check(32'h0F00_0F00, "AND");

        // ADD
        a = 32'd100; b = 32'd200; op = 4'b0010;
        check(32'd300, "ADD 100+200");

        // SUB
        a = 32'd500; b = 32'd200; op = 4'b0110;
        check(32'd300, "SUB 500-200");

        // SLT: 5 < 10
        a = 32'd5; b = 32'd10; op = 4'b0111;
        check(32'd1, "SLT 5<10");

        $display("\nResults: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end
endmodule
```

### 5.4 Simulation Tools

| Tool                | Free? | Notes                                    |
|---------------------|-------|------------------------------------------|
| **Icarus Verilog**  | Yes   | Open-source, CLI, outputs VCD for GTKWave |
| **Verilator**       | Yes   | Open-source, compiles to C++, fast       |
| **GTKWave**         | Yes   | Open-source waveform viewer              |
| **EDA Playground**  | Yes   | Browser-based, no install                |
| ModelSim (Intel)    | Free* | Free with Quartus Lite                   |
| Vivado Simulator    | Free* | Free with Vivado WebPACK                 |
| Synopsys VCS        | No    | Industry standard, fastest               |
| Cadence Xcelium     | No    | Industry standard                        |

**Workflow with Icarus + GTKWave:**

```bash
# Compile
iverilog -o sim.vvp adder.v adder_tb.v

# Run simulation
vvp sim.vvp

# View waveforms
gtkwave adder_tb.vcd
```

### 5.5 Reading Waveforms

```
              ___     ___     ___     ___     ___
  clk    ____|   |___|   |___|   |___|   |___|   |___
              ↑       ↑       ↑       ↑       ↑
  rst    ‾‾‾‾‾‾‾‾‾‾‾‾‾\________________________________
                        ↑ rst deasserted
  d      ====[ 0x05 ]===[ 0x0A ]===[ 0x0F ]============
                              ↑       ↑
  q      XXXXXXXXX[ 0x00 ]===[ 0x05 ]===[ 0x0A ]======
                    ↑ reset      ↑ captured 0x05    ↑ captured 0x0A
                    value        on rising edge      on rising edge

Key observations:
  - q changes AFTER the clock edge (clock-to-Q delay)
  - q gets the value d had BEFORE the edge (setup time requirement)
  - X at start = uninitialized (no reset applied yet)
```

---

## 6. Synthesis-Oriented Design

### 6.1 What Synthesizes and What Doesn't

Synthesis tools convert your Verilog to gates. Not all Verilog constructs have a hardware equivalent:

| Synthesizable                     | NOT synthesizable                  |
|-----------------------------------|------------------------------------|
| `wire`, `reg`, `integer` (as loop)| `initial` blocks (sim only)        |
| `assign` (continuous)             | `$display`, `$monitor`, `$finish`  |
| `always @(*)` (combinational)     | `#delay` (ignored or error)        |
| `always @(posedge clk)` (seq)     | `real` / `time` types              |
| `if/else`, `case`, `?:`           | File I/O (`$fopen`, `$readmemh`)   |
| `+`, `-`, `*` (arithmetic)        | `/`, `%` (division — large area)   |
| `<<`, `>>` by constant            | `>>` by variable (barrel shifter)  |
| `parameter`, `generate`           | `fork/join`                        |
| Module instantiation              | `deassign`, `force/release`        |

### 6.2 Common Synthesis Pitfalls

**Unintended latch — incomplete case/if:**

```verilog
// BAD — latch inferred for 'out' when sel=2'b11
always @(*) begin
    case (sel)
        2'b00: out = a;
        2'b01: out = b;
        2'b10: out = c;
        // missing 2'b11!
    endcase
end

// FIX 1: add default
always @(*) begin
    case (sel)
        2'b00: out = a;
        2'b01: out = b;
        2'b10: out = c;
        default: out = 32'd0;
    endcase
end

// FIX 2: assign default before case
always @(*) begin
    out = 32'd0;              // default
    case (sel)
        2'b00: out = a;
        2'b01: out = b;
        2'b10: out = c;
    endcase
end
```

**Unintended latch — incomplete if/else:**

```verilog
// BAD — latch on 'out' when en=0
always @(*) begin
    if (en)
        out = data;
    // no else branch!
end

// FIX
always @(*) begin
    if (en) out = data;
    else    out = 32'd0;
end
```

**Mixing blocking/non-blocking:**

```verilog
// BAD — blocking in sequential
always @(posedge clk) begin
    a = b;      // b's value assigned to a immediately
    c = a;      // c gets NEW a — NOT a pipeline!
end

// GOOD — non-blocking gives pipeline behavior
always @(posedge clk) begin
    a <= b;     // a gets old b
    c <= a;     // c gets old a → two-stage pipeline
end
```

### 6.3 Synthesis Results — Reading the Report

After synthesis, check these in the report:

```
=== Resource Usage ===
  LUTs:        1,247 / 53,200  (2.3%)       ← combinational logic
  Registers:     512 / 106,400 (0.5%)       ← flip-flops
  BRAM:            2 / 140     (1.4%)       ← block RAM
  DSP:             4 / 220     (1.8%)       ← multipliers

=== Timing Summary ===
  Target clock: 100 MHz (10.000 ns)
  Worst path:   9.237 ns                     ← critical path
  Slack:        0.763 ns                     ← positive = timing met ✓
  WNS:          0.763 ns                     ← worst negative slack

  If slack < 0 → timing violation → design won't work at target frequency
  Fix: pipeline the critical path, optimize logic, reduce fan-out
```

### 6.4 Coding for Inference

Synthesis tools recognize specific patterns and map them to optimized hardware blocks:

```verilog
// This infers a BRAM (block RAM) — tool recognizes sync read + write pattern
reg [31:0] mem [0:1023];
always @(posedge clk) begin
    if (we) mem[addr] <= wdata;
    rdata <= mem[addr];              // registered read → BRAM
end

// This infers a DSP multiplier
reg [31:0] product;
always @(posedge clk)
    product <= a * b;                // registered multiply → DSP48

// This infers distributed RAM (LUTs) — async read
reg [31:0] mem [0:31];
always @(posedge clk)
    if (we) mem[addr] <= wdata;
assign rdata = mem[addr];            // combinational read → LUT RAM
```

---

## 7. Putting It Together — MIPS Single-Cycle in Verilog

This section implements the single-cycle MIPS processor from Section 6 of the Digital Design Fundamentals guide. Every component is a module you've already seen; this wires them together.

### 7.1 Instruction Memory

```verilog
module imem (
    input  wire [31:0] addr,
    output wire [31:0] instr
);
    reg [31:0] mem [0:255];

    initial $readmemh("program.hex", mem);    // load program at sim time

    assign instr = mem[addr[9:2]];            // word-aligned (addr >> 2)
endmodule
```

### 7.2 Data Memory

```verilog
module dmem (
    input  wire        clk,
    input  wire        we,         // write enable
    input  wire [31:0] addr,
    input  wire [31:0] wdata,
    output wire [31:0] rdata
);
    reg [31:0] mem [0:255];

    always @(posedge clk)
        if (we) mem[addr[9:2]] <= wdata;

    assign rdata = mem[addr[9:2]];
endmodule
```

### 7.3 Register File

```verilog
module regfile (
    input  wire        clk,
    input  wire        we3,        // write enable (port 3)
    input  wire [4:0]  ra1, ra2,   // read addresses
    input  wire [4:0]  wa3,        // write address
    input  wire [31:0] wd3,        // write data
    output wire [31:0] rd1, rd2    // read data
);
    reg [31:0] rf [0:31];

    always @(posedge clk)
        if (we3) rf[wa3] <= wd3;

    // $0 is hardwired to zero
    assign rd1 = (ra1 != 5'd0) ? rf[ra1] : 32'd0;
    assign rd2 = (ra2 != 5'd0) ? rf[ra2] : 32'd0;
endmodule
```

### 7.4 Sign Extender

```verilog
module sign_ext (
    input  wire [15:0] in,
    output wire [31:0] out
);
    assign out = {{16{in[15]}}, in};
endmodule
```

### 7.5 Control Unit

```verilog
module control (
    input  wire [5:0] opcode,
    output reg        reg_dst, alu_src, mem_to_reg,
    output reg        reg_write, mem_read, mem_write,
    output reg        branch, jump,
    output reg  [1:0] alu_op
);
    always @(*) begin
        // Defaults
        {reg_dst, alu_src, mem_to_reg, reg_write} = 4'b0;
        {mem_read, mem_write, branch, jump}        = 4'b0;
        alu_op = 2'b00;

        case (opcode)
            6'b000000: begin // R-type
                reg_dst   = 1; reg_write = 1;
                alu_op    = 2'b10;
            end
            6'b100011: begin // lw
                alu_src   = 1; mem_to_reg = 1;
                reg_write = 1; mem_read   = 1;
            end
            6'b101011: begin // sw
                alu_src   = 1; mem_write  = 1;
            end
            6'b000100: begin // beq
                branch = 1; alu_op = 2'b01;
            end
            6'b001000: begin // addi
                alu_src   = 1; reg_write  = 1;
            end
            6'b000010: begin // j
                jump = 1;
            end
        endcase
    end
endmodule
```

### 7.6 ALU Control

```verilog
module alu_control (
    input  wire [1:0] alu_op,
    input  wire [5:0] funct,
    output reg  [3:0] alu_ctrl
);
    always @(*) begin
        case (alu_op)
            2'b00: alu_ctrl = 4'b0010;              // ADD (lw/sw/addi)
            2'b01: alu_ctrl = 4'b0110;              // SUB (beq)
            2'b10: case (funct)                      // R-type
                6'b100000: alu_ctrl = 4'b0010;       // add
                6'b100010: alu_ctrl = 4'b0110;       // sub
                6'b100100: alu_ctrl = 4'b0000;       // and
                6'b100101: alu_ctrl = 4'b0001;       // or
                6'b101010: alu_ctrl = 4'b0111;       // slt
                default:   alu_ctrl = 4'b0010;
            endcase
            default: alu_ctrl = 4'b0010;
        endcase
    end
endmodule
```

### 7.7 Top-Level: Single-Cycle MIPS

```verilog
module mips_single_cycle (
    input wire clk, rst
);
    // ── Wires ──────────────────────────────────────────────────
    wire [31:0] pc, pc_plus4, pc_branch, pc_jump, pc_next;
    wire [31:0] instr;
    wire [31:0] rd1, rd2, alu_result, read_data, write_data;
    wire [31:0] sign_imm, sign_imm_sl2;
    wire [4:0]  write_reg;
    wire [3:0]  alu_ctrl;
    wire        zero;

    // Control signals
    wire reg_dst, alu_src, mem_to_reg, reg_write;
    wire mem_read, mem_write, branch, jump;
    wire [1:0] alu_op;

    // ── PC register ────────────────────────────────────────────
    reg [31:0] pc_reg;
    always @(posedge clk or posedge rst)
        if (rst) pc_reg <= 32'h0000_0000;
        else     pc_reg <= pc_next;
    assign pc = pc_reg;

    // ── PC logic ───────────────────────────────────────────────
    assign pc_plus4    = pc + 32'd4;
    assign sign_imm_sl2 = {sign_imm[29:0], 2'b00};       // shift left 2
    assign pc_branch   = pc_plus4 + sign_imm_sl2;
    assign pc_jump     = {pc_plus4[31:28], instr[25:0], 2'b00};

    wire  pc_src = branch & zero;
    assign pc_next = jump   ? pc_jump :
                     pc_src ? pc_branch :
                              pc_plus4;

    // ── Instruction memory ─────────────────────────────────────
    imem imem_inst (.addr(pc), .instr(instr));

    // ── Control ────────────────────────────────────────────────
    control ctrl (
        .opcode(instr[31:26]),
        .reg_dst(reg_dst), .alu_src(alu_src),
        .mem_to_reg(mem_to_reg), .reg_write(reg_write),
        .mem_read(mem_read), .mem_write(mem_write),
        .branch(branch), .jump(jump), .alu_op(alu_op)
    );

    // ── Register file ──────────────────────────────────────────
    assign write_reg = reg_dst ? instr[15:11] : instr[20:16];  // rd : rt

    regfile rf (
        .clk(clk), .we3(reg_write),
        .ra1(instr[25:21]), .ra2(instr[20:16]),
        .wa3(write_reg), .wd3(write_data),
        .rd1(rd1), .rd2(rd2)
    );

    // ── Sign extend ────────────────────────────────────────────
    sign_ext se (.in(instr[15:0]), .out(sign_imm));

    // ── ALU control ────────────────────────────────────────────
    alu_control alu_c (
        .alu_op(alu_op), .funct(instr[5:0]),
        .alu_ctrl(alu_ctrl)
    );

    // ── ALU ────────────────────────────────────────────────────
    wire [31:0] alu_b = alu_src ? sign_imm : rd2;

    alu_32 alu (
        .a(rd1), .b(alu_b), .alu_ctrl(alu_ctrl),
        .result(alu_result), .zero(zero),
        .negative(), .overflow(), .carry_out()
    );

    // ── Data memory ────────────────────────────────────────────
    dmem dmem_inst (
        .clk(clk), .we(mem_write),
        .addr(alu_result), .wdata(rd2),
        .rdata(read_data)
    );

    // ── Write-back MUX ─────────────────────────────────────────
    assign write_data = mem_to_reg ? read_data : alu_result;
endmodule
```

### 7.8 Testbench — Running a Program

```verilog
`timescale 1ns / 1ps

module mips_tb;
    reg clk, rst;

    mips_single_cycle uut (.clk(clk), .rst(rst));

    // Clock: 100 MHz
    initial clk = 0;
    always #5 clk = ~clk;

    // Reset and run
    initial begin
        $dumpfile("mips_tb.vcd");
        $dumpvars(0, mips_tb);

        rst = 1;
        #20;
        rst = 0;

        // Run for 500 cycles
        #5000;

        // Check register results (read regfile directly in testbench)
        $display("$t0 (r8)  = %0d", uut.rf.rf[8]);
        $display("$t1 (r9)  = %0d", uut.rf.rf[9]);
        $display("$t2 (r10) = %0d", uut.rf.rf[10]);
        $finish;
    end
endmodule
```

**Example program (`program.hex`):**

```
// addi $t0, $zero, 5      →  0x20080005
// addi $t1, $zero, 3      →  0x20090003
// add  $t2, $t0, $t1      →  0x01095020
// sw   $t2, 0($zero)      →  0xAC0A0000
// lw   $t3, 0($zero)      →  0x8C0B0000
20080005
20090003
01095020
AC0A0000
8C0B0000
```

After simulation: `$t2` should hold 8 (5 + 3), and `$t3` should also hold 8 (loaded from memory).

> **Next step:** extend this single-cycle design to a 5-stage pipeline by adding pipeline registers (IF/ID, ID/EX, EX/MEM, MEM/WB), a forwarding unit, and a hazard detection unit. The Digital Design Fundamentals guide Section 6 has the datapath and logic — translating it to Verilog follows the same module-per-stage pattern.

---

## 8. Synthesis and FPGA Implementation

### 8.1 FPGA Architecture Overview

An FPGA implements your design by configuring programmable logic blocks (CLBs) connected by a programmable routing fabric:

```
FPGA chip (simplified):

  ┌─────────────────────────────────────────────┐
  │  IOB   IOB   IOB   IOB   IOB   IOB   IOB   │  ← I/O blocks (pins)
  │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
  │ │ CLB │─│ CLB │─│ CLB │─│ CLB │─│ CLB │   │
  │ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
  │ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   │
  │ │ CLB │─│BRAM │─│ CLB │─│ DSP │─│ CLB │   │
  │ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
  │ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   │
  │ │ CLB │─│ CLB │─│ CLB │─│ CLB │─│ CLB │   │
  │ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │
  │  IOB   IOB   IOB   IOB   IOB   IOB   IOB   │
  └─────────────────────────────────────────────┘

CLB contains: LUTs (6-input, configurable truth tables) + flip-flops
BRAM: 36 Kbit block RAM for memories
DSP: hardened multiplier-accumulate blocks
IOB: configurable I/O standards (LVCMOS, LVDS, etc.)
```

**CLB detail (Xilinx 7-series style):**

```
  ┌─────────────────────────────────────┐
  │  Slice (×2 per CLB)                 │
  │                                     │
  │  [6-LUT] → [MUX] → [FF]  →  out   │
  │  [6-LUT] → [MUX] → [FF]  →  out   │
  │  [6-LUT] → [MUX] → [FF]  →  out   │
  │  [6-LUT] → [MUX] → [FF]  →  out   │
  │                                     │
  │  + carry chain for fast arithmetic  │
  └─────────────────────────────────────┘

Each 6-LUT implements ANY function of 6 inputs (2^6 = 64-entry truth table)
Each FF = one flip-flop with sync/async reset/set, clock enable
```

### 8.2 FPGA Design Flow

```
Verilog source
    │
    ▼
Synthesis (Vivado / Quartus)
    │  → maps to LUTs, FFs, BRAM, DSP
    ▼
Implementation
    ├── Place: assign CLBs to physical locations
    ├── Route: connect CLBs through routing fabric
    └── Timing: verify setup/hold met at target frequency
    │
    ▼
Bitstream generation (.bit / .sof)
    │
    ▼
Program FPGA (JTAG / flash)
```

### 8.3 Constraints File

Tell the tools about your physical board — pin assignments and clock:

```tcl
# Xilinx XDC constraints (example: Basys 3 board)

# Clock: 100 MHz oscillator on pin W5
set_property -dict {PACKAGE_PIN W5 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# Reset: center button
set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports rst]

# LEDs
set_property -dict {PACKAGE_PIN U16 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN E19 IOSTANDARD LVCMOS33} [get_ports {led[1]}]

# Switches
set_property -dict {PACKAGE_PIN V17 IOSTANDARD LVCMOS33} [get_ports {sw[0]}]
set_property -dict {PACKAGE_PIN V16 IOSTANDARD LVCMOS33} [get_ports {sw[1]}]
```

### 8.4 Recommended FPGA Boards for Learning

| Board                  | FPGA              | Price  | Best for                        |
|------------------------|-------------------|--------|---------------------------------|
| Digilent Basys 3       | Xilinx Artix-7    | ~$150  | First board, coursework         |
| Digilent Nexys A7      | Xilinx Artix-7    | ~$265  | Larger designs, DDR memory      |
| Terasic DE10-Lite      | Intel MAX 10      | ~$85   | Budget option, Quartus          |
| Terasic DE1-SoC        | Intel Cyclone V   | ~$200  | ARM HPS + FPGA, Linux           |
| Digilent Arty A7       | Xilinx Artix-7    | ~$130  | Arduino form-factor, RISC-V     |
| Xilinx KV260 / KR260   | Zynq UltraScale+  | ~$250 | AI inference (DPU), production  |

---

## Resources

| Resource | Type | Focus |
|----------|------|-------|
| *Digital Design and Computer Architecture* — Harris & Harris | Textbook | Verilog + processor design (ch. 4–7) |
| *Verilog HDL* — Samir Palnitkar | Textbook | Comprehensive Verilog reference |
| HDLBits (hdlbits.01xz.net) | Interactive | 180+ Verilog exercises with auto-grading |
| Nandland (nandland.com) | Tutorial | Beginner Verilog + FPGA with real boards |
| ASIC World (asic-world.com) | Reference | Verilog syntax, examples, testbenches |
| EDA Playground (edaplayground.com) | Online sim | Browser-based Verilog simulation |
| *Computer Organization and Design* — Patterson & Hennessy | Textbook | MIPS/RISC-V datapath and pipeline |
| OpenCores (opencores.org) | Open-source | Real-world Verilog IP to study |
| Icarus Verilog + GTKWave | Tools | Free simulator + waveform viewer |

---

## Projects

| # | Project | Concepts practiced | Complexity |
|---|---------|-------------------|------------|
| 1 | **Full adder → 32-bit RCA** | Dataflow, structural, generate | Beginner |
| 2 | **ALU with 8 operations + flags** | Behavioral, case, concatenation | Beginner |
| 3 | **BCD to 7-segment decoder** | Combinational, case, FPGA I/O | Beginner |
| 4 | **8-bit shift register + LFSR** | Sequential, feedback, PRNG | Beginner |
| 5 | **Synchronous FIFO (parameterized)** | Pointers, full/empty, BRAM | Intermediate |
| 6 | **SPI master controller** | FSM, shift register, protocol | Intermediate |
| 7 | **UART TX + RX** | FSM, baud rate, oversampling | Intermediate |
| 8 | **Single-cycle MIPS processor** | All of the above: datapath, control, memory | Advanced |
| 9 | **Pipelined MIPS + forwarding** | Pipeline registers, hazard detection, MUX | Advanced |
| 10 | **Matrix multiply accelerator** | BRAM tiling, FSM control, DSP inference | Advanced |
