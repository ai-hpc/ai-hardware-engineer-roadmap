# Digital Design Fundamentals

<div class="course-identity auto-course" style="--course-accent: #65a30d; --course-accent-rgb: 101, 163, 13;" markdown="1">
<div class="course-identity__icon">DDF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Derive the integer datapath before coding it.</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


The circuits inside every AI accelerator — tensor cores, SRAM banks, systolic arrays — are built from the same primitives covered here. This guide takes you from bits to memory hierarchies, with deliberate attention to the representations and building blocks that show up every day in hardware design.

**Learning route:** use sections 1–4 first if you are new to bits and clocks; section 0 gives manufacturing context, and sections 5–6 connect the circuits to memory and processors. Prerequisites are basic algebra. Plan four 60–90-minute sessions plus the [CPU logic and cost lab](labs/README.md). Produce arithmetic vectors, a timing calculation, and a tested interface before continuing to [HDLs](../Hardware%20Description%20Languages%20%28HDLs%29/Guide.md). This is the digital/architecture layer for RTL, FPGA, embedded, and accelerator engineers.

---

## 0. From Design to Silicon — How Chips Are Made

### 0.1 The Tapeout Flow

"Tapeout" is the moment a chip design team hands off the final design files to a foundry (TSMC, Samsung, Intel Foundry, GlobalFoundries) for manufacturing. The name comes from the old days when designs were literally written to magnetic tape. Today it means delivering a set of verified GDSII/OASIS files — the geometric description of every transistor, wire, and via on the chip.

The full journey from idea to working silicon:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CHIP DESIGN FLOW                                 │
│                                                                         │
│  Specification                                                          │
│      │                                                                  │
│      ▼                                                                  │
│  Architecture Design          "What does the chip do?"                  │
│      │                        ISA, block diagram, memory map            │
│      ▼                                                                  │
│  RTL Design (Verilog/VHDL)    "Describe behavior in HDL"                │
│      │                        modules, FSMs, datapaths                  │
│      ▼                                                                  │
│  Functional Verification      "Does the RTL match the spec?"            │
│      │                        testbenches, UVM, formal verification     │
│      ▼                                                                  │
│  Logic Synthesis              "Convert RTL to gates"                    │
│      │                        tool: Synopsys Design Compiler            │
│      │                        input: RTL + standard cell library        │
│      │                        output: gate-level netlist                │
│      ▼                                                                  │
│  Place and Route (PnR)        "Put gates on chip, connect wires"        │
│      │                        tool: Cadence Innovus / Synopsys ICC2     │
│      │                        floorplan → placement → CTS → routing     │
│      ▼                                                                  │
│  Sign-off Checks              "Is it manufacturable and correct?"        │
│      │   ├── Timing (STA)     static timing analysis, all corners       │
│      │   ├── DRC              design rule check (min spacing, width)    │
│      │   ├── LVS              layout vs schematic (geometry = netlist?) │
│      │   ├── ERC              electrical rule check                     │
│      │   └── IR drop / EM     power integrity, electromigration         │
│      ▼                                                                  │
│  ★ TAPEOUT ★                  "Ship GDSII to foundry"                   │
│      │                                                                  │
│      ▼                                                                  │
│  Fabrication                 process-specific manufacturing schedule            │
│      │                                                                  │
│      ▼                                                                  │
│  Packaging & Test             die → package → burn-in → final test      │
│      │                                                                  │
│      ▼                                                                  │
│  Silicon Bring-up             first boot, characterization              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 0.2 RTL to Gates: Logic Synthesis

You write hardware in Verilog or VHDL — a behavioral description of what each register, MUX, and ALU does. A synthesis tool maps supported constructs to cells in the selected **standard cell library**, supplied by a foundry or library provider. FPGA synthesis instead maps to the target device’s logic, registers, memories, and arithmetic resources.

```
Your RTL (Verilog):                     Synthesized gates:

  assign y = (a & b) | c;      →       AND2 followed by OR2 (one possible mapping)
                                        (actual cell choice depends on the selected library)

  always @(posedge clk)
    if (en) q <= d;             →       DFF with enable, or DFF plus feedback MUX
```

A standard cell library contains hundreds of pre-designed, pre-characterized cells:

| Cell type          | Examples                              |
|--------------------|---------------------------------------|
| Logic gates        | INV, NAND2, NAND3, NOR2, AOI22, MUX2 |
| Sequential         | DFF, DFF with reset, DFF with enable  |
| Buffers/drivers    | BUF, CLKBUF, CLKINV                  |
| Special            | Tie-high, tie-low, filler cells       |

Each cell comes with:
- **Layout** (geometric shapes for each process layer)
- **Timing model** (.lib) — delay, setup, hold at various voltage/temperature corners
- **Power model** — leakage + switching power
- **Parasitic model** — capacitance, resistance

The synthesis tool optimizes for your constraints — typically timing (target clock frequency), area, and power — making trade-offs like using faster (larger) cells on critical paths and slower (smaller) cells elsewhere.

### 0.3 Place and Route

After synthesis, you have a netlist: instances of cells or target primitives and their connections. Its size depends on the design. Place and route turns this into a physical layout.

```
Step 1: Floorplanning
  ┌──────────────────────────────────┐
  │  ┌──────┐  ┌────────┐  ┌──────┐ │
  │  │ SRAM │  │Compute │  │ SRAM │ │
  │  │Block │  │ Core   │  │Block │ │
  │  └──────┘  └────────┘  └──────┘ │
  │  ┌────────────────────────────┐  │
  │  │        I/O ring            │  │
  │  └────────────────────────────┘  │
  └──────────────────────────────────┘
  Decide where major blocks go, power grid, pin placement

Step 2: Placement
  Standard cells placed in rows between power rails (VDD/VSS)
  ┌─VDD─────────────────────────────┐
  │ [NAND][INV][DFF][BUF][MUX]...  │
  ├─VSS─────────────────────────────┤
  │ [NOR][AOI][DFF][INV][NAND]...  │
  ├─VDD─────────────────────────────┤
  │ ...                             │
  └─────────────────────────────────┘

Step 3: Clock Tree Synthesis (CTS)
  Build a balanced tree of clock buffers so CLK arrives
  at sequential endpoints within the design’s skew/latency constraints

Step 4: Routing
  Connect signals using the process’s available routing layers
  Lower metals: short local wires
  Upper metals: long global wires, power distribution
```

### 0.4 Fabrication, packaging, and yield

Fabrication repeats material deposition, patterning, etching, implantation, and other process-specific operations. The process design kit defines the permitted layers and rules. A node name is a technology-family label; it is not a universal transistor dimension or a comparable measured density across vendors. Logic, SRAM, and analog blocks also have different densities.

Keep a real example separate from a generic drawing. NVIDIA identifies GH100 as a TSMC **4N** design, with **144 SMs in the full architecture**, **132 enabled in H100 SXM5**, and **114 in the PCIe version**. Those are different product configurations. Disabled resources do not prove that every disabled block was defective; power, yield, and product configuration all matter. [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

A package connects dies to a board and supplies power, signals, protection, and heat paths. HBM uses vertically stacked memory dies and a wide interface near the accelerator. Fine-pitch package routing supports short, dense interconnects that would be difficult to implement with ordinary PCB routing. This is an engineering tradeoff, not a claim that routing 1,024 signals on any PCB is physically impossible.

| Technique | What it describes | Question for the design |
|---|---|---|
| Wire bonding | Wires between die pads and package | Pin count, parasitics, cost |
| Flip-chip attachment | Bumps connecting a face-down die | Power delivery and interconnect density |
| Interposer/package routing | Connections among nearby dies | Bandwidth, distance, assembly yield |
| Vertical stacking | Dies placed above one another | Thermal path, bonding, test access |
| Chiplet partitioning | Function divided among several dies | Die-to-die protocol, known-good-die test, total cost |

These categories can coexist in one product. Chiplets are not themselves one bonding technology.

**Worked yield calculation — fictional inputs:** suppose a wafer provides 60 complete candidate dies, wafer-level yield is 80%, and wafer cost is 18,000 currency units. Expected good dies are `60 × 0.8 = 48`; the wafer-cost allocation is `18,000 / 48 = 375` per good die. At 60% yield, it becomes 500. Neither figure includes design, packaging, test, assembly losses, or commercial margin. Actual wafer geometry, defect distribution, and costs must come from the project; these numbers are not an H100 cost estimate.

### 0.5 Plan the evidence across the chip flow

| Stage | Artifact | Question it can answer |
|---|---|---|
| Architecture | Workload and bandwidth model | Can the proposed resources plausibly meet the requirement? |
| RTL verification | Reference model, assertions, tests | Does the described behavior satisfy the tested contract? |
| Synthesis | Mapped netlist and resource report | How does the tool implement that behavior in the selected library? |
| Implementation | Placement, routing, timing constraints/report | Does the implemented design meet the constrained timing paths? |
| Sign-off | Required rule, timing, power-integrity and equivalence checks | Does the design pass the selected checks/corners? |
| Manufacturing test | Wafer/package test results | Which manufactured devices meet their test limits? |
| Bring-up | Measured hardware evidence | Does the device function in its intended system? |

A passing RTL simulation does not establish timing closure or successful fabrication. A timing report only covers the constraints and paths actually analyzed. Project duration and cost depend on reuse, scope, process, tooling, verification, and manufacturing access; use a real project plan instead of assuming a universal multi-year schedule.

---

## 1. Number Systems

### 1.1 Binary, Hexadecimal, and Positional Notation

Every number in a digital system is a pattern of bits. Positional notation makes the value of each bit depend on its position:

```
Decimal 173  =  1×10² + 7×10¹ + 3×10⁰
Binary 10101101  =  1×2⁷ + 0×2⁶ + 1×2⁵ + 0×2⁴ + 1×2³ + 1×2² + 0×2¹ + 1×2⁰
               =  128 + 32 + 8 + 4 + 1  =  173
Hex 0xAD       =  10×16¹ + 13×16⁰  =  160 + 13  =  173
```

Hex is used everywhere in hardware because one hex digit maps exactly to four bits — easier to read, easier to type.

```
Binary group:  1010  1101
Hex digit:      A     D      →  0xAD
```

**Conversion drill** (do these by hand until they feel automatic):

| Decimal | Binary     | Hex  |
|---------|-----------|------|
| 0       | 0000 0000 | 0x00 |
| 127     | 0111 1111 | 0x7F |
| 128     | 1000 0000 | 0x80 |
| 255     | 1111 1111 | 0xFF |
| 256     | 0001 0000 0000 | 0x100 |

### 1.2 Signed Integers: Two's Complement

Two’s complement is the standard signed-integer representation in the processors studied here. Sign-magnitude remains relevant when understanding floating-point sign fields; distinguish integer encodings from floating-point formats.

| Scheme          | +5       | −5       | Problem                          |
|-----------------|----------|----------|----------------------------------|
| Sign-magnitude  | 0000 0101 | 1000 0101 | Two zeros (+0, −0); complex adder |
| One's complement | 0000 0101 | 1111 1010 | Two zeros; end-around carry      |
| **Two's complement** | **0000 0101** | **1111 1011** | One zero; same adder for add/sub |

**Two's complement rule:** flip all bits, add 1.

```
+5  =  0000 0101
flip   1111 1010
+1     1111 1011   ← this is −5 in 8-bit two's complement
```

**Verify by addition** — a correct negation adds to zero:

```
  0000 0101   (+5)
+ 1111 1011   (−5)
-----------
  0000 0000   (0, carry out discarded)  ✓
```

**Range** for N-bit two's complement: −2^(N−1) to +2^(N−1) − 1.
For 8-bit: −128 to +127. For 32-bit: −2,147,483,648 to +2,147,483,647.

**Sign extension** — when widening a value (e.g., INT8 → INT32), copy the sign bit into all new high bits:

```
INT8  −5  =  1111 1011
INT32 −5  =  1111 1111 1111 1111 1111 1111 1111 1011
                ↑ all new bits filled with sign bit (1)
```

> **AI hardware connection:** matrix multiply accumulate units (MMA/WMMA) operate on INT8 with INT32 accumulators. The sign extension from INT8 to INT32 is automatic in hardware, but matters when you mix signed/unsigned operands.

### 1.3 IEEE 754 Floating-Point

Floating-point trades the fixed range of integers for a sliding window of precision.

```
Normal finite value = (−1)^S × (1 + fraction / 2^fraction_bits) × 2^(Exponent − Bias)
```

| Format    | Width | Sign | Exponent | Mantissa | Bias | ~Range          | ~Precision |
|-----------|-------|------|----------|----------|------|-----------------|------------|
| FP32      | 32b   | 1    | 8        | 23       | 127  | ±3.4×10^38      | ~7 digits  |
| FP64      | 64b   | 1    | 11       | 52       | 1023 | ±1.8×10^308     | ~15 digits |
| FP16      | 16b   | 1    | 5        | 10       | 15   | ±65504          | ~3 digits  |
| BF16      | 16b   | 1    | 8        | 7        | 127  | about ±3.39×10^38 | ~2 digits |
| FP8 E4M3  | 8b    | 1    | 4        | 3        | 7    | ±448            | ~1 digit   |
| FP8 E5M2  | 8b    | 1    | 5        | 2        | 15   | ±57344          | ~0.5 digit |

**FP32 example — representing 0.15625:**

```
0.15625 = 0.00101 in binary = 1.01 × 2^(−3)

S = 0
Exponent field = −3 + 127 = 124 = 0111 1100
Mantissa = 01 followed by 21 zeros

Bit pattern: 0  01111100  01000000000000000000000
             S  exponent  mantissa
```

**FP32 special values** (the exponent values below are specifically FP32):

| Pattern                  | Meaning   |
|--------------------------|-----------|
| Exp=0, Mantissa=0        | ±Zero     |
| Exp=0, Mantissa≠0        | Subnormal |
| Exp=255, Mantissa=0      | ±Infinity |
| Exp=255, Mantissa≠0      | NaN       |

For ordinary binary floating-point arithmetic, distinguish nearest-even, toward zero, toward positive infinity, and toward negative infinity. The operation/API determines the selected mode. Nearest-even means:

```
Exact result  →  nearest representable value
Ties (exactly halfway)  →  choose the one with even last bit
```

Rounding modes matter in AI because accumulated rounding error over millions of operations can cause training instability. Stochastic rounding (rounding up or down randomly, weighted by proximity) is used in some FP8 training flows.

### 1.4 AI Float Formats: FP16, BF16, FP8

Modern AI training and inference use reduced-precision formats to fit more data in SRAM and reduce multiply-add energy.

```
FP32   [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM]   32 bits
BF16   [S|EEEEEEEE|MMMMMMM]                   16 bits  (shorter significand; conversion has a rounding contract)
FP16   [S|EEEEE|MMMMMMMMMM]                   16 bits  (different exponent width)
FP8    [S|EEEE|MMM]  or  [S|EEEEE|MM]          8 bits
```

**BF16 vs FP16 trade-off:**

| Property         | FP16                  | BF16                        |
|------------------|-----------------------|-----------------------------|
| Exponent bits    | 5                     | 8 (same as FP32)            |
| Range            | ±65504                | ±3.4×10^38                  |
| Precision        | ~3 decimal digits     | ~2 decimal digits           |
| FP32 conversion  | Round; overflow behavior depends on instruction/mode | Round to the BF16 significand; handle special values |
| Usage            | Inference, some training | Mixed-precision training |

**FP8 — two variants for different roles:**

- **E4M3** (4-bit exp, 3-bit mantissa): more precision, less range → weights and activations
- **E5M2** (5-bit exp, 2-bit mantissa): more range, less precision → gradients

FP8 use requires a format, scaling, rounding, and overflow contract. E4M3/E5M2 roles are common design choices, not universal rules; FP8 tensor scaling and training loss scaling are related but distinct mechanisms.

### 1.5 Fixed-Point and Quantization

Fixed-point represents Q = integer × 2^(−f) where f is the number of fractional bits.

```
Here Q8.8 means signed 16-bit storage, 8 fractional bits, sign included in the upper 8 bits

Value 3.75:
  Integer part:  3  = 0000 0011
  Fraction part: 0.75 = 0.11 in binary → stored as 1100 0000

Stored: 0000 0011  1100 0000  = 0x03C0
```

**Quantization** maps floating-point weights to INT8 for efficient inference:

```
q = clamp(round_to_nearest_even(weight / scale), -127, 127)
scale = max(|weights|) / 127          ← per-tensor
scale = max(|row|) / 127              ← optional per-row granularity

Dequantize: FP32 ≈ INT8 × scale
```

This symmetric example uses zero point 0 and the range −127…127; an all-zero group needs a defined positive scale, such as 1, to avoid division by zero. Per-row scaling can reduce quantization error but adds metadata and kernel requirements; measure its effect. Many INT8 matrix instructions accumulate into INT32, but scaling, bias, overflow, and output requantization remain separate parts of the numerical contract.

### 1.6 Error Detection and Correction

**Parity** — add one bit so the total number of 1s is even (even parity) or odd (odd parity):

```
Data: 1011 0110   →  five 1s  →  odd
Even parity bit: 1  (makes six 1s → even)
Transmitted: 1011 0110 1
```

Parity detects any single-bit error but cannot locate or correct it.

**Hamming(7,4)** — encode 4 data bits into 7 bits with 3 parity bits positioned at powers of 2:

```
Positions:  p1  p2  d1  p4  d2  d3  d4
            1   2   3   4   5   6   7

p1 covers positions 1,3,5,7  (odd positions)
p2 covers positions 2,3,6,7
p4 covers positions 4,5,6,7

Syndrome = check_p1 + 2*check_p2 + 4*check_p4
Binary display order: (check p4)(check p2)(check p1)
```

Hamming(7,4) has minimum distance 3: it can correct one error, or detect up to two errors if used only as a detector. A decoder that always corrects a nonzero syndrome can miscorrect a double error. Adding an overall parity bit gives an extended Hamming(8,4) SECDED code: correct a single error and distinguish a double error as uncorrectable. The [CPU lab](labs/README.md) exhaustively checks all 16 messages, 128 single-bit errors, and 448 double-bit errors. Three or more errors are outside that guarantee.

**CRC (Cyclic Redundancy Check)** — treats the data as a polynomial and divides by a generator polynomial:

```
Data polynomial D(x) × x^r  mod  G(x)  =  R(x)   (remainder = CRC)
Transmitted: D(x) concat R(x)
Receiver: divide received polynomial by G(x) — nonzero remainder means error
```

CRC-32 (used in Ethernet, ZIP, PNG) uses G(x) = x^32 + x^26 + x^23 + ... + 1.
A familiar memory example protects 64 data bits with 8 check bits using SECDED. Real systems use several ECC organizations; on-die ECC and system-level ECC protect different boundaries. Identify the actual code and protected path before assuming a correction guarantee.

---

## 2. Boolean Algebra and Logic Gates

### 2.1 Gates and Truth Tables

Every digital function reduces to NAND gates (or NOR gates) — they are functionally complete.

```
AND gate:          OR gate:          NOT gate:         NAND gate:
  A──┐              A──┐              A──○──Y           A──┐
     ├──Y              ├──Y                             ├──○──Y
  B──┘              B──┘                             B──┘

Truth tables:

A B | AND  OR  NAND  NOR  XOR  XNOR
0 0 |  0   0    1    1    0    1
0 1 |  0   1    1    0    1    0
1 0 |  0   1    1    0    1    0
1 1 |  1   1    0    0    0    1
```

**Universal gates — NAND implements everything:**

```
NOT A    =  NAND(A, A)
A AND B  =  NAND(NAND(A,B), NAND(A,B))   i.e. NOT(NAND(A,B))
A OR B   =  NAND(NAND(A,A), NAND(B,B))   i.e. NOT(NOT A) NAND NOT(NOT B)
```

### 2.2 Boolean Laws

| Law             | AND form                | OR form                 |
|-----------------|-------------------------|-------------------------|
| Identity        | A · 1 = A               | A + 0 = A               |
| Null            | A · 0 = 0               | A + 1 = 1               |
| Idempotent      | A · A = A               | A + A = A               |
| Complement      | A · Ā = 0               | A + Ā = 1               |
| Double negative | ¬(¬A) = A               |                         |
| Commutative     | A · B = B · A           | A + B = B + A           |
| Associative     | (A·B)·C = A·(B·C)       | (A+B)+C = A+(B+C)       |
| Distributive    | A·(B+C) = A·B + A·C     | A+(B·C) = (A+B)·(A+C)  |
| **De Morgan's** | ¬(A·B) = Ā + B̄          | ¬(A+B) = Ā · B̄          |
| Absorption      | A·(A+B) = A             | A + A·B = A             |

**De Morgan's** is the most used law in logic design — it lets you push negations through gates and convert AND↔OR.

### 2.3 Karnaugh Maps

K-maps group minterms visually to find the minimal sum-of-products (SOP) expression.

**4-variable K-map layout** (Gray code order — adjacent cells differ by one bit):

```
        CD
AB    00  01  11  10
  00 | 0 | 1 | 3 | 2 |
  01 | 4 | 5 | 7 | 6 |
  11 |12 |13 |15 |14 |
  10 | 8 | 9 |11 |10 |
```

**Rules for grouping:**
- Group size must be a power of 2: 1, 2, 4, 8, 16
- Groups must be rectangular (wrap around edges)
- Use the largest possible groups
- Cover every 1; don't cover 0s
- Each group eliminates the variables that change within it

**Example** — minimize F(A,B,C,D) = Σm(0,2,4,5,6,7,8,10,13,15):

```
        CD
AB    00  01  11  10
  00 | 1 | 0 | 0 | 1 |   →  Group of 4: cells 0,2,8,10 → B̄D̄
  01 | 1 | 1 | 1 | 1 |   →  Group of 4: cells 4,5,6,7 → AB̄
  11 | 0 | 1 | 1 | 0 |   →  Group of 2: cells 13,15 → ACD
  10 | 1 | 0 | 0 | 1 |

F = B̄D̄ + AB̄ + ACD
```

Without K-map: 10 minterms = up to 10 four-literal terms. K-map: 3 terms.

### 2.4 CMOS Implementation

CMOS (Complementary MOS) uses PMOS pull-up networks and NMOS pull-down networks. The two networks are duals of each other.

```
CMOS NAND gate (2-input):

        VDD
         |
    ─────┤  PMOS (A)     ← parallel pull-up
    │    └──┐
    └──────┤  PMOS (B)
            └─── Y
            ┌──
    ┌──────┤  NMOS (A)   ← series pull-down
    │    ┌──┘
    ─────┤  NMOS (B)
         |
        GND

Logic:  Y = NAND(A,B) = ¬(A · B)

If A=1 AND B=1:  both NMOS on (pull Y low), both PMOS off → Y = 0  ✓
If A=0 OR  B=0:  one PMOS on (pull Y high), NMOS chain broken → Y = 1  ✓
```

**Key CMOS properties:**

| Property             | Value                                        |
|----------------------|----------------------------------------------|
| Static power         | Near zero (only leakage)                     |
| Dynamic power        | C × V² × f (scales with switching activity) |
| Noise margin         | ~40% of VDD                                  |
| Fan-out              | Theoretically unlimited (MOS gate is capacitive) |
| Inversion            | Every CMOS gate inverts (NAND, NOR, NOT)     |

> **Why only NAND/NOR?** AND = NAND + NOT = two gate stages. An extra inversion costs delay and area. Modern synthesis tools always work in NAND/NOR/NOT internally.

---

## 3. Combinational Logic

### 3.1 Multiplexers and Decoders

**MUX (2:1)** — select one of two data inputs based on a control signal:

```
  D0 ──┐
        ├── Y   Y = S̄·D0 + S·D1
  D1 ──┘
        ↑
        S (select)

  S=0 → Y=D0
  S=1 → Y=D1
```

**4:1 MUX** from two layers of 2:1 MUX:

```
  D0 ─┐                  S1=0: top 2:1 → D0 or D1
  D1 ─┤ 2:1 MUX ─┐       S1=1: bot 2:1 → D2 or D3
       └──(S0)    │
  D2 ─┐           ├─ 2:1 MUX ── Y
  D3 ─┤ 2:1 MUX ─┘         └──(S1)
       └──(S0)
```

**Decoder (2-to-4)** — activate exactly one of N outputs:

```
Inputs: A1, A0
Outputs: Y0..Y3

Y0 = Ā1·Ā0   (active when A=00)
Y1 = Ā1·A0   (active when A=01)
Y2 = A1·Ā0   (active when A=10)
Y3 = A1·A0   (active when A=11)
```

Decoders appear as row/column address decoders in every SRAM and register file.

### 3.2 Adders: Ripple-Carry vs Carry-Lookahead

**Full adder (1-bit):**

```
  A ──┐
  B ──┼── XOR ──────── Sum  =  A ⊕ B ⊕ Cin
  Cin─┘

  Carry-out  =  A·B  +  Cin·(A⊕B)
             =  A·B  +  Cin·(A XOR B)
```

**Ripple-carry adder (RCA)** — chain N full adders:

```
  A3 B3    A2 B2    A1 B1    A0 B0
   │  │     │  │     │  │     │  │
  ┌┴──┴┐   ┌┴──┴┐   ┌┴──┴┐   ┌┴──┴┐
  │ FA │◄──│ FA │◄──│ FA │◄──│ FA │◄── Cin=0
  └────┘   └────┘   └────┘   └────┘
  Cout S3     S2       S1       S0

Delay: N × t_FA   (linear in word width)
Hypothetical 32-bit carry chain: 32 × 0.5 ns = 16 ns, exceeding a 1 ns budget
```

**Carry-Lookahead Adder (CLA)** — compute all carries in parallel:

```
Generate: Gi = Ai · Bi        (this bit produces a carry regardless of Cin)
Propagate: Pi = Ai ⊕ Bi      (this bit passes Cin to Cout)

C1 = G0 + P0·C0
C2 = G1 + P1·G0 + P1·P0·C0
C3 = G2 + P2·G1 + P2·P1·G0 + P2·P1·P0·C0
C4 = G3 + P3·G2 + P3·P2·G1 + P3·P2·P1·G0 + P3·P2·P1·P0·C0

The written sum-of-products has two logic levels after G/P generation if arbitrary fan-in gates are assumed. Real fan-in, wiring, buffering, and target carry primitives determine delay; hierarchical carry logic avoids an unbounded gate fan-in assumption.
```

**Delay comparison:**

| Adder type          | Delay (N-bit)   | Area      |
|---------------------|-----------------|-----------|
| Ripple-carry        | O(N)            | O(N)      |
| Carry-lookahead     | O(log N)        | O(N log N)|
| Carry-select        | O(√N)           | O(N)      |
| Kogge-Stone (tree)  | O(log N)        | O(N log N)|

Prefix structures such as Kogge–Stone and Han–Carlson trade logic depth, fan-out, area, and wiring. An FPGA’s dedicated carry chain can outperform a hand-written textbook lookahead network; compare mapped implementations on the same target.

### 3.3 Multipliers and MAC Units

**Array multiplier (4×4 unsigned):**

```
           A3  A2  A1  A0
         ×  B3  B2  B1  B0
         ─────────────────
         A3B0 A2B0 A1B0 A0B0     ← partial product row 0 (AND gates)
    A3B1 A2B1 A1B1 A0B1          ← partial product row 1 (shift 1)
A3B2 A2B2 A1B2 A0B2              ← partial product row 2 (shift 2)
A3B3 A2B3 A1B3 A0B3              ← partial product row 3 (shift 3)
─────────────────────────────────
P7   P6   P5   P4   P3  P2  P1  P0

Partial products are summed with an adder tree (Wallace tree reduces rows in O(log N) delay).
```

**MAC (Multiply-Accumulate):**

```
ACC ← ACC + (A × B)

  A ─┐
     ├── MULTIPLIER ─┐
  B ─┘               ├── ADDER ── ACC ──┐
                      │                  │
             ACC ─────┘                  │
              ↑                          │
              └──────────────────────────┘

One MAC = one ×, one +, one register write
```

The MAC is the fundamental operation in matrix multiplication. A systolic array tiles N×N MACs to compute a full matrix product:

```
A[i,k] flows right →      B[k,j] flows down ↓

Each PE: acc += A × B, then pass A right and B down.
With one MAC per cycle and skewed inputs, PE[i,j] receives its first pair at zero-based cycle i+j. For a reduction of K terms, its final MAC is at cycle K−1+i+j. A full M×N tile takes K+M+N−2 cycles through the last MAC; output drain depends on the interface.
```

### 3.4 ALU Design

A minimal ALU selects among operations via an opcode:

```
                  ┌──────────────────────────────────────────────┐
  A[31:0] ─────►  │  ADD   SUB   AND   OR   XOR   SLT   SHR   SHL │
  B[31:0] ─────►  │                                                │
                  └──────────────────┬───────────────────────────┘
                                     │ 8:1 MUX
                         OpCode[2:0] ┘
                                     ▼
                                 Result[31:0]
                                 + Flags: Zero, Carry, Overflow, Negative
```

**Flag bits:**

| Flag     | Meaning                              | Use                         |
|----------|--------------------------------------|-----------------------------|
| Zero (Z) | Result == 0                          | Branch equal/not-equal      |
| Carry (C)| Unsigned overflow / borrow           | Multi-word arithmetic       |
| Overflow (V) | Signed overflow                  | Signed arithmetic check     |
| Negative (N) | Result MSB is 1 (negative in 2's C) | Signed comparisons     |

### 3.5 Propagation delay and pipeline accounting

A register-to-register path includes source clock-to-Q, combinational delay, destination setup, and clock timing. Gate count alone is not a routed timing report. A 32-bit adder partitioned into 4-bit blocks has **eight** blocks; using four in a delay calculation accidentally models only 16 bits.

Use this hypothetical timing exercise with zero skew: logic stages take 3 ns, 4 ns, and 2 ns; source clock-to-Q plus destination setup and the selected timing margin total 0.2 ns per registered path.

| Implementation | Minimum period in this model | Latency | Steady throughput |
|---|---|---|---|
| One registered combinational path | `3+4+2+0.2 = 9.2 ns` | 1 cycle = 9.2 ns | 1 result / 9.2 ns |
| Three registered stages | `max(3,4,2)+0.2 = 4.2 ns` | 3 cycles = 12.6 ns | 1 result / 4.2 ns |

Latency increases by `12.6/9.2 ≈ 1.37×`, while steady throughput increases by `9.2/4.2 ≈ 2.19×`. Three stages does **not** imply three times the original time. Ten independent items take `10 × 9.2 = 92 ns` in the first design and `(10+3−1) × 4.2 = 50.4 ns` in the ideal pipeline, including fill but no stalls.

**Check:** one isolated item benefits from the first design’s lower latency. A continuous stream benefits from the second design’s throughput. If an output stalls, the interface must hold data correctly; use the [ready/valid RTL lab](../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md) to test that contract.

---

## 4. Sequential Logic

### 4.1 D Flip-Flop

The D flip-flop (DFF) is the canonical memory element. It captures D on the rising clock edge.

```
       D ──┤D    Q├── Q
    CLK ──┤CLK   │
           │    Q̄├── Q̄ (complement)
           └──────┘

Timing:
                ___     ___     ___
  CLK:      ___|   |___|   |___|   |___
                     ↑           ↑       ← capture edges
  D:    ──[stable]──[stable]──[stable]──
  Q:    ──────────[D₀]─────[D₁]────────
                     ^ after t_clk-to-q
```

**Critical timing parameters:**

| Parameter      | Symbol    | Definition                                    |
|----------------|-----------|-----------------------------------------------|
| Setup time     | t_su      | D must be stable this long BEFORE clock edge  |
| Hold time      | t_h       | D must be stable this long AFTER clock edge   |
| Clock-to-Q     | t_cq      | Time from clock edge to Q valid               |
| Propagation    | t_p       | Combinational logic delay between DFFs        |

**Setup/hold violation:** a transition close to the sampling edge can produce metastability or capture the old/new value unpredictably. Metastability is a possible analog outcome, not a guaranteed visible `X` in an RTL simulation.

```
  CLK:    _____|‾‾‾|_____
  D:      ────[old]X[new]    ← X = transition inside setup window
  Q:      ──────────[???]    ← metastable! may settle to wrong value
```

**Minimum period constraint:**  t_clk ≥ t_cq + t_p + t_su

### 4.2 Registers, Shift Registers, and LFSR

**N-bit register** — N DFFs sharing a clock and optional load-enable:

```
  D[N-1:0] ──► [DFF] [DFF] ... [DFF] ──► Q[N-1:0]
                CLK shared across all
```

**Shift register** — DFFs chained; data shifts one position per cycle:

```
  D_in → [DFF₀] → [DFF₁] → [DFF₂] → [DFF₃] → D_out
           Q₀       Q₁       Q₂       Q₃

  Use: serial-to-parallel conversion, FIFO, delay line, CRC computation
```

**LFSR (Linear Feedback Shift Register)** — XOR taps from specific positions feed back to input:

```
  Fibonacci LFSR, polynomial x⁴ + x³ + 1 (taps at positions 4 and 3):

  ┌─── XOR ◄──────────────────────────────┐
  │    ↑                                   │
  │  [DFF₄] ← [DFF₃] ← [DFF₂] ← [DFF₁]  │
  │    Q₄       Q₃       Q₂       Q₁      │
  │                                        │
  └──────────────────────────────── Q₄ → output

  Period: 2⁴ - 1 = 15 (maximal length for this polynomial)
  Explicit left-shift recurrence: q_next = {q[2:0], q[3] XOR q[2]}
  From 1111: 1110, 1100, 1000, 0001, 0010, 0100, 1001, ...
```

For that exact recurrence, a nonzero seed cycles through all 15 nonzero states; zero remains zero. Shift direction, bit numbering, polynomial convention, and XOR/XNOR choice must agree. The CPU lab checks the period. LFSRs are useful for test patterns and scramblers, but are not cryptographic random generators.

### 4.3 Finite State Machines

**Two types:**

| Type   | Output depends on        | Outputs change    |
|--------|--------------------------|-------------------|
| Moore  | State only               | After state changes; combinational decode can still glitch |
| Mealy  | State AND current inputs | After input/state propagation delay |

**Traffic light FSM (Moore) — 4 states:**

```
States: GREEN_NS, YELLOW_NS, GREEN_EW, YELLOW_EW
Input:  timer_expired (1 = time to change)

         timer_expired
GREEN_NS ──────────────► YELLOW_NS
   ↑                          │
   │            timer_expired │
   │                          ▼
YELLOW_EW ◄──────── GREEN_EW
          timer_expired
```

**HDL-style excerpt:** declarations, timer, and reset synchronization are omitted. This illustrates state transition logic, not a complete road traffic controller; a real design needs its specified all-red intervals and fault behavior.

```verilog
// State register — always sequential
always @(posedge clk or posedge rst) begin
    if (rst)  state <= GREEN_NS;
    else      state <= next_state;
end

// Next-state logic — always combinational
always @(*) begin
    case (state)
        GREEN_NS:  next_state = timer_expired ? YELLOW_NS : GREEN_NS;
        YELLOW_NS: next_state = timer_expired ? GREEN_EW : YELLOW_NS;
        GREEN_EW:  next_state = timer_expired ? YELLOW_EW : GREEN_EW;
        YELLOW_EW: next_state = timer_expired ? GREEN_NS : YELLOW_EW;
        default:   next_state = GREEN_NS;
    endcase
end

// Output logic — Moore: outputs depend on state only
assign light_ns = (state == GREEN_NS)  ? GREEN  :
                  (state == YELLOW_NS) ? YELLOW : RED;
assign light_ew = (state == GREEN_EW)  ? GREEN  :
                  (state == YELLOW_EW) ? YELLOW : RED;
```

The asynchronous reset shown above requires a defined assertion/deassertion contract. Synchronize deassertion in each destination clock domain when using this style. A two-flop synchronizer can reduce metastability propagation for a suitable single-bit level; it does not make an arbitrary multibit bus coherent or guarantee capture of a short pulse. Use a handshake or asynchronous FIFO for those transfer contracts. [AMD reset guidance](https://docs.amd.com/r/2023.1-English/ug1387-acap-hardware-ip-platform-dev-methodology/Synchronous-Reset-vs.-Asynchronous-Reset).

**State encoding strategies:**

| Encoding   | Bits needed | Power | Speed     | Use case                        |
|------------|-------------|-------|-----------|---------------------------------|
| Binary     | ceil(log₂ N) | Depends on switching/decode | Target-dependent | Compact state representation |
| One-hot    | N | More registers, often simpler decode | Target-dependent | Useful candidate for FPGA comparison |
| Gray code  | ceil(log₂ N) | One bit changes only on defined adjacent transitions | Target-dependent | Counters and carefully specified crossings |

> **AI hardware connection:** schedulers track states such as ready, waiting for data, and waiting for dependencies. Such state/control ideas transfer to accelerator design, but a modern SM is not described completely by one textbook FSM; limits and implementation vary by architecture.

### 4.4 Timing Analysis and Critical Path

**Setup analysis** — find the critical path across all register-to-register paths:

```
  REG A → [combo logic, delay t₁] → REG B

  Define skew = destination-clock arrival − source-clock arrival.
  Setup: t_cq,max + t_combo,max + t_su ≤ T + skew − setup_uncertainty
```

**Hold analysis** — ensure data doesn't change too fast:

```
  Hold: t_cq,min + t_combo,min ≥ t_h + skew + hold_uncertainty
  (new data must not reach the destination too soon)
```

**Clock skew** — the difference in clock arrival time at two flip-flops:

```
   CLK source
      │
   ┌──┴──┐
   │     │   ← routing delays differ
 REG A  REG B
  CLK   CLK
  t₁    t₂     skew = t₂ - t₁

  Positive skew (t₂ > t₁): relaxes setup, tightens hold
  Negative skew (t₂ < t₁): tightens setup, relaxes hold
```

**Clock gating** — disable the clock to a register bank when idle to save dynamic power:

```
  CLK ───┐
          ├── AND ──► gated CLK → register bank
  EN  ───┘

  Dynamic power scales approximately as activity × switched_capacitance × V² × f
  Energy per switched cycle scales as capacitance × V²
  An ordinary logic AND is a conceptual sketch, not a recommended FPGA clock gate.
  Use a characterized glitch-free gate/clock primitive and its timing rules.
```

---

## 5. Memory Technologies

### 5.1 SRAM — Static Random Access Memory

SRAM stores a bit in a cross-coupled inverter pair (the **6T cell**):

```
        VDD
         │
    ─────┼──────────────────────────────
    │    PMOS₁                PMOS₂   │
    │      │                    │     │
    │      ├── Q ─────── Q̄ ───┤      │
    │    NMOS₁       ┌───┐  NMOS₂    │
    │      │         │INV│    │      │
    │      │         └───┘    │      │
    ─────NMOS₃            NMOS₄──────
    │      │                    │    │
    BL     WL─────────────────WL    BL̄

6T SRAM cell:
  - 2 PMOS (pull-up)
  - 2 NMOS (pull-down) forming the latch
  - 2 NMOS access transistors (NMOS₃, NMOS₄) controlled by Word Line
```

**Operation:**

| Phase     | WL  | BL  | BL̄ | Action                                     |
|-----------|-----|-----|----|--------------------------------------------|
| Standby   | 0   | Pre | Pre| Latch holds state; access transistors off  |
| Read      | 1   | Pre | Pre| Cell discharges one BL; sense amp detects  |
| Write     | 1   | D   | D̄  | Driver overrides latch; cell flips         |

**SRAM characteristics:**

| Property            | Value                                    |
|---------------------|------------------------------------------|
| Access time | Depends on array size, ports, process, and surrounding logic |
| Density | Cell and peripheral area depend on the process and macro |
| Power               | Static leakage + dynamic read/write      |
| Retention           | Holds data as long as VDD applied        |
| On-chip use         | Registers, caches, scratchpad, FIFOs     |

### 5.2 DRAM — Dynamic RAM

DRAM stores charge on a capacitor — one transistor, one capacitor (1T1C):

```
       BL
        │
       NMOS  ← access transistor (WL controls)
        │
       CAP   ← stored charge; capacitance is process-dependent
        │
       GND

Reading shares cell charge with a precharged bit line.
A sense amplifier resolves the small voltage difference and restores the cell.
```

**Key differences from SRAM:**

| Property      | SRAM                    | DRAM                        |
|---------------|-------------------------|-----------------------------|
| Cell size     | 6 transistors           | 1 transistor + 1 capacitor  |
| Density | Larger bit cell in a comparable process | Compact cell plus specialized capacitor/process |
| Access behavior | No activate/refresh sequence | Row hits, activation, precharge, and refresh affect timing |
| Refresh | No periodic refresh while powered | Required interval depends on the part, temperature, and mode |
| Destructive read | No                   | Yes (must restore after read) |
| Economics | Area/ports/process tradeoffs | High-capacity commodity and specialized products |

**DRAM access sequence:**

```
1. Activate: open a row into sense amplifiers; observe the part’s tRCD before access.
2. Read/write: select columns; read latency and burst timing follow the device mode.
3. Precharge: close row, precharge bit lines
4. Refresh: periodically rewrite all rows before charge leaks away

Peak payload bandwidth = data_width_bits × transfers_per_second / 8
DDR5-6400 aggregate 64 data bits: 64 × 6.4e9 / 8 = 51.2e9 B/s = 51.2 GB/s
One 32-bit subchannel: 25.6 GB/s. Do not multiply by burst length again.
```

### 5.3 HBM — High Bandwidth Memory

HBM stacks DRAM dies vertically on silicon interposer, connected via Through-Silicon Vias (TSVs):

```
  GPU Die ──────────────────────────── HBM stack
              Silicon interposer

  HBM stack cross-section:

    ┌─────────────┐   ← DRAM die 4
    ├─────────────┤   ← DRAM die 3
    ├─────────────┤   ← DRAM die 2
    ├─────────────┤   ← DRAM die 1
    └─────────────┘   ← Base die (logic)
           │││         ← TSVs (Through-Silicon Vias)
    ─────────────────  ← Silicon interposer
           │││
    ┌─────────────┐   ← GPU/accelerator die
```

Memory comparisons must use the same boundary: one chip, one stack, one channel, one package, or the complete board. A DDR5 DIMM is not one unlimited-capacity “stack,” and a GPU board’s GDDR capacity is not one chip’s capacity.

| Organization | Typical physical relationship | What to obtain from its exact specification |
|---|---|---|
| DDR interface | Memory devices connected through channels/subchannels | Aggregate data width, transfer rate, rank/channel population |
| GDDR interface | Several memory packages around a processor | Width/rate per device and populated device count |
| HBM interface | Stacked DRAM near the processor in an advanced package | Active stacks, channels, per-pin rate, usable capacity |

For a purely hypothetical 1,024-bit interface at 6.4 Gbit/s per data pin, peak bandwidth is `1024 × 6.4e9 / 8 = 819.2 GB/s`. Four such independent interfaces provide 3,276.8 GB/s before utilization losses. This is arithmetic with stated inputs, not the specification of an unnamed HBM product.

For DDR5, distinguish the 6,400 MT/s data rate from its 3,200 MHz clock and the two 32-bit data subchannels of a conventional 64-data-bit DIMM. Bursts group transfers; they do not multiply an already specified transfers-per-second rate. [Micron DDR5 architecture](https://assets.micron.com/adobe/assets/urn%3Aaaid%3Aaem%3Af1706cdb-4b55-4568-84bc-b7ee42f46236/original/as/ddr5-more-than-a-generational-update-wp.pdf).

Which interface limits an AI workload depends on its data reuse, tensor shapes, cache behavior, precision, compute path, and achieved bandwidth. Neither HBM nor PCIe is universally the bottleneck.

### 5.4 Flash Memory

Flash stores charge in a floating gate or charge trap layer — no power needed for retention.

```
NAND Flash cell (floating gate):

    Control gate (CG)
         │
    ─────┼────   ← tunnel oxide
    Floating gate (FG) ← charge stored here
    ─────┼────   ← gate oxide
    Source           Drain
       │                │
       └─────── ────────┘
                Channel

Writing: tunnel electrons onto FG (Fowler-Nordheim tunneling) → raises Vt → reads as 0
Erasing: remove electrons from FG → lowers Vt → reads as 1
```

**Flash cell types by bits per cell:**

| Type | Bits per cell | Nominal distinguishable levels |
|---|---|---|
| SLC | 1 | 2 |
| MLC in the common two-bit usage | 2 | 4 |
| TLC | 3 | 8 |
| QLC | 4 | 16 |

More levels require distinguishing smaller voltage intervals and influence programming, retention, and error management. Endurance is a property of the actual part, operating conditions, and controller policy; do not substitute a universal cycle count for its datasheet or drive write-endurance rating.

For NAND, programming and erasure have different granularities: the part specifies page/program units and erase blocks. NOR and managed storage interfaces have different contracts. An SSD’s FTL maps logical writes onto its NAND while handling erase, wear, and relocation; host sectors are not a direct map of raw NAND pages.

### 5.5 Cache Hierarchy

A cache exploits spatial and temporal locality to bridge the speed gap between CPU/GPU registers and main memory.

| Resource | Why it is useful | What can limit it |
|---|---|---|
| Registers | Values close to arithmetic | Register capacity, ports, dependencies, occupancy |
| Cache | Reuses recently accessed lines automatically | Misses, conflicts, bandwidth, replacement |
| Scratchpad/shared memory | Software controls reuse and cooperation | Capacity, banks, synchronization |
| Main DRAM/HBM | Large working sets | Traffic, latency, controller scheduling |
| Host/device interconnect | Transfers between address/ownership domains | Link and software path, copies, synchronization |
| Storage | Persistent datasets and images | I/O size, queueing, filesystem, device latency |

A latency table needs the exact processor, operation, access pattern, hit/miss state, clock, and measurement method. These resources are not interchangeable rows in a universal H100/server timing table.

**Worked cache mapping:** a 1 KiB direct-mapped cache with 64-byte lines has 16 sets. Byte address `0x140` has block number `320/64=5`, set 5, tag 0, offset 0. Address `0x540` has block number 21, set 5, tag 1. Alternating those two lines causes conflict misses in this cache even though their combined data is only 128 bytes. Greater total capacity alone is not the only design variable.

**Cache organization:**

```
Direct-mapped (1-way):     Set-associative (4-way):
  Tag | Index | Offset       Tag | Index | Offset

  One slot per set          Four slots per set (4 ways)
  Fast, simple              Fewer conflict misses

  Index selects row         Index selects set, all 4 ways checked in parallel
  Tag must match            Any way whose tag matches → hit
```

**Replacement policies:**

| Policy | Evicts                            | Hardware cost |
|--------|-----------------------------------|---------------|
| Exact LRU | Least recently used way | At least ceil(log₂(W!)) bits to represent every recency ordering |
| PLRU   | Pseudo-LRU (tree approximation)   | W−1 bits per set |
| Random | Random way                        | LFSR counter  |
| FIFO   | Oldest loaded way                 | Pointer per set |

Replacement behavior is implementation-specific and may adapt to workload. Do not assume a vendor’s undisclosed policy from the cache size or associativity alone.

### 5.6 On-chip scratchpad and tiling

A cache can reuse data automatically; a scratchpad lets software place and share data explicitly. Tiling organizes computation so a loaded value contributes to several outputs before eviction.

For `C[M,N] = A[M,K] × B[K,N]`, counting one multiply and one add as two operations gives `2MNK` FLOPs. Assume FP32 operands, `C = AB` with no old C read, and a deliberately simple traffic model:

- A scalar output loop issues two operand loads per multiply: `8MNK` bytes plus `4MN` output bytes. These are **logical operand loads**, not necessarily HBM transactions: caches can serve repeated addresses.
- A `T×T` output tile reuses each loaded A/B value across T outputs. For one reduction chunk of length `Kc`, it performs `2T²Kc` FLOPs and loads `2TKc×4 = 8TKc` bytes. Ignoring output traffic for the moment, intensity is `T/4` FLOP/byte.
- If M and N divide evenly by T and tiles do not reuse one another’s inputs, traffic is `8MNK/T + 4MN` bytes. Edge tiles require their actual valid dimensions.

For a hypothetical machine with a relevant compute ceiling of **100 TFLOP/s** and memory bandwidth of **1 TB/s**, the roofline ridge is `100 FLOP/byte`. A tile of T=16 gives about 4 FLOP/byte; T=128 gives about 32. **Both remain below the ridge** in this model. At intensity 32, the bandwidth ceiling is about 32 TFLOP/s, not 100.

The bound is `min(compute_ceiling, bandwidth × intensity)`. It is not a performance prediction: instruction type, occupancy, register pressure, synchronization, cache reuse, and achieved bandwidth matter. Use a compute ceiling for the operations you actually execute, not an unrelated tensor precision or sparse peak. The [tested tiling/dataflow lab](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Labs/Tiling-and-Dataflow.md) calculates edge-tile traffic and wavefront cycles explicitly.

**Bank mapping example:** assume 32 banks, one 32-bit word per bank, and `bank = word_address mod 32`. Lane i reading word i uses different banks. Lane i reading word `2i` uses 16 banks twice: lanes 0 and 16 read different words in bank 0. That conflict can reduce throughput; it does not prove the whole kernel takes exactly twice as long. A supported same-word broadcast is different from different-word collisions.

For a warp reading a column of `float tile[32][32]`, row starts map to the same bank. Padding to `[32][33]` changes adjacent row starts by 33 words, spreading them over banks. Padding is tied to the access pattern, data width, and architecture; adding one element to every array is not a universal fix. [CUDA shared-memory access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks).

---

## 6. MIPS Processor Design

Building a small processor combines the components in sections 1–5. This section uses a classic MIPS-like 32-bit teaching subset; RISC-V is another common teaching choice. The five-stage design illustrates dependencies and forwarding. It does not describe every CPU/GPU pipeline or implement a complete architectural MIPS machine.

### 6.1 MIPS ISA Subset

A minimal core supports three instruction formats:

```
R-type (register):   [opcode 6][rs 5][rt 5][rd 5][shamt 5][funct 6]
I-type (immediate):  [opcode 6][rs 5][rt 5][imm 16]
J-type (jump):       [opcode 6][address 26]
```

**Key instructions for the datapath:**

| Instruction | Format | Operation                    | Example              |
|-------------|--------|------------------------------|----------------------|
| `add`       | R      | rd = rs + rt                 | `add $t0, $s1, $s2` |
| `sub`       | R      | rd = rs − rt                 | `sub $t0, $s1, $s2` |
| `and`       | R      | rd = rs & rt                 | `and $t0, $s1, $s2` |
| `or`        | R      | rd = rs \| rt                | `or  $t0, $s1, $s2` |
| `slt`       | R      | rd = (rs < rt) ? 1 : 0      | `slt $t0, $s1, $s2` |
| `lw`        | I      | rt = Mem[rs + imm]           | `lw  $t0, 4($s1)`   |
| `sw`        | I      | Mem[rs + imm] = rt           | `sw  $t0, 4($s1)`   |
| `beq`       | I      | if equal: PC = PC+4+(sign_ext(imm)<<2) | `beq $s1, $s2, L1`  |
| `addi`      | I      | rt = rs + sign_ext(imm)      | `addi $t0, $s1, 10` |
| `j`         | J      | PC = {(PC+4)[31:28], addr, 2'b0} | `j   target`        |

This teaching subset omits exceptions, privileged state, and architectural delay slots. Classic MIPS `add`/`addi` can trap on signed overflow; if the lab wraps instead, name that difference or use the corresponding nontrapping operation. Loads/stores must specify alignment and byte order.

**Register file — 32 registers, $0 hardwired to zero:**

```
$zero ($0)  = always 0          $t0-$t9  = temporaries
$at   ($1)  = assembler temp    $s0-$s7  = saved (callee-preserved)
$v0-$v1     = return values     $sp ($29)= stack pointer
$a0-$a3     = arguments         $ra ($31)= return address
```

### 6.2 Single-Cycle Datapath

Every instruction completes in one clock cycle. The datapath connects the building blocks from sections 3–5:

```
                            ┌────────────────────────────────────────────────────┐
                            │                    Control Unit                    │
                            │  (opcode, funct → RegDst, ALUSrc, MemtoReg,       │
                            │   RegWrite, MemRead, MemWrite, Branch, ALUOp, Jump)│
                            └──────┬──────┬──────┬──────┬──────┬──────┬─────────┘
                                   │      │      │      │      │      │
  ┌──────┐    ┌──────────┐   ┌─────┴──┐  MUX   MUX    │      │      │
  │      │    │ Register │   │        │   │     │      │      │      │
  │  PC  ├──►│  File    ├──►│  ALU   ├───┘     │    ┌─┴──┐   │      │
  │      │    │ (32×32)  │   │        │         └───►│Data│   │      │
  └──┬───┘    │          │   └────────┘              │Mem │   │      │
     │        │  rd1─────┼───► A                     │    │   │      │
     │        │  rd2─────┼───► B / MUX ◄── imm       │    │   │      │
     ▼        └──────────┘                           └────┘   │      │
 ┌──────┐                                                     │      │
 │Instr │                                                     │      │
 │ Mem  │◄────────────────────────────────────────────────────┘      │
 └──────┘                                                            │
     │                                                               │
     └─── PC+4 / branch target / jump target ◄──────────────────────┘
```

**Datapath walk-through for `lw $t0, 8($s1)`:**

```
Cycle:
  1. Fetch:     Instr Mem[PC] → instruction word
  2. Decode:    rs=$s1(17), rt=$t0(8), imm=8
                Register File reads: rd1 = Reg[17] = value of $s1
  3. Execute:   ALU computes: $s1 + sign_ext(8) = address
  4. Memory:    Data Mem[address] → loaded word
  5. Writeback: MUX selects memory output → Reg[8] ($t0)
  6. PC update: PC ← PC + 4

All in ONE cycle. Clock period = longest path (lw: through Instr Mem + Reg File + ALU + Data Mem + MUX).
```

**Control signals for each instruction type:**

| Signal    | R-type | `lw`  | `sw`  | `beq` | `addi`| `j`   |
|-----------|--------|-------|-------|-------|-------|-------|
| RegDst    | 1 (rd) | 0 (rt)| X     | X     | 0 (rt)| X     |
| ALUSrc    | 0 (rt) | 1 (imm)| 1 (imm)| 0 (rt)| 1 (imm)| X  |
| MemtoReg  | 0 (ALU)| 1 (Mem)| X     | X     | 0 (ALU)| X   |
| RegWrite  | 1      | 1     | 0     | 0     | 1     | 0     |
| MemRead   | 0      | 1     | 0     | 0     | 0     | 0     |
| MemWrite  | 0      | 0     | 1     | 0     | 0     | 0     |
| Branch    | 0      | 0     | 0     | 1     | 0     | 0     |
| Jump      | 0      | 0     | 0     | 0     | 0     | 1     |
| ALUOp     | 10     | 00    | 00    | 01    | 00    | XX    |

**ALU control** — two-level decode:

```
ALUOp (from main control) + funct field (from instruction) → ALU operation

ALUOp=00 → ADD  (lw, sw, addi: address calculation)
ALUOp=01 → SUB  (beq: compare by subtraction)
ALUOp=10 → look at funct:
             funct=100000 → ADD
             funct=100010 → SUB
             funct=100100 → AND
             funct=100101 → OR
             funct=101010 → SLT
```

### 6.3 Single-Cycle Timing Problem

The single-cycle design's clock period equals the slowest instruction:

```
lw critical path:
  Instr Mem  +  Reg Read  +  ALU  +  Data Mem  +  Reg Write MUX
    200ps    +   150ps    + 200ps +   200ps    +     25ps
  = 775ps  →  max freq ≈ 1.29 GHz

With these illustrative blocks, add uses 200+150+200+25 = 575 ps.
It still occupies the 775 ps single-cycle period: 200 ps of unused timing budget.
```

This wastes significant time on fast instructions. The solution: pipelining.

### 6.4 Five-Stage Pipeline

Break the datapath into 5 stages, each taking one clock cycle. Pipeline registers separate stages:

```
Stage 1       Stage 2        Stage 3       Stage 4        Stage 5
  IF             ID             EX            MEM             WB
┌──────┐    ┌──────────┐   ┌────────┐    ┌────────┐    ┌──────────┐
│Instr │    │ Register │   │        │    │  Data  │    │  Write   │
│Fetch │ ►REG► Decode  │►REG► ALU  │►REG►Memory │►REG► Back    │
│      │    │ + Read   │   │        │    │        │    │          │
└──────┘    └──────────┘   └────────┘    └────────┘    └──────────┘
  PC+4       rs, rt, imm   ALU result   Mem data       rd ← result
  instr      control sigs  branch addr

Illustrative clock period = max(stage delay) + register overhead
             = 200ps + 20ps = 220ps  →  max freq ≈ 4.5 GHz
Throughput = 1 instruction per 220ps (vs 775ps single-cycle → 3.5× faster)
Latency = 5 × 220ps = 1100ps per instruction (worse, but throughput wins)
```

**Pipeline timing diagram — 5 instructions flowing through:**

```
Time (cycles):  1     2     3     4     5     6     7     8     9
Instr 1:       IF    ID    EX    MEM   WB
Instr 2:             IF    ID    EX    MEM   WB
Instr 3:                   IF    ID    EX    MEM   WB
Instr 4:                         IF    ID    EX    MEM   WB
Instr 5:                               IF    ID    EX    MEM   WB

After cycle 5, one instruction completes EVERY cycle.
Steady-state CPI = 1 (Cycles Per Instruction).
```

### 6.5 Pipeline Hazards

Three types of hazards break the ideal CPI = 1:

#### Data Hazards

A later instruction reads a register that an earlier instruction hasn't written yet:

```
add $s0, $t0, $t1    # WB writes $s0 in cycle 5
sub $t2, $s0, $t3    # ID reads $s0 in cycle 3 — STALE value!

Time:      1     2     3     4     5
add:      IF    ID    EX    MEM   WB ← $s0 written here
sub:            IF    ID ←reads $s0 here (old value!)
```

**Solution 1 — Forwarding (bypassing):** route the ALU result directly back to the ALU input without waiting for WB:

```
                       ┌──────── forward path ────────┐
                       │                               │
add:   IF    ID    EX ─┤─ MEM    WB                    │
sub:         IF    ID  ─┘─ EX ◄──┘  MEM    WB
                           ↑
                    ALU gets $s0 from EX/MEM register
                    instead of register file

Forwarding unit checks: if (EX/MEM.rd == ID/EX.rs) → forward EX/MEM.ALUresult
```

**Solution 2 — Load-use hazard (forwarding can't fix):**

```
lw  $s0, 0($t0)    # data available after MEM (cycle 4)
add $t2, $s0, $t1  # EX would need $s0 at the start of cycle 4 — before load data is ready!

Must stall (insert bubble) for 1 cycle:

Time:      1     2     3     4     5     6     7
lw:       IF    ID    EX    MEM   WB
add:            IF    ID   stall  EX    MEM    WB
                       ↑          ↑
                   hazard      forward from MEM/WB register
                  detected
```

This particular interlocked pipeline inserts one stall. Instruction scheduling can place independent work between the load and consumer. That is different from an ISA-defined architectural load delay slot.

#### Control Hazards

Branch outcome isn't known until EX (or MEM), but the next instruction is already fetched:

```
beq $s0, $s1, target   # branch resolved in EX (cycle 3)
add ...                 # fetched in cycle 2 — might be wrong path!
or  ...                 # fetched in cycle 3 — might be wrong path!
```

**Solutions:**

| Strategy           | Penalty  | Hardware cost | Description                          |
|--------------------|----------|---------------|--------------------------------------|
| Stall or flush | Depends on resolution stage | Control logic | Wait before issue, or invalidate already-issued wrong-path work |
| Branch prediction  | 0 if correct | Moderate  | Predict taken/not-taken              |
| Architectural delay slot | Useful work only if a legal instruction fills it | ISA/compiler constraint | A property of some historical MIPS versions, not this simplified no-delay-slot core |
| Early branch resolution | 1 cycle | MUX in ID | Move comparison to ID stage          |

**Static prediction:** predict not-taken. In this five-stage model with resolution in EX, a taken branch generally invalidates the two younger instructions in ID and IF. The penalty changes if resolution or fetching changes.

**Dynamic prediction (branch history table):**

```
1-bit predictor: remember last outcome → predict same
  A repeatedly invoked multi-iteration loop can mispredict its first taken branch after a previous not-taken exit, then mispredict the exit. Initial state and loop trip count matter.

2-bit saturating counter:
  States: Strongly Taken (11) → Weakly Taken (10) → Weakly Not (01) → Strongly Not (00)
  From a strongly predicted state, two opposite outcomes change the prediction; from a weak state, one does. Update the counter on every observed branch outcome.
```

#### Structural Hazards

Two stages need the same hardware unit in the same cycle (e.g., single memory for both IF and MEM):

```
Solution: use separate instruction memory and data memory (Harvard architecture)
          or use a multi-ported cache
```

The classic MIPS pipeline uses separate instruction and data memories — no structural hazards in the base design.

### 6.6 Complete Pipelined Datapath

```
 ┌─────┐  IF/ID  ┌─────┐  ID/EX  ┌─────┐ EX/MEM ┌─────┐ MEM/WB ┌─────┐
 │     │  reg    │     │  reg    │     │  reg   │     │  reg   │     │
 │ IF  ├────────►│ ID  ├────────►│ EX  ├───────►│ MEM ├───────►│ WB  │
 │     │         │     │         │     │        │     │        │     │
 └──┬──┘         └──┬──┘         └──┬──┘        └─────┘        └──┬──┘
    │               │               │                              │
    │            ┌──┴──┐         ┌──┴──┐                           │
    │            │Hazard│        │Fwd  │                           │
    │            │Detect│        │Unit │◄──────────────────────────┘
    │            └──┬──┘        └─────┘         (forward paths)
    │               │
    │          stall/flush
    │
  ┌─┴──────────────────────────┐
  │   PC MUX                    │
  │   (PC+4 / branch / jump)   │
  └────────────────────────────┘

Pipeline registers store ALL signals needed by later stages:
  IF/ID:  instruction, PC+4
  ID/EX:  control signals, rs data, rt data, sign-ext imm, rd/rt dest
  EX/MEM: control signals, ALU result, rt data (for sw), dest register
  MEM/WB: control signals, ALU result OR memory data, dest register
```

**Forwarding unit logic (simplified):**

```
// Forward an available ALU result from EX/MEM (not a load address).
if (EX/MEM.RegWrite && !EX/MEM.MemRead && EX/MEM.rd != 0 && EX/MEM.rd == ID/EX.rs)
    ForwardA = EX/MEM.ALUresult

// Forward from MEM/WB stage
if (MEM/WB.RegWrite && MEM/WB.rd != 0 && MEM/WB.rd == ID/EX.rs
    && !(EX/MEM.RegWrite && EX/MEM.rd == ID/EX.rs))   // EX/MEM has priority
    ForwardA = MEM/WB.WriteData

// Same logic for ForwardB (rt input)
```

**Hazard detection unit (load-use stall):**

```
if (ID/EX.MemRead && ID/EX.rt != 0
    && ((decode_reads_rs && ID/EX.rt == IF/ID.rs)
        || (decode_reads_rt && ID/EX.rt == IF/ID.rt)))
    stall pipeline:
      - IF/ID register holds (re-decode same instruction)
      - insert NOP into ID/EX (bubble)
      - PC holds (do not advance the fetch address)
```

### 6.7 Performance Metrics

```
Execution time = IC × CPI × T_clk

  IC    = Instruction Count (depends on ISA and compiler)
  CPI   = Cycles Per Instruction (depends on pipeline + hazards)
  T_clk = Clock period (depends on critical path in longest stage)

Single-cycle:  CPI = 1,  T_clk = 775ps    →  time/instr = 775ps
Pipelined:     CPI ≈ 1.2 (stalls + flushes), T_clk = 220ps
               →  time/instr = 264ps  →  ~2.9× speedup

N is an ideal upper comparison for equally balanced stages with negligible overhead and no stalls.
For the specified unbalanced 775 ps/220 ps example, the steady-state limit is about 3.52×.
```

**Synthetic CPI accounting example (not a compiler measurement):**

| Component      | CPI contribution |
|----------------|-----------------|
| Base            | 1.0            |
| Load-use stalls | +0.05         |
| Branch mispredicts | +0.12      |
| Cache misses    | +0.07         |
| **Total CPI**   | **1.24**      |

> **AI hardware connection:** a GPU scheduler can issue eligible work from another resident warp while one waits on a dependency. This hides some latency if enough independent ready work exists; it does not remove the waiting operation or guarantee full utilization. Avoid assigning a universal pipeline depth to all GPU operations.

---

## Resources

| Resource | Type | Focus |
|----------|------|-------|
| *Digital Design and Computer Architecture* — Harris & Harris | Textbook | Gates through ISA, HDL examples |
| *Computer Organization and Design* — Patterson & Hennessy | Textbook | RISC-V, memory hierarchy, pipelining |
| *Digital Design and Computer Architecture: ARM Edition* — Harris & Harris | Textbook | Single-cycle + pipelined processor, HDL |
| *Modern VLSI Design* — Wolf | Textbook | CMOS, timing, place-and-route |
| *CMOS VLSI Design* — Weste & Harris | Textbook | Transistor-level design, layout, fabrication |
| Sam Zeloof (YouTube) | Video | DIY semiconductor fab — see the process hands-on |
| Chips and Cheese (chipsandcheese.com) | Blog | Deep dives into real chip microarchitectures |
| Nandland (nandland.com) | Online | FPGA/Verilog/VHDL hands-on |
| EEVblog / Ben Eater | Video | Intuition for digital circuits, breadboard CPU |
| JEDEC JESD79-5B (DDR5 spec) | Spec | DRAM timing parameters |
| JEDEC HBM3 spec | Spec | HBM architecture and interface |
| IEEE 754-2019 | Standard | Floating-point standard |

---

## Projects

| Project | Skills | Outcome |
|---------|--------|---------|
| **Binary/BCD/Gray code converter** | Number systems, K-map | Combinational logic in Verilog |
| **32-bit CLA adder in Verilog** | Generate/propagate, prefix tree | RTL design + timing analysis |
| **IEEE 754 FP32 adder** | Alignment shift, normalization, rounding | Understand FP hardware |
| **INT8 MAC unit** | Fixed-point, two's complement, accumulation | Core of quantized inference |
| **LFSR-based PRNG** | Shift registers, XOR feedback, polynomial | BIST, test stimulus generation |
| **4-state FSM (Moore + Mealy)** | State encoding, next-state logic | Control unit design pattern |
| **Tiled matrix multiply (CUDA)** | Shared memory, bank conflict, tiling | Connect theory to GPU programming |
| **Single-cycle MIPS in Verilog** | Datapath, control unit, ALU, register file | Complete processor from gates to ISA |
| **Pipelined MIPS in Verilog** | Pipeline registers, forwarding, hazard detection | Understand CPI, stalls, bypassing |
| **Cache simulator (Python/C++)** | Direct-map vs set-assoc, LRU, miss rates | Memory hierarchy intuition |
| **RTL-to-GDS with OpenLane** | Synthesis, PnR, DRC/LVS, SkyWater PDK | Experience the full tapeout flow (open-source) |

## Derive the integer datapath before coding it

For signed INT8 inputs, the largest positive product is `(-128)×(-128)=16384`. A length-1024 dot product can reach `2^24`, which does not fit a signed 25-bit accumulator; 26 bits are sufficient for this bound. Verify that calculation for your chosen input range, bias, and partial-sum schedule. Define whether narrowing rounds, wraps, or saturates.

Write a short reference that exercises zeros, extrema, cancellation, and a value halfway between output quantization steps. Compare truncation with your specified rounding rule; BF16 conversion is not universally bit truncation (see [CUDA BF16 functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html)). Hand these vectors to the HDL testbench before designing the multiplier.

**Carry forward:** Representable-range proof and golden arithmetic vectors. Continue with [the next course](../Hardware%20Description%20Languages%20%28HDLs%29/Guide.md) and reuse that evidence.
