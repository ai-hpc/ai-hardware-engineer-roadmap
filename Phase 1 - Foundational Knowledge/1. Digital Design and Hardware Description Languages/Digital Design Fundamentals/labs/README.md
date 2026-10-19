# Bits, errors, and honest hardware arithmetic

**Time:** 45–60 minutes. **Requirements:** Python 3.10+ and basic binary arithmetic. These are executed CPU reference calculations, not FPGA synthesis or measured memory/timing results.

Download [logic_and_costs.py](logic_and_costs.py) and [test_logic_and_costs.py](test_logic_and_costs.py) together, or enter this directory in the repository:

```bash
python3 logic_and_costs.py
python3 -m unittest discover -s . -p 'test_*.py' -v
```

Nine tests cover independent known answers, exhaustive error locations, the LFSR period, exact rational arithmetic, and rejected invalid inputs. The [validation record](validation.json) identifies the executed sources. Start by predicting one result below, then run the code and explain any difference.

## 1. Correct one bit; reject two

An extended Hamming(8,4) code protects four data bits with four check bits. This example numbers codeword positions from 1 through 8; position 1 is the least significant stored bit. Positions 1, 2, and 4 contain Hamming parity, and position 8 contains overall even parity.

Encode data `1010`, with the least significant data bit first:

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Role | p1 | p2 | d1 | p4 | d2 | d3 | d4 | overall |
| Value | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |

For example, p1 checks positions 1, 3, 5, and 7: the two data ones already have even parity, so p1 is zero. Positions 1–7 contain three ones, so position 8 must be one. Written in conventional most-significant-bit-first order, the complete codeword is **`11010010`, or `0xD2`**.

The decoder computes `syndrome = p1_check + 2*p2_check + 4*p4_check`, plus overall parity:

| Syndrome | Overall parity | Meaning under a maximum-two-error assumption |
|---|---|---|
| 0 | even | No detected error |
| nonzero | odd | One error in positions 1–7; syndrome gives its position |
| 0 | odd | One error in position 8 |
| nonzero | even | Two errors; reject rather than guess |

**Try:** flip position 5: `0xD2 ^ (1 << 4)`. The decoder returns data 10, `corrected`, position 5. Flip positions 3 and 5 together: it returns `double_error` with no data.

The tests check all 16 messages, all 128 message/single-error combinations, and all 448 message/double-error combinations. They also show the boundary of the promise: four bit flips can turn one valid codeword into a different valid codeword. SECDED is not authentication and does not detect every possible corruption. The overall parity is essential to distinguishing a single from a double error; a Hamming(7,4) decoder cannot promise both single-error correction and double-error identification.

## 2. Specify an LFSR before quoting its period

Our exact four-bit recurrence is:

```text
q_next = {q[2:0], q[3] XOR q[2]}
```

This means shift the low three bits left and insert the exclusive-OR of the old high two bits as the new low bit. From `1111`, the first four successors are `1110`, `1100`, `1000`, `0001`. All 15 nonzero states appear before returning to the seed. The all-zero state remains zero.

**Try:** change the feedback to XOR bits 3 and 1 in a copy. Run the period test and inspect which states repeat. A polynomial name without shift direction and bit numbering is an incomplete implementation contract. An LFSR’s predictable sequence can be useful test stimulus, but it is not cryptographically secure randomness.

## 3. Keep transfers, clocks, and bytes separate

Peak interface bandwidth is `data_bits × transfers_per_second / 8`. A 64-data-bit aggregate at 6,400 million transfers per second gives **51.2 GB/s**; a 32-bit subchannel gives **25.6 GB/s**. Decimal GB means one billion bytes here.

The transfer rate already counts both data edges for this DDR example. Burst length groups transfers and must not be multiplied into this formula again. Nor does peak interface bandwidth imply that an application achieves that rate: refresh, command timing, access pattern, contention, and software all affect utilization. See the [memory explanation](../Guide.md#52-dram-dynamic-ram) before applying the equation to a part’s datasheet.

**Check:** an invented 1,024-bit interface at 6.4 Gbit/s per data pin gives **819.2 GB/s**. The same formula works; the physical boundary and units must stay explicit.

## 4. Predict pipeline fill and latency

Suppose three logic sections take 3 ns, 4 ns, and 2 ns, and each registered path adds a total of 0.2 ns for clock-to-Q, setup, and the selected margin. Assume zero skew and no stalls.

Without internal pipeline registers, the period is **9.2 ns**. With three stages, it is **4.2 ns**, but a result traverses three cycles, taking **12.6 ns**. Ten results need 12 cycles including fill, or **50.4 ns**. The unpipelined version needs **92 ns** for ten results.

The code uses `Fraction` so these assumed numbers are computed exactly; that does not make the assumptions more accurate physically. The hardware still needs valid constraints and implementation timing analysis. Changing the stage count affects latency, throughput, fill, and interface state differently.

**Check:** for only one result, the three-stage design still takes 12.6 ns. Higher steady throughput does not imply lower latency.

## 5. Explain a cache conflict and a metadata cost

A 1 KiB direct-mapped cache with 64-byte lines has 16 sets. Addresses `0x140` and `0x540` have the same set index 5 but different tags, 0 and 1. Alternating between the two lines evicts the previous line, despite their small combined working set. This mapping calculation is not a complete cache simulator: it does not model coherence, prefetching, or write policy.

For exact LRU replacement in a W-way set, there are `W!` recency orderings. Representing every ordering needs at least `ceil(log2(W!))` bits. Four ways require at least five bits, since 24 orderings cannot fit into four bits. This is an information bound, not a claim that every practical circuit uses that minimum. A four-way tree pseudo-LRU needs three tree bits and approximates the replacement order.

**Completion:** submit the parity derivation, one corrected and one rejected codeword, the 15-state LFSR trace, and the bandwidth/pipeline/cache calculations. Explain one guarantee each model cannot establish. Continue with the [HDL guide](../../Hardware%20Description%20Languages%20%28HDLs%29/Guide.md), using these CPU references to specify expected behavior before writing RTL.
