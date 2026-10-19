# Native SIMD lab: prove the lanes before timing them

**Time:** 60–90 minutes. **Needs:** Python 3 and GCC or Clang with C++17. An x86 CPU with AVX2 and FMA runs the vector cases; other CPUs run the scalar case and report vector skips. Clang can cross-compile the two ARM functions without an ARM board or target runtime.

**Question:** how can a short vector loop produce the wrong answer even when it compiles and looks fast?

The answer has four parts: which elements an instruction reads, which lanes it combines, which bits control a selection, and where floating-point rounding occurs. These are also useful questions when debugging image preprocessing and neural-network kernels.

## 1. Run real code

From this directory:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The harness compiles [simd_contracts.cpp](simd_contracts.cpp), executes the binary, repeats the native checks with AddressSanitizer and UndefinedBehaviorSanitizer, and builds two deliberately broken variants. It uses a temporary directory and deletes its build products. The review on 2026-10-18 passed **11 tests**, including the actual AVX2/FMA cases and the AArch64 assembly checks. See [validation.json](validation.json) for the environment and source hashes.

To inspect one case yourself:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off simd_contracts.cpp -o /tmp/course-simd
/tmp/course-simd lengths
/tmp/course-simd lanes
/tmp/course-simd fma
```

Notice the absence of global `-mavx2`. A dispatcher compiled for the baseline CPU checks support before entering a function marked `target("avx2,fma")`. A program globally compiled for a newer ISA might execute unsupported instructions before reaching its feature check. These attributes and built-ins are GCC/Clang interfaces; other compilers require their own dispatch mechanism. [GCC documents the runtime feature checks](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html).

## 2. Get every element exactly once

The function computes `c[i] = (a[i] + b[i]) * 0.5f`. An eight-lane vector handles groups of eight and a scalar loop handles the remainder.

For **n = 19**, predict the work before running it:

| Work | Element indices |
|---|---|
| First vector | 0–7 |
| Second vector | 8–15 |
| Scalar remainder | 16–18 |

The tests cover all lengths from 0 through 97 and eight starting offsets: **784 shapes per implementation**. Inputs end at the logical array boundary, allowing the sanitizer to catch overreads. Output guard values catch writes before and after the result. Results begin as NaNs, so an unwritten tail cannot accidentally pass as zero.

The fixture uses multiples of powers of two, so a separately computed double-precision reference is exact for these small values. Arbitrary scientific data usually needs an error tolerance justified by the algorithm, not exact equality.

**Try:** compile with `-DCOURSE_MUTANT_MISSING_TAIL`. The `lengths` case fails with `tail/value mismatch`. This confirms the tests detect the bug they are meant to teach.

## 3. Draw the lanes

Start with `[1,2,3,4,5,6,7,8]` in lane order:

| Operation | Result |
|---|---|
| `shuffle_ps(..., 0x1b)` | `[4,3,2,1,8,7,6,5]` |
| `permutevar8x32_ps` with indices 7 down to 0 | `[8,7,6,5,4,3,2,1]` |
| `hadd_ps(v,v)` | `[3,7,3,7,11,15,11,15]` |
| Add halves, then reduce the four lanes | `36` |

The first operation reverses each half; it does not reverse the whole register. The second mutant substitutes it for a full reversal and is rejected. Read the operation's lane mapping in the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), rather than assuming that a 256-bit type implies every shuffle can cross its midpoint.

**Check:** why does applying `hadd` twice still not necessarily give the eight-element total? Each half reduces separately. The halves must eventually be combined.

## 4. Masks are bit patterns

For `blendv_ps(false_value, true_value, mask)`, the sign bit of each mask lane selects the result:

| Mask bits | Selected value |
|---|---|
| `0x00000000` | false |
| `0x00000001` | false |
| `0x7fffffff` | false |
| `0x80000000` | true |
| `0xffffffff` | true |

Comparison intrinsics produce all-zero or all-one lanes, which fit this contract. Converting a Boolean to the integer `1` and reinterpreting its bits does not. The fixture executes this exact distinction.

The `nan` case adds a second caution: x86 `MAXPS` can select its second operand for a NaN or equal signed zeros. Decide whether your preprocessing should reject, propagate, or replace NaNs before substituting one API's `max` for another.

## 5. One rounding can change the answer

Let `a = 1 + 2^-23`, `b = 1 - 2^-23`, and `c = -1`. Their exact product is `1 - 2^-46`.

- Separate FP32 multiply then add rounds the product to 1, then returns **0**.
- A fused multiply-add keeps the product's extra precision until the addition and returns **−2^-46**.

Neither result is evidence of an instruction malfunction. They implement different rounding contracts. The test fixes the separate intermediate and disables automatic contraction so the example remains meaningful.

The conversion case compares `[1.9,-1.9,2.5,-2.5]`: truncation yields `[1,-1,2,-2]`; round-to-nearest with ties to even yields `[2,-2,2,-2]`. It saves and restores the x86 floating-point control state. Out-of-range and NaN conversion behavior needs a separate explicit contract; this example uses finite in-range inputs.

## 6. Inspect ARM output without claiming ARM execution

```bash
clang++ --target=aarch64-none-elf -ffreestanding -O2 -S neon_contracts.cpp -o /tmp/course-neon.s
cat /tmp/course-neon.s
```

[neon_contracts.cpp](neon_contracts.cpp) has two leaf functions. The generated code contains two `FADDP` instructions for `vaddvq_f32` and `FCVTZS` for truncation toward zero. This agrees with the [Arm intrinsic specification](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html).

This is an actual compiler check, **not an ARM execution or performance measurement**. Run on the intended ARM CPU before claiming runtime correctness, throughput or power consumption there.

## Finish with evidence

Submit your test output, the n=19 lane drawing, both mutant failures, and a two-sentence explanation of the FMA result. Then extend the function to normalize an image row with a scalar reference and a row stride. Include widths that are not multiples of eight. Only after those checks pass should you time it; include input size, CPU, compiler flags, repetitions and dispersion with every performance claim.

Return to the [SIMD guide](../Guide.md).
