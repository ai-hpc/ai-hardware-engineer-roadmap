# Execute one RVV binary at three vector lengths

**Time:** 45–75 minutes. **Needs:** Linux, Python 3, Clang, LLD and QEMU user-mode RISC-V support. This lab executes actual RISC-V machine code in an emulator. It does not simulate a particular chip's cache, cycle timing, power or peripherals.

**Question:** can a vector program remain correct when the register width changes?

## 1. Set up the tools

The recorded run used Clang/LLD **18.1.3** and QEMU **8.2.2** from Ubuntu 24.04 packages. These are tested versions, not claims about the newest upstream release. On a matching Ubuntu environment, the package names are:

```bash
sudo apt-get install clang lld qemu-user
clang --version
ld.lld --version
qemu-riscv64 --version
```

If you already have compatible tools, use them. The test harness accepts `COURSE_RVV_CLANG`, `COURSE_RVV_LLD` and `COURSE_RVV_QEMU` environment variables containing executable paths. It reports an explicit skip if a required tool is absent.

From this directory:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The four tests compile real binaries, execute them at three VLEN settings, check two deliberately wrong variants and inspect generated assembly. All four passed in the [recorded validation](validation.json), including the expected mutant failures.

## 2. Understand the operation

[rvv_contract.c](rvv_contract.c) computes `output[i] = 2*a[i] + b[i]` using bounded 32-bit integers. It tests lengths 0 through 97 and eight starting offsets: **784 cases per emulator configuration**. Prefix/suffix guards remain unchanged, every requested output must match an independently written scalar formula, and the total number of processed elements must equal the requested count.

With SEW=32 and LMUL=1:

| Emulated VLEN | Maximum elements in the vector register |
|---:|---:|
| 128 bits | 4 |
| 256 bits | 8 |
| 512 bits | 16 |

The loop asks `__riscv_vsetvl_e32m1(remaining)` for the number to process. It loads, computes and stores only those active elements, then advances pointers by the **returned** value.

Do not replace the returned value with `VLEN/32` or a fixed eight. The instruction's rules can choose a useful length below the maximum for some remaining lengths. Tail values beyond the active elements are not valid results to store.

The [ratified RVV specification](https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html) defines these semantics. VLEN is architectural register capacity; it is not a measurement of physical execution width or operations per clock.

## 3. Build once and execute three times

```bash
clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
  -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize \
  -fno-stack-protector -nostdlib -static -fuse-ld=lld \
  -Wl,--no-relax -Wl,-e,_start rvv_contract.c -o /tmp/course-rvv

qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64,vext_spec=v1.0 /tmp/course-rvv
qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 /tmp/course-rvv
qemu-riscv64 -cpu rv64,v=true,vlen=512,elen=64,vext_spec=v1.0 /tmp/course-rvv
```

Each correct run prints `PASS: 784 length/offset cases; guards and processed count agree` and exits zero. The executable is unchanged between runs.

The program is freestanding: it uses a small `_start` entry and Linux write/exit system calls rather than a target C library. QEMU user mode translates the target instructions and supported system calls for the host. It does **not** boot a RISC-V kernel or validate a board BSP. [QEMU's user-mode documentation](https://qemu.readthedocs.io/en/master/user/main.html) explains this boundary.

The compiler's automatic loop vectorizers are disabled so the vector operations being studied come from explicit intrinsics. Inspect the emitted `vsetvli`, `vle32.v`, `vadd.vv` and `vse32.v` instructions in the assembly test.

## 4. Learn from two failures

The test harness builds these variants automatically:

| Variant | 128-bit emulator | 256-bit emulator | 512-bit emulator |
|---|---|---|---|
| Correct, advance by returned `vl` | Pass | Pass | Pass |
| `COURSE_MUTANT_FIXED_WIDTH`, advance by eight | Fail | Pass in this QEMU configuration | Fail |
| `COURSE_MUTANT_TAIL`, always request the full width | Fail | Fail | Fail |

The fixed-width variant can skip elements on a narrow configuration or process some elements more than once on a wide configuration. The processed-count check catches duplicated work even when repeated stores happen to leave the right final values.

Its passing 256-bit result is deliberately misleading: one tested width cannot establish portability. Nor does that pass prove every legal implementation's `vl` choice would fit the hardcoded increment.

The full-width tail variant processes beyond the requested logical end. For n=1, most of those lanes are not part of the requested output. A buffer's spare capacity does not authorize treating extra elements as valid data.

## 5. Transfer the result carefully

Try n=1, 7, 8, 9, 31 and 33 by hand. Draw the active region and the first forbidden output. Then change the operation to a bounded clamp or an integer dot product and write an independent reference. For a reduction, decide how accumulation width and overflow are handled before choosing an intrinsic.

If you later run on a real board, record its core, vector extensions, compiler, affinity and software image. A vendor matrix extension is not supplied by the base vector example. Timing under QEMU cannot establish that board's throughput, power or energy efficiency.

**Finish:** submit the unchanged binary's hash, the three configurations, both mutant outcomes and a short explanation of VLEN versus work per cycle. Carry that distinction into the [chip and market review](../Guide.md).
