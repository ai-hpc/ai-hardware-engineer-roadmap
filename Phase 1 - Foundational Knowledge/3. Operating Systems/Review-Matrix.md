# Operating systems: review and evidence matrix

Reviewed during the 2026-10-18 course revision. This matrix distinguishes conceptual/source review from executed experiments. A revised explanation is not automatically a hardware validation. Return to the [course guide](Guide.md) for the learning sequence.

## Lecture coverage

| Lecture | Main contract reviewed | Reproducible exercise or remaining boundary |
|---|---|---|
| [1](Lectures/Lecture-01.md) | Privilege, kernel/BSP scope, sysfs/procfs and driver isolation | Read-only host inspection; exact board release remains target-specific |
| [2](Lectures/Lecture-02.md) | Thread identity, CoW, reaping, affinity, quota and wait states | SPSC real threads; kernel internal layouts are not UAPI |
| [3](Lectures/Lecture-03.md) | IRQ/wakeup/execution separation, workqueue contexts and teardown | RT wakeup measurement; no device IRQ measurement claimed |
| [4](Lectures/Lecture-04.md) | Syscall error contracts, shm sizing, clocks, epoll, timer slack | RT/BPF build labs; shared-memory lifecycle shown as pseudocode |
| [5](Lectures/Lecture-05.md) | Boot stages, UEFI, DT cells/status, module build and trust scope | Links real DT tools and build-only module; no module load |
| [6](Lectures/Lecture-06.md) | Fair service versus response deadlines and CPU reservations | RT demo; privileged admission/scheduling is a separate milestone |
| [7](Lectures/Lecture-07.md) | PREEMPT_RT mechanisms versus measured/analytical bounds | Finite wakeup experiment, not a worst-case proof |
| [8](Lectures/Lecture-08.md) | Eligible/effective CPUs, IRQ/housekeeping and interference | Obtained-affinity test; no global host isolation changes |
| [9](Lectures/Lecture-09.md) | Ownership, waiting, completion reuse and locking contexts | Existing RT/SPSC exercises; no live DMA abort test |
| [10](Lectures/Lecture-10.md) | Publication, slot release, progress and reclamation | Native SPSC plus sanitizer and intentional-overwrite checks |
| [11](Lectures/Lecture-11.md) | Lock-order graph, PI/ceiling limits, corrected Pathfinder account | Worked blocking arithmetic; no privileged PI timing experiment |
| [12](Lectures/Lecture-12.md) | Mapping, faults, lifetime and physical-address distinctions | Worked buffer/mapping analysis |
| [13](Lectures/Lecture-13.md) | Translation counts, TLB cost and huge-page policy | Explicit binary-unit arithmetic; no huge-page speed claim |
| [14](Lectures/Lecture-14.md) | Allocation, contiguity, CMA and failure paths | API/source review; no device allocation stress test |
| [15](Lectures/Lecture-15.md) | DMA direction/counts/order/completion, managed-memory APIs | Six-entry/four-segment worked example; no GPU/device run |
| [16](Lectures/Lecture-16.md) | NUMA topology, policy versus residency and measured costs | Documented placement experiment; no multi-node result invented |
| [17](Lectures/Lecture-17.md) | Probe publication, dependency errors, devm lifetime, overlays | Linked DT compilation/overlay lab; physical bring-up remains |
| [18](Lectures/Lecture-18.md) | ioctl dispatch/UAPI, mapping lifetime, V4L2/GPU ownership | Driver fragments explicitly labeled; no camera capture claimed |
| [19](Lectures/Lecture-19.md) | io_uring terminal completion and DMA-BUF sharing | Bounded-lifetime reasoning; no invented IOPS or liburing runtime |
| [20](Lectures/Lecture-20.md) | One-way PCIe ceilings, NVMe queue limits, conditional P2P | Worked bandwidth/Little's-law arithmetic; no P2P benchmark |
| [21](Lectures/Lecture-21.md) | Durable replacement, snapshots, tmpfs and OverlayFS | Read-only storage inspection; no host filesystem remount |
| [22](Lectures/Lecture-22.md) | RPMB scope, endurance units, storage identity and A/B state | Links actual file-based OTA protocol tests, not firmware/power-cut proof |
| [23](Lectures/Lecture-23.md) | Quota/mask semantics, toolkit dependencies, MIG/MPS scope | Linked actual CPU/container lifecycle lab; Orin MIG claim removed |
| [24](Lectures/Lecture-24.md) | Pinned deployed-system claims and full-path timing | Links maintained AGNOS/Jetson material rather than old platform assumptions |
| [25](Lectures/Lecture-25.md) | Recipe→package→image boundaries and evaluated variables | Reuses actual native package and 922/1105-recipe metadata parses; no target image boot |
| [26](Lectures/Lecture-26.md) | BPF pairing, event ABI, loss, compilation versus load/attach | Real BPF ELF/skeleton/reader build and malformed CLI checks |

Lectures 6–10, 12–14, 16 and 24 retain the earlier substantive revision in this mission. This pass corrected the other bodies and revisited 11/15 where old historical/device claims remained. All eight consolidation notes were checked for consistency; notes 1–3 and 6–8 were rewritten to remove conflicting short-form claims. The [14-question final practice](Final-Test-Problems.md) now checks buffer reuse, durability, resource limits, and build-evidence boundaries as well as scheduling and memory.

## Executed OS checks

Run from the repository root with the listed prerequisites:

```bash
python3 -m unittest discover -s 'Phase 1 - Foundational Knowledge/3. Operating Systems/labs' -p 'test_*.py' -v
```

The recorded review used an isolated libbpf header directory through `COURSE_BPF_INCLUDE`; without those optional prerequisites the BPF suite explicitly skips. A skip is not a native build pass.

| Suite | Actual result | What it exercised |
|---|---:|---|
| `test_rt_demo.py` | 3 passed | C17 warnings-as-errors; actual default/allowed CPU affinity; malformed options; rejected memory lock |
| `test_spsc_queue.py` | 3 passed | Optimized reference/real-thread transfer; ASan/UBSan; intentional missing-full-check defect detected |
| `test_bpf_build.py` | 3 passed | Clang BPF object; bpftool skeleton; C17 reader; ELF sections; seven invalid argument cases |

The combined run passed all **9 tests**. No BPF program was loaded or attached, no kernel module was loaded, and no camera/GPU/flash operation was performed by this OS test suite. Linked labs have their own narrower validation records and should be cited on those terms.

## Sources and maintenance

Version-sensitive statements link to primary Linux/man-pages, NVIDIA, Yocto/BitBake, liburing, and OP-TEE documentation at the point of use. The Pathfinder correction uses a preserved first-person letter by its flight-software lead, identified as a historical account. Numerical examples state their units and assumptions; unmeasured universal latency/IOPS tables were removed.

For a later target run, add the exact kernel/configuration, BSP/firmware, architecture, device identities, tool versions, workload, permissions, measurement clocks, failures, and artifacts. Keep “source reviewed,” “compiled,” “executed on a host,” and “validated on hardware” separate in the resulting report.
