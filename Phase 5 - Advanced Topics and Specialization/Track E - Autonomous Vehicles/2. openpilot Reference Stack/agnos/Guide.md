# AGNOS: connect OS mechanisms to an observation deadline

**Before:** processes, scheduling, virtual memory and [openpilot boundaries](../Guide.md). **Time:** 60–90 minutes for source and budget analysis. **Artifact:** a platform manifest and acquisition-to-command timing budget. This lesson does not flash, update or modify a device.

## 1. Separate the operating system from one kernel tree

[AGNOS](https://github.com/commaai/agnos-builder) is comma's operating-system build/distribution work for supported devices. A platform kernel repository is one component; it is not the whole OS. The selected device generation determines applicable kernel, boot chain, drivers, firmware, partition layout and userspace configuration.

The old `agnos-kernel-sdm845` tree can be useful for a named historical platform. Its paths and configuration must not be treated as the current kernel for every device. Pin builder/kernel/device identities before associating an OS lecture with a source path.

In the reviewed openpilot snapshot, the common hardware code lives under `openpilot/common/hardware/`; some old paths under `openpilot/system/hardware/` are compatibility links. Inspect the target of a link instead of publishing its path as an independent implementation.

## 2. Read the update manifest as data

The pinned [AGNOS manifest](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/common/hardware/comma/agnos.json) describes named image components with URLs, hashes, sizes and flags. Its entries identify different layers of the boot/platform system. Reading it is useful even when no hardware is available.

In the source checkout from the parent guide:

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/tmp/course-openpilot-source/openpilot/common/hardware/comma/agnos.json')
for component in json.loads(path.read_text()):
    print(component['name'], component['size'], component.get('has_ab'))
PY
```

This reads local JSON only. Do not infer that a compressed-download hash, expanded-image hash and device-side verification digest have identical meanings just because their lengths match. Follow the updater's checks and record which object each hash covers. A valid checksum also does not establish that an image targets the correct hardware.

**Exercise:** choose a boot-stage component and a system component. Explain why changing userspace source revision alone does not describe the whole running system. Record the corresponding artifact identities in an experiment manifest.

## 3. Allocate a timing budget by boundary

Consider a hypothetical 150 ms maximum acquisition-to-command age:

| Stage | Chosen budget |
|---|---:|
| Acquisition completion and delivery | 30 ms |
| Preprocessing | 10 ms |
| Inference | 45 ms |
| Queueing/scheduling | 20 ms |
| Planning/control | 10 ms |
| Interface/application delay | 15 ms |
| Explicit margin | 20 ms |
| Total | 150 ms |

These are invented budgets for arithmetic, not measured openpilot timings. A sequential sum is useful only after defining the critical path; overlapping stages and pipelining need a trace. The timestamp gate should check actual age even if every individual mean latency appears below its budget.

If inference improves from 45 to 30 ms while batching adds 40 ms of waiting, total age worsens by 25 ms. This is why model FPS is not a sufficient platform objective.

<a id="os-lectures--agnos-development-changes-fork--custom-mods-for-openpilot"></a>

## 4. Match OS concepts to observable effects

| Mechanism | What it can affect | Evidence to seek |
|---|---|---|
| Scheduling and priorities | Ready-to-run delay and interference | Release/start/completion timestamps under stated load |
| Interrupt handling | Sensor completion visibility | IRQ/event trace with clock and loss indicators |
| Memory allocation/page faults | Latency spikes and footprint | Fault counts, allocations, steady-state versus startup |
| IPC and buffer ownership | Copies, queue age and stale reuse | Buffer/frame identity plus queue occupancy |
| Thermal/power management | Frequency changes and sustained latency | Temperature, frequency and repeated timing samples |
| Process supervision | Restart/recovery behavior | Producer epoch, health state and first valid output after restart |

A real-time policy or kernel does not itself prove a deadline. CPU pinning can also move contention or starve another required process. Establish a baseline, change one mechanism, and measure complete-path timing and failure behavior on a suitable isolated target.

**CPU-only exercise:** take the replay's acquisition/ready/use timestamps and insert a 40 ms queue delay. Which age changes and which computation duration stays the same? Repeat with an epoch reset. The source-level result is a hypothesis about a platform; actual target scheduling must still be measured.

## 5. Recovery belongs to the timing contract

A restarted producer may reuse a frame number or reset a local clock. A consumer must distinguish a new epoch from an out-of-order old message. Specify when the old stream becomes invalid, what the controller does while waiting and which fresh metadata permits recovery. Do not clear ordering state indiscriminately on every message.

Storage/update recovery is a different concern from control-loop recovery. Boot-slot selection, verified artifacts and rollback semantics require platform-specific tests; a successful process restart does not exercise them.

## Completion and next steps

Submit the manifest identities, a 150 ms budget with its assumptions, and a trace showing why added queue time can defeat a faster model. Use the [replay lesson](../Replay-Lab.md) for an executed timing fault. Hardware measurements remain a separate milestone with named target/software/load; no kernel-latency or update-recovery result is supplied here.
