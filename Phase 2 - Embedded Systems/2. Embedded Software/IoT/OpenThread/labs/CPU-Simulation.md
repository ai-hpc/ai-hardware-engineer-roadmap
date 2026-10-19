# Build a real Thread network on your CPU

You will build OpenThread, attach two nodes, send IPv6 and UDP traffic, take one node offline, and prove that new traffic arrives after it reconnects. The nodes run the actual OpenThread network stack; their IEEE 802.15.4 radio is replaced by a UDP transport on the computer's loopback interface.

Allow 45–75 minutes for a first build and about 30 seconds for a run. Use Linux with a C/C++ compiler, CMake, Make, Git, Python 3.10+, and Python virtual-environment support. No physical radio, privileged network configuration, border router, or Internet route is required after dependency downloads. These commands were executed on Ubuntu 24.04 with GCC 13.3 and CMake 3.28; other host platforms need their own validation.

Before starting, explain the difference between a Thread leader, a router, and a border router using the [guide](../Guide.md). Then predict whether a successful `udp send` command proves that the receiver consumed the payload.

## 1. Pin and build the stack

This lab pins the official repository at `227561fe2446fa57e8a1aa192c81e31396db9ac9`, reviewed on 2026-10-18. It is a reproducible upstream source snapshot, not a claim that every SDK vendors this revision. Follow the [official build guide](https://openthread.io/guides/build) when deliberately migrating it.

Run from the course repository root. Choose unused temporary directories; the commands do not delete an existing checkout.

```bash
git init /tmp/course-thread-src
git -C /tmp/course-thread-src remote add origin https://github.com/openthread/openthread.git
git -C /tmp/course-thread-src fetch --depth 1 origin 227561fe2446fa57e8a1aa192c81e31396db9ac9
git -C /tmp/course-thread-src checkout --detach FETCH_HEAD
git -C /tmp/course-thread-src submodule update --init --recursive

cmake -S /tmp/course-thread-src -B /tmp/course-thread-build \
  -DOT_PLATFORM=simulation -DOT_APP_CLI=ON \
  -DOT_APP_NCP=OFF -DOT_APP_RCP=OFF \
  -DOT_FTD=ON -DOT_MTD=OFF -DOT_RCP=OFF \
  -DOT_COMMISSIONER=ON -DOT_JOINER=ON \
  '-DOT_POSIX_SETTINGS_PATH="."' -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/course-thread-build --target ot-cli-ftd --parallel 4
```

The quoted dot sets persistent simulated flash files relative to the node's working directory. The harness gives each run a temporary working directory and verifies that both nodes wrote their settings there. This prevents yesterday's credentials from silently changing today's experiment. Keep the quotes around the dot: CMake must pass a C string literal to the compiler.

`FTD` means Full Thread Device, so these binaries can become routers or a leader. Enabling commissioner/joiner support does not mean this lab exercises commissioning: it provisions both nodes with the same dataset directly.

## 2. Run and inspect the experiment

```bash
python3 -m venv /tmp/course-thread-venv
/tmp/course-thread-venv/bin/python -m pip install \
  -r 'Phase 2 - Embedded Systems/2. Embedded Software/IoT/OpenThread/labs/requirements.txt'

/tmp/course-thread-venv/bin/python \
  'Phase 2 - Embedded Systems/2. Embedded Software/IoT/OpenThread/labs/run_thread_simulation.py' \
  --cli /tmp/course-thread-build/examples/apps/cli/ot-cli-ftd \
  --output /tmp/thread-evidence.json
```

The runner prints `PASS` only after all six checks succeed. Read `/tmp/thread-evidence.json`: it includes CLI version, binary hash, commands, actual role transitions, ping results, and payloads observed on the receiving process. Simulated addresses and attach times vary. The harness uses a randomly selected simulation port offset; use `--port-offset 840` for a controlled namespace when that offset is free.

The network key `00112233445566778899aabbccddeeff` is an intentionally public test fixture. The log includes the full dataset. Use it only in this local simulation; do not reuse real deployment credentials or publish production datasets.

For the course's optional integration test:

```bash
OPENTHREAD_CLI=/tmp/course-thread-build/examples/apps/cli/ot-cli-ftd \
  /tmp/course-thread-venv/bin/python -m unittest discover \
  -s 'Phase 2 - Embedded Systems/2. Embedded Software/IoT/OpenThread/labs' -v
```

Without `OPENTHREAD_CLI`, this integration test reports a skip. A skip is not evidence that the protocol ran. The [recorded validation](validation.json) identifies the actual checked build and scenarios.

## 3. Understand the state transitions

| Step | Node 1 | Node 2 | Evidence required |
|---|---|---|---|
| Create and commit dataset | Detached, then leader | Disabled | `state` eventually reports `leader` |
| Set the same active dataset | Leader | Detached, then child/router | Accepted role, not just a successful configuration command |
| Discover mesh-local endpoint address | Leader | Attached | A valid IPv6 address from `ipaddr mleid` |
| Send traffic | Three echo replies | Receives `course-seq-1` | Ping summary and receiver-side UDP line |
| Stop Thread and bring interface down | Still running | Disabled | Zero replies to an additional ping |
| Bring interface up and start Thread | Still running | Reattaches | Three new replies and `course-seq-2` on receiver |

A **dataset** holds network configuration and credentials. `dataset commit active` stores the selected configuration; it does not by itself enable the IPv6 interface or attach the device. `ifconfig up` and `thread start` initiate those steps. Attachment is asynchronous, so the runner polls a state condition with a bounded timeout instead of assuming a fixed sleep is sufficient.

The first node becomes a leader of its partition. The second may remain a child or become a router; both are valid for this two-node test. A leader is not an Internet gateway. Both endpoints use mesh-local addresses, so no border router is involved. Consult the [CLI reference](https://openthread.io/reference/cli/commands) when interpreting commands rather than treating command success as network success.

## 4. What this proves, and what to measure on a board

This run exercises real dataset parsing, MLE attachment, IPv6 echo, UDP delivery, stop/start behavior, and persistent settings. The explicit offline check prevents a harness that always reports successful connectivity from passing. Distinct payloads before and after recovery prevent an old log line from standing in for new data.

The simulation does not measure antenna performance, interference, channel occupancy, RF packet loss, power consumption, sleepy-child buffering, or hardware timing. Two nodes also do not demonstrate multi-hop route repair or leader failover. POSIX scheduler timing is not a radio latency specification. Do not copy its millisecond ping times into a product requirement.

For robotic telemetry, a restored route is only one step toward usable data. Add a boot/session identifier, sequence, and timestamp to each application record, then reject stale observations at the consumer. Reuse the [framed telemetry parser](../../../labs/Framed-Telemetry.md) and [observation freshness exercise](../../../../Labs/Observation-Freshness.md) to make those application rules explicit. UDP preserves datagram boundaries, so COBS framing is unnecessary inside a single UDP datagram; its length, CRC, and sequence exercises still help teach transport-independent validation. CRC is not authentication.

## Troubleshooting by symptom

| Symptom | First check |
|---|---|
| CMake reports missing Mbed TLS sources | Run the recursive submodule command at the pinned revision |
| `pexpect` cannot be imported | Invoke the virtual environment's Python and install this lab's requirements |
| Node never becomes leader | Inspect the CLI error and loopback UDP restrictions; this simulation needs local sockets |
| Node 2 never attaches | Compare complete datasets, channel, key, and simulator port offset |
| Ping succeeds but UDP test fails | Check `udp open`, the receiver's bind port, and the receiver process's output |
| Private flash-file check fails | Rebuild with the quoted settings-path option, rather than reusing an unrelated binary |
| Sporadic attach failures with another lab running | Use an unused offset in `1..1000`; this range assumes the default 33-node simulator build |

Do not increase timeouts until an error becomes invisible. Save the failed command/output, check dataset equality and process lifetime, then rerun one controlled change.

## Exercises with answers

1. **Why is `udp send ... Done` insufficient?** It confirms local command acceptance. Only the receiving process's payload line proves that this datagram reached that application endpoint.
2. **Does sharing a dataset exercise secure joining?** No. It is direct credential provisioning. A commissioning exercise must run commissioner/joiner flows and separately test a rejected join credential.
3. **Why permit both `child` and `router`?** The required behavior is attached bidirectional communication, not a particular dynamic role assignment.
4. **What does the offline ping establish?** The test notices a deliberately disabled endpoint. It does not prove a bound on mesh route repair under arbitrary radio failures.
5. **Next experiment:** Add a third node, record role and neighbor tables, then remove a router. To claim a multi-hop path was repaired, show that the initial path actually used that router; three processes alone do not establish the topology.

Deliver the source revision, build configuration, evidence JSON, and an explanation of each observation. That is enough for another learner to reproduce this network and understand its limits.
