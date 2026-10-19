# Industrial integration lab: read a register without losing its meaning

**Purpose:** decode a fictional sensor through a real localhost TCP connection,
then distinguish a valid packet from a valid measurement. **Prerequisites:**
Python bytes, hexadecimal, and the [industrial overview](Lecture-01.md).
**Layers and roles:** embedded interfaces, networking, and robot integration;
preparation for embedded robotics and automation engineering. Allow 15 minutes
for setup, 30 for execution and packet decoding, and 45 for fault analysis.

The worked example uses generated data and a read-only subset of Modbus TCP.
The server binds only to `127.0.0.1`, chooses an unused port, and exits after its
scripted exchanges. No PLC, physical sensor, robot, or external endpoint is used.
You need Python 3.10+ and permission to open a local TCP socket. The optional
library exercise additionally downloads one small pinned Python package.

## 1. Three agreements are needed

A network connection tells you that bytes can move. A protocol tells you which
bytes belong together and what operation they encode. An application contract
tells you what a value means. Keep all three visible:

| Layer | This fixture | A defect the lower layer cannot detect |
|---|---|---|
| Transport | TCP stream between two local sockets | A peer sends a valid but wrong transaction |
| Protocol | Unit 1, function 03, two holding registers | Register 0 represents the wrong engineering quantity |
| Application | Register 0 = temperature in hundredths °C; register 1 = supply in mV | The sensor stopped updating while reads still succeed |

Our fictional map contains `[2500, 5000]`, meaning **25 °C and 5 V**. These scales
are supplied by this exercise, not by Modbus. A device manual must define its
own map, signedness, scale, invalid values, and any multi-register word order.
A display label such as `40001` is not the same thing as the zero-based address
encoded in the request. Confirm the device's convention rather than subtracting
a prefix from every unfamiliar label. The protocol organization maintains the
[Modbus specifications](https://www.modbus.org/modbus-specifications).

## 2. Decode a complete exchange by hand

We request two holding registers starting at address zero. Each multibyte field
in this example uses big-endian byte order:

```text
request:  12 34  00 00  00 06  01  03  00 00  00 02
          tx     proto  length unit fn  address count
response: 12 34  00 00  00 07  01  03  04  09 c4  13 88
          tx     proto  length unit fn bytes value0 value1
```

`0x1234` identifies this transaction. Protocol identifier zero selects Modbus.
The length counts the **unit identifier plus PDU**, so it is 6 in the request
and 7 in the response. It does not count the first six header bytes. Function
`03` reads holding registers. The response byte count is four because two
16-bit registers occupy four bytes. `0x09c4 = 2500`, and `0x1388 = 5000`.

The response must match the outstanding transaction, unit, function, and expected
data length. Reading unknown address 100 produces function `83`, exception
code `02` (illegal data address). That is an application-visible failure, not a
successful empty reading. A request for an unsupported function receives exception
code `01`; this teaching server has no write implementation. Compare the framing
logic with [libmodbus's TCP implementation](https://github.com/stephane/libmodbus/blob/master/src/modbus-tcp.c).

## 3. TCP does not preserve your `send` boundaries

One send can arrive in several reads; several sends can arrive together. Calling
`recv(12)` does not guarantee twelve returned bytes. The implementation first
collects the seven-byte header, checks its bounded length, then collects exactly
the remaining PDU. A peer closing early raises an error. It leaves the following
frame in the stream for the next call.

Separate sends in a real socket experiment do not force separate receives. The
tests therefore use **both** actual TCP exchanges and a deterministic stream
fixture that returns short reads at every chunk size. This proves a property that
a successful run on an unloaded loopback interface could miss.

From the repository root, run these **tested here** commands:

```bash
cd "Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Industrial and Embedded Robotics/labs"
python3 -m unittest -v
python3 modbus_read.py > my-modbus-result.json
```

Expected: **11 tests pass**. The result contains three exchanges: registers
`[2500, 5000]`, illegal-address exception `2`, and unsupported-function exception
`1`. The independent golden packet, truncations, incorrect length, mismatched
transaction/unit, excessive quantity, and the end of the 16-bit address space
are checked. Compare with [the recorded output](labs/modbus-reference.json).

The socket timeout bounds each blocking operation in this small experiment.
It is not an end-to-end control deadline, a network benchmark, or proof of a
hard real-time system. The fixture does not implement a full Modbus server.

## 4. Repeat with an actual maintained client

The optional exercise was executed with **PyModbus 3.16.1**. Install it in a
separate environment from the same `labs` directory:

```bash
python3 -m venv .venv-modbus
.venv-modbus/bin/python -m pip install 'pymodbus==3.16.1'
.venv-modbus/bin/python -m pip check
.venv-modbus/bin/python pymodbus_client.py --output my-library-result.json
```

The script starts its own local teaching server, connects a real client, reads
the two registers, then checks the illegal-address response. Expected:
`checks_passed: 2`, `registers: [2500, 5000]`, and exception code `2`. The
[recorded library result](labs/pymodbus-reference.json) includes source hashes
and the request bytes produced by the library.

For this pinned version the call is
`client.read_holding_registers(0, count=2, device_id=1)`; older examples may use
different parameter names. Check `.isError()` before consuming `.registers`.
Serialize calls on one client: the synchronous client is not thread-safe.
See the [versioned client API](https://pymodbus.readthedocs.io/en/v3.16.1/source/client.html)
and [3.16.1 release](https://github.com/pymodbus-dev/pymodbus/releases/tag/v3.16.1).

## 5. A good packet can still be an old measurement

Suppose reads complete every 20 ms while the physical sensor refreshes every
100 ms. Five successful reads can contain the same sample. A transaction ID
correlates a response with a request; it does not identify a new sensor sample.
TCP delivery likewise does not prove that the device's acquisition task is live.

For a real integration, define a measurement counter or capture timestamp if the
device supplies one, the counter wrap/reset rules, and the allowed acquisition-to-use
age. If only host receive time exists, label it honestly: it cannot reveal an
unknown device-side delay. A multi-register read also needs the device's guarantee
about whether those values form one coherent acquisition. A network timeout
requires a documented controller response; ordinary protocol success does not
establish a protective function.

For example, a conveyor at 0.5 m/s moves **0.05 m during 100 ms**. Converting a
temperature register correctly says nothing about whether a position estimate
is recent enough for a grasp. Connect this timing argument to the
[capture-time perception lab](../Advanced%20Perception%20and%20AI%20for%20Robotics/Tracking-Lab.md).

## 6. Variations and answer guidance

1. Change register 0 to 3125. Predict the application value before executing.
2. Change one response transaction byte while leaving its measurement unchanged.
   Explain why an apparently plausible value must be rejected.
3. Sketch a device record containing value, acquisition counter, and a validity
   flag. Define how a consumer treats repeated counters, reconnects, and an
   invalid sensor reading. Do not add a physical endpoint to this lab.

<details markdown="1"><summary>Answers and completion criterion</summary>

1. The map specifies hundredths °C, so the result is **31.25 °C**. A different
   device's map could specify another scale or signed representation.
2. A valid value from another outstanding or delayed request cannot satisfy this
   request. Payload plausibility does not replace transaction correlation.
3. A repeat can be a valid repeat of an old acquisition; it is not new evidence.
   Reconnect/reset creates an epoch ambiguity that the contract must resolve.
   Invalid sensor data must remain invalid rather than becoming a default zero.
   The device's update and coherence guarantees determine what is defensible.

Finish with your packet annotation, both raw outputs if the optional dependency
is available, one failing mutation and its explanation, and a hypothetical
measurement-freshness contract. State separately that no physical I/O,
TLS/authentication, serial bus, timing guarantee, or protocol certification was
tested. Continue to the [ROS integration lab](../Advanced%20Robot%20Operating%20System/Integration-Lab.md)
to see how reliable delivery and application backlog differ.

</details>
