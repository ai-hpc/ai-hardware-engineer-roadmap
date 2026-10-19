# CPU lab: messages, timeouts and duplicate commands

**Time:** 45–75 minutes. **Needs:** Python 3.10+; no packages, radio, root privileges or internet access. Work from the repository root. This lab uses real TCP/UDP sockets bound only to `127.0.0.1` on OS-assigned ports. It modifies only a simulated counter.

## 1. Predict the result

TCP carries a byte stream. A message may span multiple reads, and a read may contain several messages. The lab prefixes each UTF-8 JSON payload with a four-byte unsigned big-endian length and caps payloads at 4,096 bytes. The receiver loops for both header and payload and uses one absolute deadline for the complete frame.

**Worked frame:** the bytes of `{}` have length 2, so the prefix is `00 00 00 02`. Length counts encoded bytes, not Python characters: non-ASCII text may use several bytes per character. Two `sendall` calls do not imply two packets or two reads. [Python socket contract](https://docs.python.org/3/library/socket.html).

The UDP experiment sends the following simulated counter changes:

| Sequence | Delta | Injected condition | Expected application effect |
|---:|---:|---|---:|
| 0 | +2 | First request discarded | +2 after retry |
| 1 | +3 | First acknowledgement discarded after application | +3 once, cached response on retry |
| 3 | +4 | Delivered before sequence 2 | +4 |
| 2 | +100 | Older than last applied sequence | Rejected |
| 4 | +100 | Deadline already expired | Rejected |
| 5 | +1 | Fresh | +1 |

Predict final counter **10**, four distinct applications, and two retries. These faults are deliberately injected by the application; they are not measurements of random UDP loss.

## 2. Run actual sockets

```bash
python3 'Phase 4 - Track B - Nvidia Jetson/5. Application Development/2. Network and Connectivity/labs/localhost_lab.py' \
  --output /tmp/course-network.json
python3 -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/2. Network and Connectivity/labs' -v
```

The [recorded run](labs/measured-run-2026-10-18.json) contains each request, retry count, result, Python/host version and source hash. Eleven tests passed on 18 October 2026. They exercise real loopback connections, split/coalesced frames, EOF, excessive lengths, absolute deadlines, UDP faults and bounded SLIP parsing.

The 30 TCP round trips inject 15 ms of server work on every fifth request. Recorded p50 was **0.277 ms**, p95 **15.506 ms**, and maximum **15.564 ms**. Values vary by host and load. Percentiles use the nearest-rank rule: sort `n` observations and choose index `ceil(p*n/100)-1`. This small sample illustrates the difference between median and tail behavior; it does not estimate a rare production failure rate.

The two UDP retries took about 30 ms each because the client deliberately waited 30 ms for the missing response. The network itself did not incur 30 ms of RF delay.

## 3. Trace the important code

Read [localhost_lab.py](labs/localhost_lab.py):

- `read_exact` keeps one whole-frame deadline, so a peer cannot extend it indefinitely by sending another byte just before every per-read timeout.
- `CommandGate` checks session, integer fields, deadline, sequence and bounded capacity before changing the counter.
- A retry keeps the same sequence, delta and deadline. Reusing an ID with changed content is rejected.
- A cached acknowledgement reports the value after the original application. It need not equal the counter's current value after newer commands.

The lab deliberately uses one in-memory session with at most 32 applied IDs and then refuses more. It does not discard old IDs and claim perpetual deduplication. Restarting loses the cache. This is **not exactly-once delivery across crashes**, persistent transaction handling, authentication, or a deployable robot command server.

The deadline clock is valid because both endpoints run on the same host. Across machines, monotonic clocks have unrelated origins. A production freshness design needs a defined clock/lease contract and bounded uncertainty. An acknowledgement can be lost after the action; timeout alone cannot prove non-execution.

## 4. Learn serial framing without pretending to run Zigbee

[slip.py](labs/slip.py) implements SLIP byte framing. `C0` ends a frame; payload `C0` becomes `DB DC`; payload `DB` becomes `DB DD`. For payload `01 C0 DB`, this implementation writes:

```text
C0 01 DB DC DB DD C0
```

The decoder accepts fragmented input, ignores empty boundaries, bounds the buffer, and discards malformed/oversized frames until the next delimiter. The negative tests verify recovery on a later valid frame. Its malformed-input policy is explicitly chosen for this exercise.

SLIP itself provides no checksum, authentication or Zigbee semantics. Legacy ESP ZNSP adds its own message layout and CRC. Passing these framing tests does not create a compatible Zigbee host. [RFC 1055](https://www.rfc-editor.org/rfc/rfc1055).

## 5. Change one thing and explain it

1. Run with `--slow-ms 30 --samples 50`. Predict which percentiles increase. **Answer:** server work raises the delayed tail; the median should remain near the normal path, subject to scheduling noise.
2. Temporarily remove the duplicate-cache lookup, then run the UDP test. **Answer:** replay may be rejected as old, or may execute twice if ordering checks are also relaxed. Either behavior breaks the intended retry contract.
3. Why does the expired command not advance the highest accepted sequence? **Answer:** it was not applied; treating it as accepted can suppress a later valid lower sequence. Real applications must define this policy explicitly.
4. Replace the 32-entry limit with a two-entry limit in a separate gate test. Predict the third distinct accepted request. **Answer:** `session_full`; silently forgetting IDs would change replay behavior.
5. If a stream pauses halfway through a frame and disconnects, should the parser join its bytes to a new connection? **Answer:** no; reset framing state at the connection boundary.

**Submit:** the original and varied reports, your predicted counter trace, and one added negative case with a clear reason. Then choose a [physical network path](Guide.md) and repeat measurement on real hardware without reusing loopback numbers as radio evidence.
