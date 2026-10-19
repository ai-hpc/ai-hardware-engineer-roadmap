# CPU lab: make the product's status and cancellation claims executable

**Executed 18 October 2026.** Build a virtual-time prototype for the local timer task in [Lecture 04](Lecture/Lecture-04.md). It has no microphone, speech model, network connection or robot actuator. Python executes the actual state logic and tests; all event times are synthetic, so they do not measure operating-system or hardware latency.

**Prerequisites:** Python dictionaries and the difference between an event and a state. **Time:** 45–60 minutes. **Artifact:** an event transcript, a requirement-to-test table and one independently written failure scenario.

## 1. Define the promise before the implementation

The fixture makes these explicit product choices:

| Requirement | Boundary tested |
|---|---|
| Connected Wi-Fi does not mean the application is ready | Local services must report readiness |
| Mute blocks new voice frames | A software capture-permission gate rejects fixture bytes |
| Mute discards queued audio and unaccepted voice requests | Queue cleared; their late results cannot create timers |
| A timer already accepted continues while muted | Input permission and task state are independent |
| Typed local timers work without Internet | No network-dependent timer path |
| Cancel clears pending requests and the active timer | Late results cannot restart cancelled work |
| A result is valid only before its 500 ms request deadline | At the deadline, it is expired |
| Results from a previous boot are rejected | Token includes a boot/session identity and a sequence |

The **500 ms deadline is an exercise requirement**. It is not a measured model response time or a robot safety limit. The mute contract is a software gate; it does not prove a physical microphone disconnect or deletion of data already sent outside the application.

## 2. Run the interaction and tests

From the repository root:

```bash
python3 'Phase 2 - Embedded Systems/4. Product Design/labs/product_contract.py'
python3 -m unittest discover -s \
  'Phase 2 - Embedded Systems/4. Product Design/labs' -v
```

The [implementation](labs/product_contract.py) uses a virtual monotonic clock. `advance(1250)` processes events at 1250 ms immediately; it does not sleep for 1.25 seconds. The demo should show:

1. Network connected but state still `starting`.
2. Services ready at 100 ms; a voice request starts at 120 ms.
3. Mute at 150 ms clears queued fixture audio and that pending voice request.
4. Its result arrives at 200 ms and is rejected.
5. A typed timer is accepted at 250 ms while muted and offline.
6. Exactly one timer expiry is recorded at 1250 ms.

[Recorded transcript](labs/measured-demo-2026-10-18.json) · [eight executed tests](labs/validation-2026-10-18.json).

## 3. Trace the delayed-result bug

Imagine an implementation that sets `state = ready` on cancel but leaves an old request eligible. The worker finishes later and creates a timer. The user sees a cancelled request come back to life.

The fixture instead keeps a bounded pending-request record. Cancellation removes it. A result must match an outstanding token in the current session before it may create a timer. Duplicate results find no pending request after the first acceptance. A new boot gets a different identity, so a reused numeric sequence is not sufficient to match an old result.

This mechanism handles local state transitions in the fixture. If a request already reached an external service, removing a local record does not undo its side effect. That service needs an explicit cancellation/expiry or idempotency/reconciliation contract. Apply this distinction before connecting the later [agent workflow](../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Local-RAG-Workflow-Lab/Guide.md).

## 4. Do not force independent facts into one exclusive state

The device can be both **muted** and **running a timer**. It can be both **offline** and **ready for a local task**. The prototype records input permission, service readiness, task state and network state separately. Its display summary derives from those facts.

A single enumeration containing only `ready`, `muted`, `offline` and `active` would lose information unless you defined every combination. Draw a small state machine for the timer, then annotate separate mute/network facts. Decide which information the user needs at each step.

The audio fixture queue has four entries and drops its oldest frame when full; the test checks the drop count. That is a deliberate freshness policy for teaching. A real speech pipeline must evaluate whether dropped audio corrupts recognition and should cancel/restart a segment. This lesson does not claim that silently dropping arbitrary speech frames is an acceptable product policy.

## 5. Guided experiments and answer checks

- **Deadline boundary:** Start a request at 10 ms. A response at 509 ms is eligible; one at 510 ms is expired. Explain why two components must agree on the boundary convention.
- **Mute an accepted timer:** Start and accept a timer, then mute. It should still expire. If you want a different product behavior, change the requirement and tests together; do not silently equate mute with cancel.
- **Cancel during processing:** Start a request, cancel it, and then deliver its result. It must be rejected even after unmuting or reconnecting.
- **Independent extension:** Add an explicit model-load failure and recovery transition. A network reconnect must not clear that failure. Write the expected transcript before implementation and make the ready indicator depend on the required services.

**Diagnostic:** Does passing the virtual-time cancel test prove a real cancel button responds within 100 ms? No. It proves the state rule for the supplied event order. Real timing needs the input hardware, scheduling, queues and task load, with timestamps at the defined start/end boundary.

## 6. Finish the product brief with evidence

Map each requirement to its test name and one observed transcript line. Mark physical mute, acoustic behavior, real-time scheduling, privacy outside the process and actuator behavior as unmeasured. Conduct the paper walkthrough from Lecture 04 separately: correct state logic does not establish that a person understands its indicator.

Carry the token/session/deadline contract into the firmware and robot integration work. Keep conversational acceptance, controller acceptance and actual physical completion as different events.
