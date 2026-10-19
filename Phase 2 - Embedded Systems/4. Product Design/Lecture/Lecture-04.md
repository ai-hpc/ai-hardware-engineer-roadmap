# Lecture 4 - AI appliance and robot design studio

**Course:** [Product Design](../Guide.md) | **Previous:** [Integration boundaries](Lecture-03.md)

Design a small voice appliance, or a simulated robot assistant, around one useful task and a state machine the user can understand. Your deliverable is a reviewable product brief and an executable or paper interaction prototype. A component shopping list alone is not sufficient.

## 1. State the task and limits

Example: “A person can request a local timer, see whether it was accepted, cancel it, and hear expiry while the Internet is unavailable.” For a robot simulation: “A person can request one allowed navigation goal, see its status, and cancel it when observations become unavailable.” The navigation version needs an explicit simulated operating envelope; it is not a validated physical robot controller.

Record intended environment, users, supported inputs, performance targets, power source, and dependencies. Label every number as requirement, estimate, or measurement. Include the measurement start/end event and test conditions.

## 2. Define visible states and owners

| State | User-visible meaning | Allowed transition / owner |
|---|---|---|
| Starting | Not ready for requests | Initialization reports required services ready |
| Ready | Can accept a supported request | Input manager starts capture or a typed command |
| Capturing | Input is currently being collected | Audio/input owner ends capture or cancels |
| Processing | A bounded request is in progress | Validator accepts/rejects; user can cancel |
| Active | The defined task is running | Task owner reports progress/completion |
| Unavailable | A required dependency or observation is missing | Recovery verifies the dependency before ready |
| Muted / input disabled | The stated input path is disabled | Local input/privacy control changes the state |
| Fault | Normal task execution is inhibited | Explicit recovery after diagnosis/health checks |

Treat the table as meanings to represent, not a requirement that every row be mutually exclusive. A local timer can run while microphone input is muted; a local task can remain ready while the network is unavailable. Keep independent state dimensions explicit and derive the display summary from them.

These states need not each have a different color. Test that people can distinguish required states without relying on color alone; pair meaningful text, sound, pattern or physical position with the task context. Do not claim accessibility from a color palette alone.

## 3. Connect the indicator to the actual boundary

If “microphone off” means hardware capture is disabled, show the power/data gating and which signals drive its indicator. If it means software ignores input, say so accurately. On a transition, decide what happens to queued audio, pending transcripts, model requests, and logs. Cutting new capture does not erase a previously queued recording.

For robot commands, validate type, range, source/session, expiry and allowed state before execution. Cancellation and unavailable-observation behavior must work while a model or network request is delayed. Begin with simulated motion or an indicator and record the boundary; ordinary application controls do not establish a certified safety function.

## 4. Allocate resources to the task

Draw input, processing, output, power, thermal, and network paths. Estimate memory for model parameters, live activations, buffers, framework overhead and logs. Identify the component owning each queue and its overflow policy. Estimate sustained power at the intended workload and identify how the enclosure rejects heat.

Use a microphone/speaker/enclosure arrangement as a prototype hypothesis. Measure acoustic interference and thermal noise before freezing it. Select radio/antenna placement from module integration requirements; test coexistence under the intended combined workload. A quieter visual design does not automatically produce quieter audio.

## 5. Run four walkthroughs

1. **First use:** Can a new user reach ready and tell what remains unavailable? Record assistance and ambiguous steps.
2. **Normal task:** Does the system confirm the exact accepted task and report its actual outcome?
3. **Delayed or stale input:** Inject delayed inference, missing observations, or network loss. Verify that old data does not become a new action.
4. **Cancel and recover:** Cancel during processing/active operation, then restore the dependency. Check that a cancelled request is not replayed automatically.

For a software prototype, reuse the [observation freshness lab](../../Labs/Observation-Freshness.md) and [native telemetry tests](../../2.%20Embedded%20Software/labs/Framed-Telemetry.md). They provide concrete stale/duplicate/corruption cases; they do not validate the finished enclosure or physical actuator.

### Execute the delayed-result walkthrough

The [CPU product-state lab](../CPU-Product-Contract-Lab.md) supplies a working timer prototype and eight tests. Run mute while a voice request is pending, deliver the result late, then confirm a typed local timer works while muted and offline. Its virtual clock verifies state transitions; use real input-to-application timestamps for a physical cancel-latency claim.

## 6. Write the final brief

Include the task/envelope, requirement table, state diagram, architecture, component assumptions, latency/memory/power budgets, data-retention/control behavior, recovery flow, and observed walkthrough results. Finish with unresolved experiments and a clear distinction between simulated, compiled, and hardware-tested behavior.

**Worked review example:** A prototype displays “ready” immediately after Wi-Fi connects, but its model is still loading. The fix is to define ready from required service readiness, or show a narrower “network connected” status. A stronger Wi-Fi signal would not resolve this state-model defect.

The brief passes when another engineer can derive tests from it and a learner can explain what happens during delayed inference, stale observations, cancel, and recovery. Carry the same requirements into the later AI and robotics implementation courses.
