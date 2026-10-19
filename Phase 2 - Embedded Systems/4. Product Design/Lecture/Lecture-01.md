# Lecture 1 - Turn a product need into an embedded contract

**Course:** [Product Design](../Guide.md) | **Next:** [Read a product case study critically](Lecture-02.md)

Product design connects a person's task to observable system behavior. For an embedded device, that connection changes buttons, indicators, microphone placement, power states, firmware ownership, and recovery—not just the enclosure's appearance.

Use an AI voice appliance as the running example, then transfer the method to a simulated robot. You need a basic hardware/software block diagram; no manufactured device or marketing study is assumed.

## Start with a task and an operating envelope

“Add an LLM” describes a component. “A person can start a local timer and understand whether it was accepted while the Internet is unavailable” describes a task. The latter lets you decide what runs locally, which states need indicators, and what a test must observe.

Write the environment beside the task: quiet desk or noisy kitchen, fixed power or battery, intended speaking distance, single user or shared room, and available network. Treat each as an assumption until tested. A design optimized for a bedside appliance may be inappropriate for a mobile robot or workshop.

## Translate a need into a measurable requirement

The values below are **teaching targets**, not measured product performance or validated robot safety limits.

| User need | Example requirement | Engineering consequence | Test |
|---|---|---|---|
| Know whether speech is being captured | Visible capture state matches the specified audio boundary | State owned by the audio/privacy path | Inspect transition during recording and recovery |
| Stop the current interaction | Local cancel reaches the application within 100 ms | Separate input path from model/network wait | Delay inference and inject cancel |
| Understand network failure | Show unavailable within 2 s of the defined failure condition | Bounded timeout and explicit state | Remove network in a controlled test |
| Avoid stale robot observations | Reject data older than the selected 200 ms exercise limit | Timestamp/session checks at consumer | Delay/reorder replayed observations |

A requirement must define where timing starts and ends. “Fast response” is ambiguous; time from end-of-utterance detection to first audible response is different from time to a confirmed action. Record each stage rather than hiding all latency in one number.

## Make one state authoritative

If the UI says “muted” while the audio worker still publishes new microphone frames, the system has conflicting owners. Define which subsystem owns capture permission, how it changes, and which indicators derive from that state. A software mute button is a legitimate design choice when described accurately; a hardware disconnect claim requires the corresponding electrical behavior.

For a robot, separate a user cancel request, a controller's accepted stop command, and measured stopped motion. An ordinary UI stop button is not evidence of a certified emergency-stop function. Begin with simulation or an indicator before extending the requirement to physical motion.

## Choose tradeoffs with an experiment

Compare two enclosure/control sketches by asking a learner to identify capture, busy, offline, and fault states. Record confusion rather than defending the preferred aesthetic. Evaluate microphone and speaker placement with measurements; putting microphones on top or heavy components below can be a hypothesis, not a universal optimum. Thermal noise, vibration, room reflections, access, and manufacturing all matter.

## Exercise and answer check

Write three requirements for your appliance or simulated robot. For each, identify the owner, input event, expected output, timeout, and failure behavior. Then let another person walk through startup, normal use, disconnect, and recovery using a paper prototype.

**Example:** “On stale input, the display shows unavailable while retaining the last sample time for diagnosis.” A timestamped stale replay can test this. “The device feels trustworthy” cannot yet be tested until you specify what a person should correctly understand or control.

Deliver the requirement table and a state diagram. The next two lectures use historical products to generate testable hypotheses, not to infer commercial success from specifications.
