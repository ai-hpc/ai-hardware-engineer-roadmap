# Lecture 3 - HomePod 2023: integration, sensing, and service boundaries

**Course:** [Product Design](../Guide.md) | **Previous:** [2018 case-study method](Lecture-02.md) | **Next:** [Design studio](Lecture-04.md)

The 2023 HomePod announcement illustrates how an appliance combines audio, sensing, networking, and home integration. Use it to examine boundaries between subsystems, while keeping feature descriptions separate from assumptions about user value or commercial outcomes.

## Read the source with a date attached

Apple's January 2023 announcement described computational audio and room sensing, built-in temperature/humidity sensing, and Matter home integration. It also distinguished features scheduled for later software updates. The [original announcement](https://www.apple.com/newsroom/2023/01/apple-introduces-the-new-homepod-with-breakthrough-sound-and-intelligence/) is evidence for what was announced, not proof that every feature was available on launch day or under every configuration.

For your own release plan, label each feature as implemented, measured, planned, or dependent on a particular update. A product brief should not let future work appear as current behavior.

## Trace three independent paths

```text
microphone → audio processing → request interpretation → allowed action
sensor → calibrated/validated observation → application policy → status
network endpoint → protocol translation → application state → user feedback
```

Each path has its own identity, clock, error, and ownership rules. A room-temperature measurement can lag changes in ambient conditions. An automation system may read a cached value. An acoustic system can adapt to placement while a network connection remains unavailable. One “healthy” indicator cannot explain every subsystem without a defined aggregation policy.

## Work a latency budget

Suppose a teaching voice interaction allocates 120 ms to endpoint detection, 80 ms to recognition, 150 ms to action planning/validation, and 100 ms to first audio output. A serialized budget sums to **450 ms** after the chosen reference event. These are assumed allocations, not HomePod measurements. If stages overlap, draw their dependencies before subtracting time; if recognition calls a variable-latency service, measure its distribution and cancellation behavior.

For a simulated robot, keep the observation-to-controller path separate from conversational response. A spoken “done” must not stand in for actuator completion. Link the acknowledgment to the actual state being claimed.

## Study acoustic and thermal coupling

A compact enclosure couples microphone placement, loudspeaker vibration, fan noise, heat removal, antenna placement, and service access. A top microphone array or a lower speaker chamber is one candidate arrangement, not a proven default. Compare layout candidates with recorded noise, intelligibility, vibration and temperature under the same workload.

Measure at the user's task boundary. Peak compute throughput is useful only if sustained temperature and power allow the device to meet its latency target. An idle benchmark cannot represent a simultaneous audio, inference, network and display workload.

## Make integration degrade intelligibly

List dependencies for each user task: local firmware, model availability, network, external service, and paired accessories. Then simulate one dependency failing at a time. Decide whether the task can complete locally, needs a retry, or must report unavailable. Preserve a clear path to cancel and recover.

Thread is a network technology; Matter supplies application-level interoperability over supported IP transports. A radio joining a network does not prove an accessory command succeeded. The [Thread CPU lab](../../2.%20Embedded%20Software/IoT/OpenThread/labs/CPU-Simulation.md) demonstrates how to separate network attachment, traffic, and application evidence.

## Exercise and answers

Create a dependency table for three tasks, including one fully local task. Remove a network dependency in a paper or software prototype and walk through what the user sees.

**Answer check:** Can a release note prove good household usability? No. It establishes described features/changes; usability needs appropriate observation. Can local execution alone prove privacy? No. Inspect data collection, retention, permissions, and outgoing paths.

Deliver the dependency table, a latency diagram with explicit assumptions, and a failed-dependency walkthrough. Use these in the design studio rather than copying a product's component count.
