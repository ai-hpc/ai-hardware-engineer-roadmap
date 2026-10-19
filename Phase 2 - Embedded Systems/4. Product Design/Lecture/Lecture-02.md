# Lecture 2 - HomePod 2018: separate specifications from product hypotheses

**Course:** [Product Design](../Guide.md) | **Previous:** [Requirements](Lecture-01.md) | **Next:** [2023 platform boundaries](Lecture-03.md)

A historical product is useful when it helps you ask better engineering questions. It is weak evidence for claims such as “users rejected it because the assistant was poor” unless you have appropriate user research or commercial data. This case study therefore separates documented launch features from hypotheses you can test in your own prototype.

## Establish the historical facts

Apple's January 2018 announcement described HomePod as a music speaker with Siri tasks, HomeKit accessory control, and an Apple Music integration. It announced February 9 availability in the initial markets. Those are documented launch statements; promotional descriptions of sound quality are not independent comparative measurements. See the [original launch announcement](https://www.apple.com/newsroom/2018/01/homepod-arrives-february-9-available-to-order-this-friday/).

Do not use a current support page to silently rewrite a 2018 feature set. Also, do not assume that a launch-era limitation is still a current limitation. State the date and software context whenever they matter.

## Convert observations into your own experiment

| Product question | Prototype experiment | Evidence to record |
|---|---|---|
| Does audio quality help users understand responses? | Play the same spoken replies at controlled levels/noise positions | Correctly understood content and replay requests |
| Is the assistant useful for routine tasks? | Ask participants to complete timer and supported home-control tasks | Completion, correction and failure counts |
| Is setup understandable? | Give a first-time participant the package and intended instructions | Time to ready, assistance required, abandoned steps |
| Are privacy states clear? | Ask participants to identify capture/mute states during scripted changes | Correct interpretation and ambiguous states |

Keep your own observations separate from claims about HomePod customers. A five-person formative session can reveal usability problems; it does not establish market demand or a reliable population-wide percentage.

## Shared-room behavior is an interface requirement

A voice appliance may hear guests, children, a television, or another household member. Voice recognition can help personalize a response, but recognition alone should not silently authorize every private-data or physical-control action. Specify the allowed action classes and the extra conditions each requires.

For a prototype, choose a small local command set and make rejected/ambiguous requests understandable. A timer request can be confirmed with its duration; a request that changes an external device needs an explicit action policy. A generated answer and an executed action are distinct outputs.

Privacy claims should identify a boundary. “Local wake-word processing” describes where one stage runs; it does not establish where later audio, text, logs, or embeddings go. Draw those data paths and retention settings. If the product says a physical control disables microphone capture, inspect the electrical path and queued-data behavior that implement that statement.

## Ecosystem dependencies are design inputs

A service integration can provide useful capabilities and also add account, network, availability, and compatibility dependencies. List which basic tasks remain available without it. Test a disconnected session and show the user which task could not complete. Do not invent a generic rule that every closed ecosystem is unusable or every local implementation is private and reliable.

## Exercise with an example answer

Choose one launch feature from the primary source and write: documented fact → your design hypothesis → experiment → possible failure.

Example: “The launch described voice-controlled timers” → “A local timer will remain useful during a network outage” → disconnect the network and attempt start/cancel/expiry → failure if the system silently accepts the request but never schedules the timer.

Deliver two such chains and one unresolved question. This approach gives you a practical design lesson without presenting speculation as product history.
