# Lecture 42 - OpenAI Agents SDK: Native Sandbox and Durable Agent Harness

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 41](Lecture-41.md) | **Next:** [Lab 01](Lab-01-Research-Agent.md)

---

Agent runtime infrastructure is becoming product infrastructure.

The important signal in OpenAI's April 2026 Agents SDK update is not only that the SDK can call tools.

The important signal is that baseline agent platforms now need:

- filesystem workspaces
- sandbox execution
- shell tools
- patch tools
- manifests
- resumable state
- skills
- MCP
- repo instructions
- approval and interruption surfaces
- provider-aware orchestration

That is the same direction this course has been tracking through OpenClaw:

```text
model
  -> harness
  -> tools
  -> sandbox
  -> state
  -> audit and recovery
```

The model alone is no longer the product.

The harness is becoming part of the product.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why file/tool-oriented long-running agents need a harness.
2. Describe the updated OpenAI Agents SDK sandbox model.
3. Understand manifests as portable workspace contracts.
4. Explain why separating harness from compute improves security, durability, and scale.
5. Understand how shell, `apply_patch`, MCP, skills, and `AGENTS.md` fit into an agent runtime.
6. Compare OpenAI's SDK direction with OpenClaw's Gateway and node architecture.
7. Identify where provider-native harnesses help and where application-owned policy is still required.
8. Design a minimal evaluation plan for sandboxed durable agent workflows.

---

## 1. The shift: from chat calls to agent harnesses

Early LLM integrations looked like:

```text
prompt
  -> model
  -> response
```

Tool-using agents changed that:

```text
prompt
  -> model
  -> tool call
  -> tool result
  -> model
  -> final response
```

Long-running agents add more requirements:

```text
workspace
  -> inspect files
  -> run commands
  -> edit files
  -> checkpoint state
  -> recover after interruption
  -> continue work safely
```

That is no longer just model invocation.

It is a runtime system.

The OpenAI Agents SDK update productizes that runtime layer for OpenAI model workflows.

---

## 2. What changed in the Agents SDK

OpenAI describes the updated Agents SDK as adding a more capable harness for agents that work with documents, files, and systems.

The notable primitives:

- configurable memory
- sandbox-aware orchestration
- filesystem-oriented tools
- MCP tool use
- skills
- `AGENTS.md` instructions
- shell execution
- `apply_patch` file edits
- sandbox manifests
- snapshotting and rehydration
- resumable state after interruptions

These are the same primitives that show up in production coding agents.

The pattern:

```text
agent task
  -> controlled workspace
  -> explicit instructions
  -> bounded tools
  -> durable run state
  -> reviewed artifacts
```

For OpenClaw-style architecture, this validates a key design thesis:

```text
agent systems need a harness, not just a chat loop
```

---

## 3. Native sandbox execution

The updated Agents SDK supports sandbox execution natively.

The sandbox gives the agent a controlled environment where it can:

- read files
- write files
- run code
- use installed dependencies
- produce artifacts
- operate without directly sharing the application host

OpenAI's docs describe `SandboxAgent` as the agent abstraction for this path.

The simplified flow:

```text
Manifest
  -> SandboxAgent
  -> sandbox client
  -> Runner.run(...)
  -> artifacts and result state
```

This is important because many useful agents need a workspace.

Examples:

- audit a data room
- inspect a repository
- patch code
- cluster support exports
- run tests
- generate a report from mounted inputs

Without a sandbox, teams often improvise their own filesystem and execution layer.

That is risky and inconsistent.

---

## 4. Manifest as workspace contract

The SDK's `Manifest` describes the initial sandbox workspace.

It can specify:

- input files
- directories
- repositories
- mounted storage
- output directories
- environment values
- sandbox-local users and groups where supported

The key design idea:

```text
manifest = fresh-session workspace contract
```

It is not necessarily the full source of truth for every live sandbox, because a run may continue from:

- a live sandbox session
- serialized session state
- a snapshot chosen at runtime
- persistent storage

Good manifest design:

- put input artifacts in the manifest
- put output directories in the manifest
- keep paths workspace-relative
- avoid absolute paths and `..` escapes
- scope mounted storage to the task
- keep secrets out of persisted workspace state

This is the sandbox equivalent of an API schema.

It makes the agent's environment explicit and reviewable.

---

## 5. Filesystem tools, shell, and apply_patch

The OpenAI article calls out Codex-like filesystem tools.

Two primitives matter for coding and document agents:

```text
shell:
  run commands in a sandboxed environment

apply_patch:
  make structured file edits
```

This is a major product signal.

Agent platforms are converging on the same low-level tool set:

- inspect files
- search files
- run shell commands
- edit files through patch operations
- test changes
- produce artifacts

Why `apply_patch` matters:

```text
free-form file writes are hard to review
patch operations are easier to audit, replay, and constrain
```

Why shell matters:

```text
many real tasks require execution:
tests, scripts, linters, data transforms, builds, profilers
```

But shell must be policy-gated.

A sandbox reduces blast radius.

It does not remove the need for approvals, allowlists, logs, and network controls.

---

## 6. Skills and AGENTS.md

The updated SDK also standardizes two instruction surfaces that coding agents already use in practice.

### Skills

OpenAI's skills docs describe a skill as a versioned bundle of files plus a `SKILL.md` manifest.

Skills codify procedures:

- style guides
- spreadsheet workflows
- maintenance workflows
- analysis recipes
- company conventions
- multi-step tool use

The model sees skill metadata and can load the full `SKILL.md` instructions when needed.

Security caveat:

```text
skills influence planning, tool use, and command execution
```

Treat them as privileged code-adjacent behavior.

### AGENTS.md

`AGENTS.md` is a repo-local instruction file.

It lets the workspace tell the agent:

- project conventions
- test commands
- code style
- repository-specific rules
- safety notes
- contribution workflow

The important architecture point:

```text
global prompt
  -> agent policy
  -> skill instructions
  -> repo-local instructions
  -> task prompt
```

You need clear priority and trust rules for each layer.

---

## 7. MCP and external tools

MCP appears in the updated SDK as a standard way to connect tools.

The design point:

```text
agent harness
  -> MCP server
  -> external system tools/resources
```

MCP helps decouple the harness from every tool implementation.

But it creates a trust boundary.

Questions to ask:

- Who controls the MCP server?
- What scopes does it expose?
- Can it read secrets?
- Can it write external systems?
- Are tool schemas precise?
- Are destructive actions approval-gated?
- Are outputs treated as untrusted?
- Are calls logged?

MCP is an interface.

It is not a security policy by itself.

---

## 8. Durable execution: state, snapshots, and rehydration

Long-running agents fail.

Containers expire.

Approvals pause.

Networks break.

Runs need human review.

The updated SDK emphasizes durable execution by separating state from the sandbox compute environment.

The official article describes a model where externalized state allows a run to continue after a sandbox fails or expires, using snapshotting and rehydration.

The Agents SDK results docs also expose state surfaces for interruptions and approvals:

```text
interruptions:
  pending tool calls that need a decision

state / to_state():
  resumable snapshot to pass back after approval or rejection
```

The runtime design principle:

```text
compute is replaceable
state is durable
```

This matches the event-sourced session model in Lecture 24b.

If the sandbox dies, the application should not lose:

- user intent
- tool-call history
- pending approvals
- workspace artifacts
- agent ownership
- continuation state

---

## 9. Separating harness from compute

OpenAI explicitly frames harness/compute separation as a security, durability, and scale improvement.

Security:

```text
keep credentials out of environments where model-generated code executes
```

Durability:

```text
if a sandbox dies, restore state into a fresh environment
```

Scale:

```text
use one sandbox or many,
invoke sandboxes only when needed,
route subagents to isolated environments,
parallelize work across containers
```

This should be a default design principle for agent platforms.

Do not put everything in one process with one filesystem and one credential set.

Prefer:

```text
harness:
  owns identity, policy, state, approvals, routing

compute sandbox:
  owns temporary execution, files, dependencies, artifacts
```

The sandbox should be disposable.

The harness should be auditable.

---

## 10. OpenAI SDK versus OpenClaw Gateway

OpenAI Agents SDK and OpenClaw solve overlapping but different layers.

OpenAI Agents SDK:

- provider-native model harness
- sandbox-aware orchestration
- OpenAI model alignment
- OpenAI tool primitives
- Python-first release for new sandbox features
- managed API pricing and tool usage

OpenClaw Gateway:

- local-first control plane
- channels and sessions
- device pairing
- remote nodes
- multi-provider routing
- local tools and app SDKs
- explicit Gateway RPC protocol
- user-owned deployment boundary

Comparison:

```text
OpenAI Agents SDK:
  strong provider-native harness for OpenAI model workflows

OpenClaw:
  broader local control plane for multi-channel, multi-node, user-owned agent operation
```

The useful direction is not necessarily replacement.

It is integration and architectural learning.

An OpenClaw-style system can learn from the SDK's productized sandbox abstractions.

An OpenAI SDK app can learn from OpenClaw's explicit gateway, pairing, threat model, and node architecture.

---

## 11. What becomes baseline infrastructure

This release is a useful market signal.

The following are no longer "advanced extras" for serious agents:

```text
filesystem workspace
sandbox boundary
shell execution
patch-based edits
tool registry
skills
repo-local instructions
state snapshots
human approval interruptions
observability
artifact review
MCP integration
secrets separation
```

If an agent platform lacks these, it is probably a prototype or a narrow wrapper.

Production users will expect:

- reliable continuation
- safe execution
- inspectable artifacts
- bounded tool access
- replayable history
- policy enforcement
- integration with their own storage and sandbox providers

---

## 12. Security implications

Native sandboxing is useful, but it is not a complete security model.

Threats remain:

- malicious skills
- prompt injection through files
- malicious repository instructions
- unsafe shell commands
- artifact exfiltration
- overbroad mounts
- secrets copied into workspace
- MCP server compromise
- tool-result injection
- approval prompt manipulation

Controls:

- minimal manifests
- scoped mounts
- secrets outside sandbox when possible
- explicit approval for high-impact actions
- network egress policy
- skill review
- `AGENTS.md` trust rules
- output artifact review
- audit logs
- snapshot redaction
- deterministic tool policy

Tie this back to Lecture 41:

```text
sandbox is a boundary,
not a trust substitute
```

---

## 13. Evaluation plan

To evaluate a durable sandboxed agent harness, do not only ask whether it can complete one task.

Evaluate:

```text
task success:
  correct final artifact

durability:
  can resume after interruption?

sandbox safety:
  can it only access mounted files?

tool correctness:
  did shell/apply_patch calls match policy?

state quality:
  is continuation faithful after rehydration?

artifact quality:
  are outputs inspectable and reproducible?

security:
  does prompt injection fail to cross authority boundaries?

latency/cost:
  does sandbox startup or snapshotting dominate?
```

Example test:

```text
1. Mount a small repo and task file.
2. Ask the agent to fix a bug.
3. Require it to run tests.
4. Interrupt before a high-impact command.
5. Serialize state.
6. Resume in a fresh sandbox.
7. Verify final patch, logs, and artifacts.
8. Confirm no files outside the manifest were read.
```

That is the right level of test for the new baseline.

---

## 14. Hardware and systems view

For AI hardware engineers, this release matters because it changes workload shape.

Long-running sandboxed agents create:

- bursty inference
- tool-heavy pauses
- background mode runs
- file I/O around model calls
- shell execution outside the model server
- checkpoint and snapshot events
- parallel subagent workloads
- larger context from files and histories

The GPU is not the only bottleneck.

The full system includes:

```text
model serving latency
sandbox startup time
filesystem throughput
network egress
tool runtime
snapshot size
orchestration queueing
approval latency
```

That means agent performance must be measured end-to-end.

Use model metrics, but also collect:

- sandbox lifecycle timing
- tool-call timing
- artifact size
- resume time
- queue delay
- token usage
- error/retry counts

---

## Mini-lab: durable sandbox agent design

Design a small agent app using the updated harness model.

Scenario:

```text
An agent reviews a repository, edits one file, runs tests,
and produces a patch plus a short report.
```

Specify:

```text
Manifest:
  mounted repo:
  output directory:
  task file:
  environment:

Tools:
  shell:
  apply_patch:
  MCP:
  skills:

Policy:
  allowed commands:
  denied commands:
  approval-required commands:
  network policy:

Durability:
  snapshot trigger:
  state serialization:
  rehydration test:

Security:
  untrusted files:
  secrets boundary:
  prompt-injection test:

Evidence:
  logs:
  final patch:
  test output:
  artifact review:
```

Then write the acceptance criteria:

```text
Task passes only if:
  tests pass
  patch is minimal
  state resumes correctly after interruption
  no out-of-scope file access occurs
  artifact report cites exact files changed
```

---

## Key takeaways

- The OpenAI Agents SDK update productizes infrastructure that serious agent systems already need.
- Native sandbox execution gives agents controlled workspaces for files, tools, dependencies, and artifacts.
- Manifests make sandbox workspaces explicit, portable, and reviewable.
- Shell and `apply_patch` are becoming baseline tools for file-oriented agents.
- Skills, MCP, and `AGENTS.md` standardize reusable instructions and tool ecosystems.
- Durable state, interruptions, snapshotting, and rehydration are required for long-running work.
- Separating harness from compute improves security, durability, and scale.
- Provider-native harnesses are useful, but application-owned policy, threat modeling, and evals remain necessary.
- OpenClaw and OpenAI Agents SDK point toward the same baseline: agents need a real runtime around the model.

---

## References

- OpenAI, "The next evolution of the Agents SDK": [https://openai.com/index/the-next-evolution-of-the-agents-sdk/](https://openai.com/index/the-next-evolution-of-the-agents-sdk/)
- OpenAI Agents SDK guide: [https://developers.openai.com/api/docs/guides/agents](https://developers.openai.com/api/docs/guides/agents)
- OpenAI Sandbox Agents guide: [https://developers.openai.com/api/docs/guides/agents/sandboxes](https://developers.openai.com/api/docs/guides/agents/sandboxes)
- OpenAI Results and state guide: [https://developers.openai.com/api/docs/guides/agents/results](https://developers.openai.com/api/docs/guides/agents/results)
- OpenAI Skills guide: [https://developers.openai.com/api/docs/guides/tools-skills](https://developers.openai.com/api/docs/guides/tools-skills)
- Lecture 24b - Event-Sourced Agent State: [Lecture-24b.md](Lecture-24b.md)
- Lecture 29 - Agent Skills: [Lecture-29.md](Lecture-29.md)
- Lecture 41 - OpenClaw Threat Model: [Lecture-41.md](Lecture-41.md)

---

*Next: [Lab 01 - Research Agent](Lab-01-Research-Agent.md)*
