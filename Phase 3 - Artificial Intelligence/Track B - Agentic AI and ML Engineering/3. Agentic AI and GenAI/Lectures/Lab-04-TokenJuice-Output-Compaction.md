# Lab 04 — TokenJuice Output Compaction for Terminal-Heavy Agents

**Track B · Agentic AI & GenAI** | [← Index](README.md) | [Previous → Lab 03](Lab-03-Production-RAG.md)

---

## Overview

This is a small, fun systems lab.

You will use [TokenJuice](https://github.com/vincentkoc/tokenjuice) to compact noisy terminal output before it enters an agent transcript.

The core idea is simple:

```text
run command normally
  -> observe output
  -> deterministically reduce prompt-facing text
  -> keep raw output available only when explicitly needed
```

TokenJuice is not an LLM summarizer.

It is a rule-driven output reducer for terminal-heavy workflows such as:

- `git status`
- `pnpm test`
- `docker build`
- `rg --files`
- `pnpm --help`
- build logs
- lint output
- package-manager noise

The lab goal is to measure whether compaction improves agent workflow quality without hiding critical debugging information.

**Estimated time:** 45-75 minutes

**Difficulty:** Beginner to intermediate

---

## Why this matters

Agent workflows waste context on terminal output.

Example:

```text
agent runs pnpm test
  -> receives 800 lines
  -> only 25 lines matter
  -> transcript fills with noise
  -> next turn has less useful context
  -> agent reruns commands because it missed the important part
```

TokenJuice attacks that waste by making terminal output leaner.

The key design properties:

- command semantics stay untouched
- reduction happens after execution
- rules are inspectable JSON
- raw output is available through explicit bypasses
- host integrations stay thin wrappers around the same reducer
- project rules can override built-in rules

This is the right kind of "boring" infrastructure for agents.

---

## Learning objectives

By the end of this lab, you should be able to:

1. Explain why terminal output is a token-budget problem for agents.
2. Use `tokenjuice reduce` to compact existing logs.
3. Use `tokenjuice wrap` to run a command and compact its observed output.
4. Use `--raw`, `--full`, and artifact storage when exact bytes matter.
5. Inspect machine-facing output with `reduce-json`.
6. Understand the rule precedence model.
7. Write a small project-specific reducer.
8. Decide when compaction is safe and when it is dangerous.
9. Connect TokenJuice-style reducers to OpenClaw, Codex, Claude Code, and other agent harnesses.

---

## Step 0 — Safety model

Before installing hooks into an agent, understand the safety boundary.

TokenJuice should not:

- rewrite commands silently
- pretend lossy output is complete
- summarize with an LLM
- hide raw output when exact bytes are required
- compact exact file-content reads such as `cat`, `sed`, `head`, or `tail`

Good use:

```text
inventory commands
test logs
build logs
package-manager output
lint summaries
```

Risky use:

```text
security logs
binary dumps
exact config file reads
one-off debugging where every byte matters
mixed shell sequences with side effects
```

Use raw mode when necessary:

```bash
tokenjuice wrap --raw -- git status
tokenjuice wrap --full -- pnpm --help
```

---

## Step 1 — Install TokenJuice

Install with your preferred package manager:

```bash
npm install -g tokenjuice
```

or:

```bash
pnpm add -g tokenjuice
```

or, if using Homebrew:

```bash
brew tap vincentkoc/tap
brew install tokenjuice
```

Verify:

```bash
tokenjuice --version
tokenjuice --help
```

If you do not want a global install, use a scratch project:

```bash
mkdir tokenjuice-lab
cd tokenjuice-lab
pnpm init
pnpm add -D tokenjuice
pnpm exec tokenjuice --version
```

For the rest of the lab, replace `tokenjuice` with `pnpm exec tokenjuice` if using a local install.

---

## Step 2 — Create noisy sample output

Create a lab folder:

```bash
mkdir -p tokenjuice-lab/logs
cd tokenjuice-lab
```

Create a fake test log:

```bash
cat > logs/test-output.txt <<'EOF'
RUN  v3.2.4 /repo

stdout | packages/core/test/reducer.test.ts > reducer keeps failure detail
loading config from /repo/tokenjuice.config.json
loading built-in rules from src/rules
loading user rules from ~/.config/tokenjuice/rules
loading project rules from .tokenjuice/rules

✓ packages/core/test/classify.test.ts (28 tests) 132ms
✓ packages/core/test/command.test.ts (42 tests) 188ms
✓ packages/core/test/artifacts.test.ts (17 tests) 96ms
✓ packages/hosts/test/codex.test.ts (18 tests) 120ms
✓ packages/hosts/test/claude-code.test.ts (21 tests) 140ms

stderr | packages/core/test/reducer.test.ts > reducer keeps failure detail
AssertionError: expected reducer to preserve "exit code 2"
  at test/core/reducer.test.ts:88:15
  at runTest test-runner.ts:55:7

FAIL packages/core/test/reducer.test.ts > reducer keeps failure detail
Expected preserved lines:
  exit code 2
  src/rules/fixtures/pnpm-test-failure.txt
Received:
  exit code omitted

Test Files  1 failed | 5 passed (6)
Tests       1 failed | 126 passed (127)
Duration    2.8s
EOF
```

Check raw size:

```bash
wc -l logs/test-output.txt
wc -c logs/test-output.txt
```

---

## Step 3 — Reduce existing text

Run:

```bash
tokenjuice reduce logs/test-output.txt
```

Also compare with a pipe:

```bash
cat logs/test-output.txt | tokenjuice reduce
```

Record:

```text
raw line count:
reduced line count:
what details were preserved:
what details were removed:
```

The important question is not "was it shorter?"

The important question is:

```text
Would an agent still know what to do next?
```

For a failing test, the reducer should preserve enough detail to identify:

- failing file
- failing test name
- expected versus received clue
- stack frame or source location
- total failure count

---

## Step 4 — Wrap real commands

Now run commands through TokenJuice directly.

Start with safe inventory:

```bash
tokenjuice wrap -- git status --short
tokenjuice wrap -- git ls-files
```

Try a noisy help command:

```bash
tokenjuice wrap -- pnpm --help
```

Try a command that may fail:

```bash
tokenjuice wrap -- bash -lc 'echo "compile start"; echo "error TS2322: Type string is not assignable"; exit 2'
```

Observe:

- output is compacted after execution
- the command still runs normally
- non-zero exits should preserve more detail than success noise

This is the key distinction:

```text
TokenJuice is not a shell replacement.
It is an output reducer around normal command execution.
```

---

## Step 5 — Use raw and full bypasses

Compaction is useful until it is not.

Run:

```bash
tokenjuice wrap --raw -- pnpm --help
tokenjuice wrap --full -- git status
```

Use raw/full modes when:

- exact text is required
- you are debugging a reducer
- a fixture needs exact expected output
- the agent needs every line of a file-like response
- the reduced output hides a relevant middle section

Write this rule into your own agent practice:

```text
Default to compact for noisy terminal output.
Use raw when exact bytes are part of the task.
```

---

## Step 6 — Store and inspect artifacts

TokenJuice can store raw output as an artifact when explicitly requested.

Run:

```bash
tokenjuice wrap --store -- bash -lc 'for i in $(seq 1 120); do echo "line $i"; done'
```

List artifacts:

```bash
tokenjuice ls
```

Inspect one:

```bash
tokenjuice cat <artifact-id>
```

Why this matters:

```text
prompt-facing output can be compact
while raw output remains recoverable by explicit action
```

That is the right compromise for agent transcripts.

---

## Step 7 — Inspect machine-facing JSON

Host integrations need stable machine output.

Create a tool payload:

```bash
cat > payload.json <<'EOF'
{
  "toolName": "exec",
  "command": "pnpm test",
  "argv": ["pnpm", "test"],
  "combinedText": "RUN  v3.2.4 /repo\nFAIL packages/core/test/reducer.test.ts > reducer keeps failure detail\nAssertionError: expected reducer to preserve exit code 2\nTest Files 1 failed | 5 passed\nTests 1 failed | 126 passed\n",
  "exitCode": 1
}
EOF
```

Run:

```bash
tokenjuice reduce-json payload.json
```

Look for:

- reduced output text
- command classification
- savings information
- whether failure context was preserved

This is the surface host adapters should use.

Human-facing CLIs can be flexible.

Adapter protocols should be boring and structured.

---

## Step 8 — Verify rules

TokenJuice rules are JSON.

Built-in rules live in the package. Overrides can live in:

```text
~/.config/tokenjuice/rules
.tokenjuice/rules
```

Rule precedence:

```text
built-in rules
  -> user rules
  -> project rules
```

Later layers override earlier layers by rule id.

Run:

```bash
tokenjuice verify
```

If fixtures are available:

```bash
tokenjuice verify --fixtures
```

This should check:

- JSON parses
- schema shape is valid
- regexes compile
- duplicate ids are rejected inside the same layer
- fixture expectations still match reducers

---

## Step 9 — Write a tiny project reducer

Create a project override folder:

```bash
mkdir -p .tokenjuice/rules
```

Create a reducer for a fake hardware bring-up log:

```bash
cat > logs/otbr-output.txt <<'EOF'
Apr 20 09:34:20 ubuntu otbr-agent[10839]: Attach attempt 8, AnyPartition
Apr 20 09:34:20 ubuntu otbr-agent[10839]: Send Parent Request to routers
Apr 20 09:34:22 ubuntu otbr-agent[10839]: Attach attempt 8 unsuccessful, will try again in 32.128 seconds
Apr 20 09:34:46 ubuntu otbr-agent[11149]: Running 0.3.0-1e957ca
Apr 20 09:34:46 ubuntu otbr-agent[11149]: Thread version: 1.4.0
Apr 20 09:34:46 ubuntu otbr-agent[11149]: Radio URL: spinel+hdlc+uart:///dev/ttyTHS1?uart-baudrate=460800
Apr 20 09:34:46 ubuntu otbr-agent[11149]: InitMulticastRouterSock() at multicast_routing.cpp:227: Protocol not available
Apr 20 09:34:47 ubuntu otbr-agent[11149]: TrelDiscoverer: DNS-SD service registered successfully
EOF
```

Create a project rule:

```bash
cat > .tokenjuice/rules/otbr-journal.json <<'EOF'
{
  "id": "otbr/journal",
  "family": "otbr-journal",
  "match": {
    "commandIncludes": ["journalctl", "otbr-agent"]
  },
  "transforms": {
    "trimEmptyEdges": true,
    "dedupeAdjacent": true
  },
  "filters": {
    "keepPatterns": [
      "Attach attempt",
      "unsuccessful",
      "Radio URL",
      "Thread version",
      "Protocol not available",
      "InitMulticastRouterSock"
    ],
    "skipPatterns": [
      "DNS-SD service registered successfully"
    ]
  },
  "summarize": {
    "head": 20,
    "tail": 8
  },
  "failure": {
    "preserveOnFailure": true,
    "head": 24,
    "tail": 12
  },
  "counters": [
    {
      "name": "attach_attempts",
      "pattern": "Attach attempt"
    },
    {
      "name": "kernel_missing_mroute",
      "pattern": "Protocol not available"
    }
  ]
}
EOF
```

Verify:

```bash
tokenjuice verify
```

Test with JSON input:

```bash
cat > otbr-payload.json <<'EOF'
{
  "toolName": "exec",
  "command": "journalctl -u otbr-agent -n 80 --no-pager",
  "argv": ["journalctl", "-u", "otbr-agent", "-n", "80", "--no-pager"],
  "combinedText": "",
  "exitCode": 0
}
EOF
```

Inject the log content:

```bash
python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("otbr-payload.json").read_text())
payload["combinedText"] = Path("logs/otbr-output.txt").read_text()
Path("otbr-payload.json").write_text(json.dumps(payload, indent=2))
PY
```

Run:

```bash
tokenjuice reduce-json otbr-payload.json
```

If the rule does not match, inspect the docs and simplify the `match` block.

The learning goal is not to memorize the exact schema.

The learning goal is to understand that project-local reducers can encode domain-specific signal.

---

## Step 10 — Measure savings

Create a simple measurement script:

```bash
cat > measure.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

file="${1:?usage: ./measure.sh <file>}"
raw_bytes="$(wc -c < "$file" | tr -d ' ')"
reduced="$(tokenjuice reduce "$file")"
reduced_bytes="$(printf "%s" "$reduced" | wc -c | tr -d ' ')"

python3 - <<PY
raw = int("$raw_bytes")
reduced = int("$reduced_bytes")
savings = 0 if raw == 0 else (1 - reduced / raw) * 100
print(f"raw bytes:     {raw}")
print(f"reduced bytes: {reduced}")
print(f"savings:       {savings:.1f}%")
PY
EOF

chmod +x measure.sh
```

Run:

```bash
./measure.sh logs/test-output.txt
./measure.sh logs/otbr-output.txt
```

For this lab, good savings are not enough.

You must also answer:

```text
Did the compacted output preserve the next action?
```

For the OTBR example, the compacted output should still preserve:

- attach attempts failed
- Thread version
- Radio URL
- `Protocol not available`
- likely kernel multicast routing issue

If those disappear, the reducer is too aggressive.

---

## Step 11 — Connect to an agent host

TokenJuice supports several host integrations, including Codex CLI, Claude Code, Cursor, OpenCode, pi, and OpenClaw.

For Codex CLI:

```bash
tokenjuice install codex
tokenjuice doctor codex
```

For Claude Code:

```bash
tokenjuice install claude-code
tokenjuice doctor claude-code
```

For aggregate hook state:

```bash
tokenjuice doctor hooks
```

For OpenClaw, TokenJuice support is bundled on the OpenClaw side. The upstream docs say to enable the plugin instead of running `tokenjuice install openclaw`:

```bash
openclaw config set plugins.entries.tokenjuice.enabled true
```

This requires OpenClaw `2026.4.22` or newer.

Do not install host hooks on machines where you do not understand the hook behavior.

Use `doctor` first, and keep a rollback path.

---

## Step 12 — Evaluate the lab

Build a small table:

| Command or log | Raw bytes | Reduced bytes | Savings | Did it preserve next action? |
|---|---:|---:|---:|---|
| `logs/test-output.txt` |  |  |  |  |
| `logs/otbr-output.txt` |  |  |  |  |
| `git ls-files` |  |  |  |  |
| `pnpm --help` |  |  |  |  |

Then write a short conclusion:

```text
TokenJuice is useful for:

TokenJuice is risky for:

The reducer I would add next is:

The command family I would always keep raw is:
```

---

## What you should learn

TokenJuice is funny because the product language is playful: "token weight loss."

But the underlying systems idea is serious:

```text
agent productivity depends on transcript hygiene.
```

Terminal-heavy agents need:

- less noise
- preserved failure clues
- deterministic behavior
- explicit raw bypass
- recoverable artifacts
- inspectable rules
- measurable savings

This is a small tool, but it touches a real production problem.

---

## Extensions

1. **Add a reducer for Jetson audio logs** — Preserve `arecord`, `aplay`, APE card, I2S2, and ALSA control lines.
2. **Add a reducer for CUDA profiler output** — Preserve kernel names, occupancy, memory throughput, and warnings.
3. **Add fixture tests** — Create before/after expected outputs for your project reducer.
4. **Add OpenClaw workflow notes** — Define when your agents should use raw output versus compacted output.
5. **Compare with LLM summarization** — Run the same logs through an LLM summary and compare determinism, cost, and missed details.

---

## References

- TokenJuice repository: [https://github.com/vincentkoc/tokenjuice](https://github.com/vincentkoc/tokenjuice)
- TokenJuice spec: [https://github.com/vincentkoc/tokenjuice/blob/main/docs/spec.md](https://github.com/vincentkoc/tokenjuice/blob/main/docs/spec.md)
- TokenJuice rules: [https://github.com/vincentkoc/tokenjuice/blob/main/docs/rules.md](https://github.com/vincentkoc/tokenjuice/blob/main/docs/rules.md)
- TokenJuice integration playbook: [https://github.com/vincentkoc/tokenjuice/blob/main/docs/integration-playbook.md](https://github.com/vincentkoc/tokenjuice/blob/main/docs/integration-playbook.md)

---

*End of Lab 04. Return to the [Track Index](README.md).*
