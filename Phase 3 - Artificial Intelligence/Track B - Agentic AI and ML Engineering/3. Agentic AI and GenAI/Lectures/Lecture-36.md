# Lecture 36 - OpenClaw Case Study: Cron, Scheduled Agent Runs, and Automation Reliability

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 35](Lecture-35.md) | **Next:** [Lecture 37](Lecture-37.md)

---

## Why this lecture exists

Cron looks simple from the outside:

> run something at a scheduled time

But in an agent system, scheduling is **not just a timer**.

The scheduled job may need:

- a model
- a session
- a prompt
- tool permissions
- delivery routing
- retries
- logs
- failure notifications
- cleanup

OpenClaw is a useful case study because its cron system is not just classic Unix cron. It is a **scheduler for agent work**.

The better mental model is:

```text
traditional cron:
  schedule -> run command -> exit

OpenClaw cron:
  schedule -> run agent task -> manage session -> deliver output -> retry -> log -> clean up
```

This lecture teaches the **scheduling layer** that sits between "always-on gateway" and "agent execution."

---

## Learning objectives

By the end of this lecture you will be able to:

1. Explain what cron expressions mean.
2. Compare one-shot, interval, and cron schedules.
3. Explain how OpenClaw turns schedules into agent runs.
4. Choose the right session mode for scheduled automation.
5. Understand delivery fallback, failure alerts, retries, logs, and retention.
6. Debug a cron job that fired but produced no visible output.
7. Explain why cron validation belongs before job creation.

---

## 1. Cron from first principles

Cron is a **scheduler**.

Its job is to answer one question:

> when should this task run?

Classic Unix cron has a **background daemon** that reads job definitions and launches shell commands at matching times.

Example:

```cron
0 7 * * *
```

This means:

> run at 07:00 every day

The common 5-field format is:

```text
minute hour day-of-month month day-of-week
```

Written as a diagram:

```text
minute        0-59
| hour        0-23
| | day       1-31
| | | month   1-12
| | | | week  0-6 or names, depending on parser
| | | | |
* * * * *
```

Common examples:

| Expression | Meaning |
|---|---|
| `0 * * * *` | every hour |
| `*/10 * * * *` | every 10 minutes |
| `0 9 * * 1` | every Monday at 09:00 |
| `0 0 1 * *` | first day of every month |
| `0 7 * * *` | every day at 07:00 |

OpenClaw supports 5-field and 6-field cron expressions through Croner. A 6-field expression includes seconds.

---

## 2. The first trap: cron is a language, not just a string

Students often treat cron expressions like harmless text.

They are not.

They are a **small scheduling language** with edge cases:

- timezone interpretation
- seconds vs no seconds
- day-of-month and day-of-week behavior
- parser-specific syntax
- invalid ranges
- impossible dates

OpenClaw's docs call out a specific Croner behavior:

when both day-of-month and day-of-week are non-wildcard, Croner follows Vixie cron-style **OR** logic.

Example:

```cron
0 9 15 * 1
```

Many people expect:

> run at 09:00 on the 15th only when it is Monday

But the usual cron behavior is:

> run at 09:00 on every 15th and at 09:00 on every Monday

That is a system-design lesson:

> schedule syntax must be treated as executable configuration

Invalid or surprising schedules should be **caught early**, before the job becomes durable state.

---

## 3. What OpenClaw cron adds

OpenClaw cron runs inside the Gateway process.

It persists:

- job definitions in `~/.openclaw/cron/jobs.json`
- runtime state in `~/.openclaw/cron/jobs-state.json`
- run history under `~/.openclaw/cron/runs/`

That matters because a scheduled task should **survive a Gateway restart**.

OpenClaw cron also creates background task records, so a scheduled agent run can be inspected as operational work, not just as a hidden timer callback.

The runtime shape is:

```text
[ Gateway scheduler ]
        |
        v
[ Job definition + runtime state ]
        |
        v
[ Agent execution or system event ]
        |
        v
[ Delivery router ]
        |
        v
[ Run log + retry/failure policy ]
```

This is why it is closer to a **small workflow system** than to a plain crontab.

---

## 4. Schedule types in OpenClaw

OpenClaw has three schedule types.

| Kind | CLI flag | Best for |
|---|---|---|
| `at` | `--at` | one-shot reminder or one-time automation |
| `every` | `--every` | fixed interval checks |
| `cron` | `--cron` | calendar-style schedules |

Example one-shot:

```bash
openclaw cron add \
  --name "Calendar check" \
  --at "20m" \
  --session main \
  --system-event "Next heartbeat: check calendar." \
  --wake now
```

Example recurring cron:

```bash
openclaw cron add \
  --name "Morning brief" \
  --cron "0 7 * * *" \
  --tz "America/Los_Angeles" \
  --session isolated \
  --message "Summarize overnight updates." \
  --announce
```

One-shot jobs delete after success by default. Use `--keep-after-run` when you want to preserve the job after it completes.

Recurring top-of-hour schedules may be staggered to reduce load spikes. Use `--exact` for precise cron boundaries or `--stagger 30s` for an explicit spread window.

---

## 5. What actually runs

Traditional cron usually runs a command:

```text
run this shell script at 07:00
```

OpenClaw can run different kinds of scheduled work.

Two important modes:

| Work type | Example | Meaning |
|---|---|---|
| System event | `--system-event "Reminder: check calendar"` | enqueue an event into a session |
| Agent turn | `--message "Summarize overnight updates"` | start an agent run with a prompt |

This distinction matters.

A main-session reminder is more like:

```text
put this reminder into the normal assistant flow
```

An isolated cron job is more like:

```text
start a clean background agent task and send the result somewhere
```

That is why scheduled agent work needs **session design**.

---

## 6. Session modes

OpenClaw cron supports several session targets.

| `--session` value | Behavior | Use it for |
|---|---|---|
| `main` | use the agent's main session | reminders and ordinary wakeups |
| `isolated` | create a fresh `cron:<jobId>` run session | reports, checks, background chores |
| `current` | bind to the active session at job creation | context-aware recurring tasks |
| `session:<id>` | use a persistent named session | workflows that deliberately build history |

The most important one is `isolated`.

An isolated cron run gets a **fresh transcript/session id** for each run. It does not inherit ambient conversation context such as channel routing, queue policy, elevation, origin, or stale runtime bindings.

It may still carry safe preferences such as:

- selected model/auth overrides
- thinking/fast/verbose preferences
- labels

The lesson:

> use isolated sessions when repeated automation should behave like a clean task, not like a long conversation

Good uses:

- daily reports
- monitoring checks
- inbox summaries
- periodic project sweeps

Bad uses:

- jobs that intentionally need accumulated conversation memory
- long-running workflows where yesterday's result should shape today's run

For those, use `session:<id>`.

---

## 7. Delivery is part of the job

A scheduled agent run is **not complete** until someone can see the result or the system intentionally suppresses it.

OpenClaw's delivery modes are:

| Mode | Meaning |
|---|---|
| `announce` | fallback-deliver final text to a chat target if the agent did not send directly |
| `webhook` | POST the finished event payload to a URL |
| `none` | no runner fallback delivery |

CLI mapping:

```bash
--announce     # enable announce fallback
--no-deliver   # delivery.mode = none
```

The important detail is **"fallback."**

For isolated jobs, chat delivery is shared between:

- the agent itself, which may use the `message` tool when a chat route exists
- the runner, which can announce the final reply if the agent did not send it

So `--announce` does not mean "always duplicate the message." It means:

> if the agent did not already send the result to the target, deliver the final reply

This prevents common **silent failures**.

Example:

```bash
openclaw cron add \
  --name "Morning brief" \
  --cron "0 7 * * *" \
  --session isolated \
  --message "Summarize overnight AI and hardware news." \
  --announce \
  --channel telegram \
  --to "-1001234567890"
```

---

## 8. Failure delivery

Scheduled automation must **report failures**.

Otherwise cron turns into:

> the system did nothing, and no one knows why

OpenClaw resolves failure notifications in this order:

1. job-specific `delivery.failureDestination`
2. global `cron.failureDestination`
3. the job's primary announce target, when the job already uses announce delivery

This gives you a safe default:

if a scheduled report normally posts to a chat, failures can fall back to the same target unless you configure a more specific destination.

Configuration shape:

```json5
{
  cron: {
    failureDestination: {
      mode: "announce",
      channel: "last",
      to: "channel:C1234567890"
    }
  }
}
```

This is **operational design**, not UI polish.

For autonomous scheduled jobs, failure delivery is a **control-plane requirement**.

---

## 9. Retry behavior

OpenClaw has two retry stories.

One-shot jobs use configured transient-error retry:

```json5
{
  cron: {
    retry: {
      maxAttempts: 3,
      backoffMs: [30000, 60000, 300000],
      retryOn: ["rate_limit", "overloaded", "network", "timeout", "server_error"]
    }
  }
}
```

Recurring jobs use a recurring failure backoff pattern:

```text
30s -> 1m -> 5m -> 15m -> 60m
```

The backoff **resets** after the next successful run.

This is a major difference from a **simple prompt loop**.

Without backoff:

- a dead local model endpoint can be hammered repeatedly
- provider outages can create request storms
- a broken job can spam users or logs

OpenClaw also has provider preflight behavior for isolated jobs that target local providers such as Ollama or OpenAI-compatible local endpoints. If the endpoint is unreachable, the run can be recorded as `skipped` rather than beginning a doomed model call. Matching dead endpoints are cached briefly to avoid many jobs hitting the same broken local service.

---

## 10. Model selection for isolated cron

Scheduled tasks need **predictable model behavior**.

OpenClaw resolves isolated cron model selection in this order:

1. Gmail-hook model override, when the run came from Gmail and the override is allowed
2. per-job `--model`
3. stored cron-session model override
4. agent/default model selection

Example:

```bash
openclaw cron add \
  --name "Weekly deep analysis" \
  --cron "0 6 * * 1" \
  --session isolated \
  --message "Analyze project progress and risks." \
  --model "opus" \
  --thinking high \
  --announce
```

OpenClaw treats `--model` as a job primary, not as a normal chat-session `/model` override.

That means:

- configured fallback chains can still apply
- per-job `fallbacks` can replace the configured fallback list
- `fallbacks: []` makes the job strict
- invalid or disallowed model refs fail clearly instead of silently using another model

The lesson:

> scheduled jobs should be explicit about model intent, because they may run when no human is watching

---

## 11. Logging and retention

OpenClaw cron keeps run history.

Useful commands:

```bash
openclaw cron list
openclaw cron show <job-id>
openclaw cron runs --id <job-id> --limit 50
```

Run history includes delivery diagnostics such as:

- intended target
- resolved target
- message-tool sends
- fallback use
- delivered/not delivered status

Retention controls:

```json5
{
  cron: {
    sessionRetention: "24h",
    runLog: {
      maxBytes: "2mb",
      keepLines: 2000
    }
  }
}
```

This matters because isolated cron jobs create sessions and transcripts. Without retention, automation creates **state forever**.

The production pattern is:

> keep enough history to debug, prune enough history to avoid state growth

---

## 12. Manual execution

Cron jobs should be testable without waiting for the next scheduled time.

OpenClaw supports:

```bash
openclaw cron run <job-id>
```

This force-runs by default and returns once the run is queued.

Successful responses include:

```json
{ "ok": true, "enqueued": true, "runId": "..." }
```

Then inspect:

```bash
openclaw cron runs --id <job-id> --limit 50
```

Use:

```bash
openclaw cron run <job-id> --due
```

when you want "run only if currently due" behavior.

Manual run support is important because scheduled work must be **debuggable on demand**.

---

## 13. Runtime cleanup

Cron jobs can touch tools and runtimes.

For isolated jobs, OpenClaw includes cleanup behavior such as:

- best-effort browser cleanup for the cron session
- cleanup of bundled MCP runtime instances created for the job
- suppression of stale acknowledgement-only replies
- structured handling of execution denial metadata

This is an important design point.

A scheduled run should not leave **hidden long-lived resources** behind.

If a daily report opens browser tabs or starts MCP child processes, the scheduler should have a cleanup path. Otherwise, scheduled automation slowly becomes **system drift**.

---

## 14. Debugging ladder

Use a **boring command ladder** before guessing.

```bash
openclaw status
openclaw gateway status
openclaw cron status
openclaw cron list
openclaw cron show <job-id>
openclaw cron runs --id <job-id> --limit 20
openclaw system heartbeat last
openclaw logs --follow
openclaw doctor
```

Common cases:

| Symptom | Likely area |
|---|---|
| job never fires | `cron.enabled`, Gateway not running, timezone, bad schedule |
| manual `--due` says not due | schedule is valid but not currently due |
| job runs but no chat output | delivery mode, route resolution, silent token, channel auth |
| local model job is skipped | provider preflight failed |
| blocked command reports failure | tool policy or execution denial |
| repeated failures slow down | recurring retry backoff is working |

The key is to separate:

- schedule problem
- execution problem
- model problem
- delivery problem
- retention/logging problem

---

## 15. Validation belongs before job creation

This is the clean layering rule:

> invalid schedules should fail before durable job creation

Why?

Because once a job is persisted, the system has to answer harder questions:

- should it appear in `cron list`?
- should it be editable?
- should it fail every Gateway startup?
- should the runtime parser throw later?
- should `jobs-state.json` track it?

For `--cron`, validation should happen at the boundary where CLI/API input becomes a job definition.

The same principle applies to:

- bad timezones
- invalid delivery targets
- unsupported session targets
- disallowed models
- invalid tool restrictions

This is a general production lesson:

> configuration validation should be closest to the write boundary, not deferred to the runtime loop

Runtime code can still defend itself, but it should not be the **first place** a user learns that their schedule string was invalid.

---

## 16. Example: morning operations brief

Command:

```bash
openclaw cron add \
  --name "Morning Ops Brief" \
  --cron "0 7 * * 1-5" \
  --tz "America/Los_Angeles" \
  --session isolated \
  --message "Summarize overnight incidents, open deployment risks, and unresolved alerts. Keep it concise and include next actions." \
  --agent ops \
  --model "opus" \
  --thinking high \
  --announce \
  --channel slack \
  --to "channel:C1234567890"
```

What happens:

1. Gateway scheduler computes the next due time.
2. At 07:00 on weekdays, it creates a cron run.
3. The run uses the `ops` agent.
4. The run starts in an isolated session.
5. The model is resolved from the job's model selection.
6. The agent receives the message.
7. If the agent sends directly to the Slack target, fallback announce is skipped.
8. If the agent does not send, the runner announces the final reply.
9. The run is logged.
10. Failures follow job/global/primary failure delivery rules.

This is not just **"scheduled prompting."**

It is scheduled agent work with **routing, isolation, delivery, retry, and auditability**.

---

## 17. Design exercise

Design three jobs for a persistent engineering assistant.

| Job | Schedule | Session | Delivery | Failure policy |
|---|---|---|---|---|
| Morning brief | `0 7 * * 1-5` | `isolated` | Slack announce | failure to ops-alerts |
| Weekly planning summary | `0 16 * * 5` | `session:weekly-planning` | webhook to dashboard | failure to Slack |
| Local model health probe | `*/30 * * * *` | `isolated` | `none` | alert after 3 failures |

For each job, answer:

- Should this job remember previous runs?
- Who should see the output?
- What tools should it be allowed to use?
- What happens if the model provider is down?
- How long should run logs be retained?
- Should a skipped run alert anyone?

That is the **professional version** of "add a cron job."

---

## Key takeaways

- Cron is a scheduling language, not just a text field.
- OpenClaw cron runs inside the Gateway and persists job definitions, runtime state, and run history.
- `--at`, `--every`, and `--cron` solve different scheduling problems.
- Isolated sessions are the default shape for clean recurring agent work.
- Delivery is a first-class part of cron because scheduled results should not disappear.
- Failure destinations, retries, logs, and retention are reliability features, not extras.
- Model selection for cron must be explicit and inspectable.
- Schedule validation should happen before job creation.

---

## References

- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)
- OpenClaw docs:
  - `docs/automation/cron-jobs.md`
  - `docs/cli/cron.md`
  - `docs/gateway/configuration-reference.md`
  - `docs/reference/session-management-compaction.md`

---

*Next: [Lecture 37](Lecture-37.md)*
