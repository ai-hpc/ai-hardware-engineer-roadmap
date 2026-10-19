# Lecture 19 — Multi-Agent Systems

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 18](Lecture-18.md) | **Next:** [Lecture 20](Lecture-20.md)

---

## Learning Objectives

- Design multi-agent topologies (supervisor, pipeline, peer-to-peer)
- Implement an agent-to-agent communication protocol
- Distinguish a role prompt from a worker with actual tools, evidence and enforced limits
- Prevent coordination failures and infinite loops

---

## 1. Multi-Agent Topologies

| Topology | Pattern | Best for |
|----------|---------|----------|
| **Supervisor** | One orchestrator routes tasks to specialist workers | Complex tasks with clear sub-roles |
| **Pipeline** | Agent A → Agent B → Agent C → output | Sequential processing stages |
| **Peer-to-peer** | Agents communicate directly, consensus-based | Debate, code review, adversarial evaluation |
| **Hierarchical** | Supervisor spawns sub-supervisors | Very large, deeply nested tasks |

---

## 2. A supervisor needs a bounded dispatcher

A supervisor asks a worker to produce an artifact, then evaluates that artifact before deciding the next action. Role prompts alone do not grant tools or evidence. A “researcher” with no retrieval tool is generating from model context, not searching current sources.

Start from the [executed provider dispatcher](Provider-Contract-Lab.md), then extend its allowlist with a `delegate_task` operation. Validate the worker name, task length, total worker-call budget and per-worker model-turn budget before dispatch. Return a result for every accepted call ID; an unknown worker is an error result, not a missing response. Refusal/truncation must not be labeled success. Preserve all assistant blocks and provider continuation state.

For this design exercise, use this application-level result schema:

```json
{
  "task_id": "review-1",
  "worker": "reviewer",
  "status": "completed",
  "artifact_ids": ["review-1.txt"],
  "checks": [{"name": "unit_suite", "passed": false}],
  "claims_verified": false
}
```

“Completed” says the worker returned; `checks` say whether its artifact passed the declared criteria. The supervisor must not equate those fields. A final-answer tool must not bypass outstanding worker results or unresolved failed checks.

**Worked budget:** four workers, each allowed three model turns, plus a supervisor allowed five turns can use up to 17 model requests, before retries. A five-turn supervisor limit alone does not cap the workers' hidden loops. Account for all child work under the same run budget.

No hosted multi-agent run is claimed here. For a fully executed account-free graph with two independent deterministic checks, approval, saved state and failure injection, use [Lab 02](Lab-02-Multi-Agent-Pipeline.md). A model-backed reviewer is a separate extension requiring its own held-out quality assessment.

The remaining snippets are optional provider integration fragments: configure the SDK client and supported model IDs first. They are not account-free programs or executed model-quality evidence. Multiple role prompts using one model may share correlated errors; agreement is not an independent correctness check.

## 3. Pipeline Pattern

Simple **sequential hand-off** between agents:

```python
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()  # Live integration only; requires credentials.

@dataclass
class PipelineContext:
    original_task: str
    artifacts: dict = None

    def __post_init__(self):
        self.artifacts = {}

def build_pipeline(*stages: tuple[str, Callable[[str, dict], str]]):
    """Build a linear agent pipeline."""
    def run(task: str) -> PipelineContext:
        ctx = PipelineContext(original_task=task)
        previous_output = task

        for stage_name, stage_fn in stages:
            print(f"[{stage_name}] running...")
            output = stage_fn(previous_output, ctx.artifacts)
            ctx.artifacts[stage_name] = output
            previous_output = output

        return ctx
    return run

# Define pipeline stages
def outline_stage(task: str, artifacts: dict) -> str:
    response = client.messages.create(
        model="your-fast-model-id",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Create a brief outline for: {task}"}]
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"incomplete worker: {response.stop_reason}")
    return "".join(b.text for b in response.content if b.type == "text")

def draft_stage(outline: str, artifacts: dict) -> str:
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"Write a technical guide based on this outline:\n{outline}"}]
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"incomplete worker: {response.stop_reason}")
    return "".join(b.text for b in response.content if b.type == "text")

def review_stage(draft: str, artifacts: dict) -> str:
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        system="Identify errors and suggest improvements. Be specific.",
        messages=[{"role": "user", "content": draft}]
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"incomplete worker: {response.stop_reason}")
    return "".join(b.text for b in response.content if b.type == "text")

def revise_stage(review: str, artifacts: dict) -> str:
    draft = artifacts.get("draft_stage", "")
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Revise this draft based on the review:\n\nDraft:\n{draft}\n\nReview:\n{review}"
        }]
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"incomplete worker: {response.stop_reason}")
    return "".join(b.text for b in response.content if b.type == "text")

# Build and run pipeline
write_guide = build_pipeline(
    ("outline_stage", outline_stage),
    ("draft_stage", draft_stage),
    ("review_stage", review_stage),
    ("revise_stage", revise_stage),
)

ctx = write_guide("CUDA shared memory optimization techniques")
final_guide = ctx.artifacts["revise_stage"]
print(final_guide[:500])
```

---

## 4. Adversarial Peer-to-Peer (Debate Pattern)

Two agents **debate**; a **judge** picks the winner or synthesizes:

```python
def debate_agents(question: str, rounds: int = 2) -> str:
    """Two agents debate a question, judge synthesizes."""

    def agent_respond(position: str, previous_arguments: list, question: str) -> str:
        history = "\n".join(
            f"{pos}: {arg}" for pos, arg in previous_arguments
        )
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=512,
            system=f"You argue {position}. Be persuasive but technically accurate.",
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\nArguments so far:\n{history}\n\nYour turn:"
            }]
        )
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"incomplete worker: {response.stop_reason}")
        return "".join(b.text for b in response.content if b.type == "text")

    arguments = []
    for round_num in range(rounds):
        pro = agent_respond("in favor", arguments, question)
        arguments.append(("PRO", pro))

        con = agent_respond("against", arguments, question)
        arguments.append(("CON", con))

    # Judge synthesizes
    transcript = "\n\n".join(f"{pos}:\n{arg}" for pos, arg in arguments)
    judge_response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        system="You are an impartial judge. Synthesize the strongest points from both sides.",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nDebate:\n{transcript}\n\nSynthesize the best answer:"
        }]
    )
    if judge_response.stop_reason != "end_turn":
        raise RuntimeError(f"incomplete judge: {judge_response.stop_reason}")
    return "".join(b.text for b in judge_response.content if b.type == "text")

verdict = debate_agents(
    "Should ML inference pipelines use dynamic batching or static batching for latency-sensitive workloads?"
)
print(verdict)
```

---

## 5. Coordination Failures and Fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| **Infinite delegation loop** | Supervisor keeps re-delegating | Track delegation count per task |
| **Context explosion** | Passing full outputs between agents | Summarize agent outputs before passing |
| **Role confusion** | Agent goes outside its specialty | Strong system prompts + validation |
| **Silent failure** | Agent returns empty or error output | Always validate output before passing |

```python
def safe_delegate(agent_fn: Callable, task: str, max_retries: int = 2) -> str:
    """Delegate with retry and validation."""
    for attempt in range(max_retries + 1):
        result = agent_fn(task)
        if result and len(result) > 20:  # basic output validation
            return result
        if attempt < max_retries:
            task = f"{task}\n\n(Previous attempt returned insufficient output. Try again.)"

    return f"[Agent failed after {max_retries + 1} attempts]"
```

---

## Key Takeaways

1. **Supervisor pattern**: use for complex tasks where different specialists add value
2. **Pipeline pattern**: use for sequential stages where each builds on the previous
3. **Debate pattern**: use when you want balanced, adversarial evaluation
4. Keep agent outputs short before passing between agents — summarize if > 500 tokens
5. Always validate agent outputs and implement retry with feedback for failures

---

## Exercises

1. Build a 4-agent code review system: planner → coder → security-reviewer → test-writer
2. Implement a "consensus" multi-agent system where 3 agents vote on the best answer to a question
3. Add a delegation counter to the supervisor — if the same subtask is delegated twice, escalate to human

---

**Previous:** [Lecture 18](Lecture-18.md) | **Next:** [Lecture 20](Lecture-20.md)
