# Lecture 19 — Multi-Agent Systems

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 18](Lecture-18.md) | **Next:** [Lecture 20](Lecture-20.md)

---

## Learning Objectives

- Design multi-agent topologies (supervisor, pipeline, peer-to-peer)
- Implement an agent-to-agent communication protocol
- Use CrewAI for role-based agent teams
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

## 2. Supervisor Pattern (from scratch)

```python
import anthropic
import json
from typing import Callable

client = anthropic.Anthropic()

# ── Worker agents ─────────────────────────────────────────────

def researcher_agent(task: str) -> str:
    """Gathers information and facts."""
    response = client.messages.create(
        model="your-fast-model-id",
        max_tokens=1024,
        system="You are a research specialist. Find relevant facts and technical details. Be thorough.",
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

def coder_agent(task: str) -> str:
    """Writes and reviews code."""
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=2048,
        system="You are a senior software engineer. Write clean, well-commented code with examples.",
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

def reviewer_agent(task: str) -> str:
    """Reviews and critiques work."""
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        system="You are a strict technical reviewer. Identify bugs, edge cases, and improvements.",
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

def writer_agent(task: str) -> str:
    """Writes documentation and explanations."""
    response = client.messages.create(
        model="your-fast-model-id",
        max_tokens=1024,
        system="You are a technical writer. Create clear, well-structured documentation.",
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

WORKERS: dict[str, Callable[[str], str]] = {
    "researcher": researcher_agent,
    "coder": coder_agent,
    "reviewer": reviewer_agent,
    "writer": writer_agent,
}

# ── Supervisor ────────────────────────────────────────────────

SUPERVISOR_TOOLS = [
    {
        "name": "delegate_task",
        "description": "Delegate a subtask to a specialist agent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": list(WORKERS.keys()),
                    "description": "Which specialist to use"
                },
                "task": {
                    "type": "string",
                    "description": "Specific task description for this agent"
                }
            },
            "required": ["agent", "task"]
        }
    },
    {
        "name": "finish",
        "description": "Return the final answer to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Final synthesized answer"}
            },
            "required": ["answer"]
        }
    }
]

SUPERVISOR_SYSTEM = """You are a supervisor managing a team of specialist agents:
- researcher: gathers information and facts
- coder: writes and explains code
- reviewer: identifies bugs and improvements
- writer: creates documentation

Break the user's task into subtasks, delegate each to the right specialist,
then synthesize the results. Use 'finish' when done."""

def supervisor_agent(user_task: str) -> str:
    messages = [{"role": "user", "content": user_task}]
    results = {}

    for _ in range(20):  # max iterations
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            system=SUPERVISOR_SYSTEM,
            tools=SUPERVISOR_TOOLS,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "finish":
                return block.input["answer"]

            elif block.name == "delegate_task":
                agent_name = block.input["agent"]
                subtask = block.input["task"]
                print(f"  → delegating to {agent_name}: {subtask[:60]}...")

                agent_result = WORKERS[agent_name](subtask)
                results[agent_name] = agent_result

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"[{agent_name}]: {agent_result}"
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "Supervisor did not complete within iteration limit."

# Test
result = supervisor_agent(
    "Build a Python class for a simple LRU cache. Include code, review, and a docstring."
)
print(result)
```

---

## 3. Pipeline Pattern

Simple **sequential hand-off** between agents:

```python
from dataclasses import dataclass

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
    return response.content[0].text

def draft_stage(outline: str, artifacts: dict) -> str:
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"Write a technical guide based on this outline:\n{outline}"}]
    )
    return response.content[0].text

def review_stage(draft: str, artifacts: dict) -> str:
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        system="Identify errors and suggest improvements. Be specific.",
        messages=[{"role": "user", "content": draft}]
    )
    return response.content[0].text

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
    return response.content[0].text

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
        return response.content[0].text

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
    return judge_response.content[0].text

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
