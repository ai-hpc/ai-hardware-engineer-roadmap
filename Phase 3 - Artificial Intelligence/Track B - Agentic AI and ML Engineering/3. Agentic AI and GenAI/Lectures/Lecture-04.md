# Lecture 04 — Agent Architecture Patterns

**Track B · Agentic AI & GenAI** | [← Lecture 03](Lecture-03.md) | [Next →](Lecture-05.md)

---

## Learning Objectives

- Understand the major agent design patterns and when to use each
- Implement ReAct from scratch
- Know when to use plan-and-execute vs. reactive agents
- Recognize failure modes and build in safety rails

---

## 1. Pattern Overview

| Pattern | Best for | Weakness |
|---------|----------|----------|
| **ReAct** | Open-ended tasks, tool use | Can get stuck in loops |
| **Plan-and-Execute** | Long multi-step tasks, predictable workflows | Plan goes stale mid-execution |
| **Reflexion** | Tasks with clear success criteria | Expensive — multiple LLM calls per step |
| **LATS (Tree Search)** | Optimization, code generation | Very expensive, complex to implement |

---

## 2. ReAct (Reason + Act)

ReAct interleaves reasoning and action in a loop: **Thought → Action → Observation → Thought → ...**

This is the foundation of most production agents.

```python
import anthropic
import json
from datetime import datetime

client = anthropic.Anthropic()

REACT_SYSTEM = """You are a research agent. You solve problems by thinking step by step
and using tools. Work through the problem systematically.

At each step:
1. Think about what you know and what you need
2. Decide on the best action (tool call or final answer)
3. Use the result to inform your next step

Be systematic. Don't guess — use tools to verify."""

# Tools available to the agent
tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information. Use for facts, recent events, technical specs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate mathematical expressions. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python math expression, e.g. '1024 * 80 / 1000'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "python_repl",
        "description": "Execute Python code and return output. Use for data processing, analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    }
]

def web_search(query: str) -> str:
    # Mock — in production use SerpAPI, Tavily, or Brave Search
    return f"[Search results for '{query}': ... mock data ...]"

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def python_repl(code: str) -> str:
    import io, contextlib
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})
        return buffer.getvalue() or "(no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

TOOL_MAP = {"web_search": web_search, "calculator": calculator, "python_repl": python_repl}

class ReActAgent:
    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.steps = []

    def run(self, task: str) -> str:
        messages = [{"role": "user", "content": task}]
        step = 0

        while step < self.max_steps:
            step += 1
            response = client.messages.create(
                model="your-agent-model-id",
                max_tokens=2048,
                system=REACT_SYSTEM,
                tools=tools,
                messages=messages
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = TOOL_MAP[block.name](**block.input)
                        self.steps.append({
                            "step": step,
                            "tool": block.name,
                            "input": block.input,
                            "output": result[:200]  # truncate for logging
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                messages.append({"role": "user", "content": tool_results})

        return "Agent reached max steps without completing."

agent = ReActAgent(max_steps=15)
answer = agent.run("What is the memory bandwidth of an H100 SXM5 in GB/s? How many times faster is it than DDR5-5600?")
print(answer)
print(f"\nSteps taken: {len(agent.steps)}")
```

---

## 3. Plan-and-Execute

For complex multi-step tasks, separate planning from execution. The planner creates a task list; the executor works through it.

```python
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"

@dataclass
class Task:
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    depends_on: list[int] = field(default_factory=list)

class PlanAndExecuteAgent:
    def __init__(self):
        self.plan: list[Task] = []

    def plan_task(self, goal: str) -> list[Task]:
        """Generate a structured plan for the goal."""
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            system="""Create a numbered execution plan. Return JSON array of tasks.
Each task: {"id": int, "description": str, "depends_on": [int]}
Keep tasks atomic — one action each. Maximum 10 tasks.""",
            messages=[{"role": "user", "content": f"Goal: {goal}"}]
        )

        raw = response.content[0].text.strip().strip("```json").strip("```")
        task_dicts = json.loads(raw)
        return [Task(**t) for t in task_dicts]

    def execute_task(self, task: Task, context: str) -> str:
        """Execute a single task given accumulated context."""
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            system="Execute the given task. Be concise. Return only the result.",
            tools=tools,
            messages=[{
                "role": "user",
                "content": f"Task: {task.description}\n\nContext from previous steps:\n{context}"
            }]
        )
        # Handle tool use inline
        messages = [{"role": "user", "content": f"Task: {task.description}\n\nContext:\n{context}"}]
        while True:
            response = client.messages.create(
                model="your-agent-model-id",
                max_tokens=1024,
                system="Execute the given task. Be concise. Return only the result.",
                tools=tools,
                messages=messages
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "end_turn":
                return next((b.text for b in response.content if hasattr(b, "text")), "")
            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = TOOL_MAP[block.name](**block.input)
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})

    def run(self, goal: str) -> str:
        print(f"Planning for: {goal}")
        self.plan = self.plan_task(goal)
        print(f"Plan ({len(self.plan)} steps):")
        for t in self.plan:
            print(f"  {t.id}. {t.description}")

        context_parts = []
        for task in self.plan:
            # Check dependencies
            deps_done = all(
                any(t.id == dep and t.status == TaskStatus.DONE for t in self.plan)
                for dep in task.depends_on
            )
            if task.depends_on and not deps_done:
                task.status = TaskStatus.FAILED
                task.result = "Dependency failed"
                continue

            task.status = TaskStatus.IN_PROGRESS
            context = "\n".join(context_parts)
            task.result = self.execute_task(task, context)
            task.status = TaskStatus.DONE
            context_parts.append(f"Step {task.id} ({task.description}): {task.result}")
            print(f"  ✓ Step {task.id} done")

        # Final synthesis
        all_results = "\n".join(context_parts)
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Goal: {goal}\n\nResults:\n{all_results}\n\nWrite a concise final answer."
            }]
        )
        return response.content[0].text
```

---

## 4. Reflexion — Self-Critique Loop

After completing a task, the agent evaluates its own output and retries if needed.

```python
def reflexion_agent(task: str, max_retries: int = 3) -> str:
    """Agent that self-critiques and retries until satisfied."""
    attempt = ""

    for i in range(max_retries):
        # Generate attempt
        attempt_response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": task if i == 0 else f"{task}\n\nPrevious attempt:\n{attempt}\n\nTry again, fixing the issues."
            }]
        )
        attempt = attempt_response.content[0].text

        # Self-critique
        critique_response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=512,
            system="You are a strict quality reviewer. Identify any errors, gaps, or improvements needed.",
            messages=[{
                "role": "user",
                "content": f"Task: {task}\n\nAttempt:\n{attempt}\n\nIs this correct and complete? Reply with 'PASS' if good, or list specific issues."
            }]
        )
        critique = critique_response.content[0].text

        if "PASS" in critique:
            print(f"✓ Passed critique on attempt {i+1}")
            return attempt

        print(f"✗ Attempt {i+1} failed critique: {critique[:100]}...")

    return attempt  # Return best attempt after max retries
```

---

## 5. Failure Mode Prevention

| Failure mode | Symptom | Prevention |
|-------------|---------|-----------|
| **Infinite loop** | Agent keeps calling same tool | `max_steps` counter |
| **Tool hallucination** | Model invents tool parameters | Strict schema + `enum` constraints |
| **Context overflow** | Responses degrade as context grows | Summarize history periodically |
| **Stale plan** | Plan-and-execute follows outdated plan | Re-plan if environment changes |
| **Runaway cost** | Hundreds of API calls | Budget limit + step counter |

```python
class SafeAgentWrapper:
    def __init__(self, agent_fn, max_steps=20, budget_usd=1.0):
        self.agent_fn = agent_fn
        self.max_steps = max_steps
        self.budget = budget_usd
        self.total_cost = 0.0
        self.step_count = 0

    def __call__(self, *args, **kwargs):
        if self.step_count >= self.max_steps:
            raise RuntimeError(f"Agent exceeded {self.max_steps} steps")
        if self.total_cost >= self.budget:
            raise RuntimeError(f"Agent exceeded ${self.budget:.2f} budget")
        self.step_count += 1
        return self.agent_fn(*args, **kwargs)
```

---

## Key Takeaways

1. **ReAct** is the default pattern — simple, effective, handles most tasks
2. **Plan-and-Execute** for predictable multi-step workflows where you want visibility
3. **Reflexion** when quality matters more than speed — adds self-correction
4. Always set `max_steps` and a budget limit — agents can loop indefinitely without them
5. Log every tool call for debugging — agent failures are hard to diagnose without a trace

---

## Exercises

1. Implement a ReAct agent that solves a 3-step data analysis task (fetch data → process → visualize).
2. Add context compression to `ReActAgent`: after every 5 steps, summarize the conversation history.
3. Build a `Reflexion` agent for code generation — it runs the code and uses the output/errors as critique.

---

**Previous:** [Lecture 03](Lecture-03.md) | **Next:** [Lecture 05 — Memory Systems](Lecture-05.md)
