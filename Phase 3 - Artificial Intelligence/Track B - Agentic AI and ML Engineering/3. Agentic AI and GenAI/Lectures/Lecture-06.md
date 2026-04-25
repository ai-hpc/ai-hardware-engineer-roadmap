# Lecture 06 — LangGraph: Stateful Workflows

**Track B · Agentic AI & GenAI** | [← Lecture 05](Lecture-05.md) | [Next →](Lecture-07.md)

---

## Learning Objectives

- Understand why graph-based orchestration beats raw loops for complex agents
- Build nodes, edges, and state schemas with LangGraph
- Implement conditional branching and cycles
- Add checkpointing and human-in-the-loop interrupts

---

## 1. Why LangGraph?

Raw `while` loops work for simple ReAct agents, but break down when you need:

- **Branching** — different paths based on tool results
- **Parallel execution** — run multiple steps concurrently
- **Checkpointing** — pause, resume, replay
- **Human-in-the-loop** — wait for approval before continuing
- **Cycles with exit conditions** — retry loops, reflection cycles

LangGraph models agents as a **directed graph** where nodes are functions and edges are transitions.

---

## 2. Core Concepts

```
State: TypedDict that flows between nodes
Node:  function(state) → updated_state
Edge:  connection from node A to node B (static or conditional)
```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define state
class AgentState(TypedDict):
    messages: list           # conversation history
    task: str                # original task
    plan: list[str]          # steps to execute
    current_step: int        # which step we're on
    results: list[str]       # accumulated results
    error_count: int         # for retry logic

# 2. Define nodes (functions that transform state)
def planner_node(state: AgentState) -> AgentState:
    """Generate a plan from the task."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Break this task into 3-5 steps (numbered list only):\n{state['task']}"
        }]
    )
    lines = [l.strip() for l in response.content[0].text.split("\n") if l.strip()]
    plan = [l.split(". ", 1)[-1] for l in lines if l[0].isdigit()]

    return {**state, "plan": plan, "current_step": 0}

def executor_node(state: AgentState) -> AgentState:
    """Execute the current step of the plan."""
    import anthropic
    client = anthropic.Anthropic()

    step = state["plan"][state["current_step"]]
    context = "\n".join(state["results"])

    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Execute this step:\n{step}\n\nContext:\n{context}"
        }]
    )

    result = response.content[0].text
    new_results = state["results"] + [f"Step {state['current_step']+1}: {result}"]

    return {**state, "results": new_results, "current_step": state["current_step"] + 1}

def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesize all results into a final answer."""
    import anthropic
    client = anthropic.Anthropic()

    all_results = "\n".join(state["results"])
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Task: {state['task']}\n\nResults:\n{all_results}\n\nWrite the final answer."
        }]
    )

    final = response.content[0].text
    return {**state, "results": state["results"] + [f"FINAL: {final}"]}

# 3. Define routing logic
def should_continue(state: AgentState) -> str:
    """Decide whether to execute more steps or synthesize."""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

# 4. Build the graph
def build_agent_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")

    # Conditional edge: after executor, go back or synthesize
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "execute": "executor",   # loop back
            "synthesize": "synthesizer"
        }
    )
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

# 5. Run it
app = build_agent_graph()
result = app.invoke({
    "task": "Explain the memory hierarchy in NVIDIA H100 and its implications for kernel optimization",
    "messages": [],
    "plan": [],
    "current_step": 0,
    "results": [],
    "error_count": 0
})
print(result["results"][-1])  # FINAL answer
```

---

## 3. Checkpointing (Pause & Resume)

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory checkpointing (for testing)
memory_checkpointer = MemorySaver()

# Persistent SQLite checkpointing (for production)
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Compile with checkpointer
app = workflow.compile(checkpointer=sqlite_checkpointer)

# Each run needs a thread_id to track its checkpoint
config = {"configurable": {"thread_id": "task-001"}}

# First run
result = app.invoke(initial_state, config=config)

# Resume from checkpoint (after a crash or restart)
result = app.invoke(None, config=config)  # None = resume from last checkpoint

# View checkpoint state
state = app.get_state(config)
print(state.values)  # Current state
print(state.next)    # Next node to execute
```

---

## 4. Human-in-the-Loop

Interrupt the graph before sensitive operations and wait for human approval.

```python
from langgraph.graph import StateGraph, END, START

class ReviewState(TypedDict):
    code: str
    review_comments: str
    approved: bool
    final_code: str

def generate_code_node(state: ReviewState) -> ReviewState:
    """Generate code based on a task."""
    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a CUDA kernel for matrix multiplication."}]
    )
    return {**state, "code": response.content[0].text}

def human_review_node(state: ReviewState) -> ReviewState:
    """This node will be interrupted — human reviews here."""
    # In a real app, this would send a notification and wait
    # The graph pauses here until .update_state() is called
    print("\n--- Code ready for review ---")
    print(state["code"][:500])
    return state  # State unchanged — human will update it

def apply_review_node(state: ReviewState) -> ReviewState:
    if not state["approved"]:
        # Regenerate with feedback
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Fix this code based on review:\n{state['code']}\nFeedback: {state['review_comments']}"
            }]
        )
        return {**state, "final_code": response.content[0].text}
    return {**state, "final_code": state["code"]}

# Build with interrupt
workflow = StateGraph(ReviewState)
workflow.add_node("generate", generate_code_node)
workflow.add_node("review", human_review_node)
workflow.add_node("apply", apply_review_node)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
workflow.add_edge("review", "apply")
workflow.add_edge("apply", END)

checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["review"]  # Pause before human_review_node
)

config = {"configurable": {"thread_id": "review-001"}}

# Run until interrupt
app.invoke({"code": "", "review_comments": "", "approved": False, "final_code": ""}, config=config)
print("Graph paused at review node.")

# Human reviews and updates state
app.update_state(config, {"approved": True, "review_comments": "Looks good"})

# Resume
final = app.invoke(None, config=config)
print("Final code:", final["final_code"][:200])
```

---

## 5. Parallel Execution (Fan-Out / Fan-In)

```python
# Fan-out: run multiple nodes in parallel
workflow.add_node("research_gpu", research_gpu_node)
workflow.add_node("research_cpu", research_cpu_node)
workflow.add_node("research_memory", research_memory_node)
workflow.add_node("synthesize", synthesize_node)

workflow.add_edge("start", "research_gpu")
workflow.add_edge("start", "research_cpu")
workflow.add_edge("start", "research_memory")

# Fan-in: all parallel nodes must complete before synthesize
workflow.add_edge("research_gpu", "synthesize")
workflow.add_edge("research_cpu", "synthesize")
workflow.add_edge("research_memory", "synthesize")
```

---

## Key Takeaways

1. LangGraph is a graph where nodes = functions, edges = transitions
2. State is a `TypedDict` that flows through every node — always return `{**state, ...updated_fields}`
3. Conditional edges enable branching and loops — the backbone of ReAct in LangGraph
4. Checkpointing enables pause/resume across process restarts — use SQLite for production
5. `interrupt_before` pauses the graph for human review — resume with `.invoke(None, config)`

---

## Exercises

1. Build a code-review graph: generate → lint → test → human review (if tests fail) → revise
2. Add error recovery: if `executor_node` fails 3 times (`error_count >= 3`), route to a fallback node
3. Implement a parallel research graph that fetches info from 3 different sources, then synthesizes

---

**Previous:** [Lecture 05](Lecture-05.md) | **Next:** [Lecture 07 - Agent SDKs and Runtime APIs](Lecture-07.md)
