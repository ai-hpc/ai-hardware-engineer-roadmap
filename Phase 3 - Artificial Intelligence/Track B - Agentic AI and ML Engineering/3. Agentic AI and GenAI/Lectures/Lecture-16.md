# Lecture 16 — LangGraph state, joins and durable recovery

**Course:** [AI Agent Development](../Guide.md) | [Previous](Lecture-15.md) | [Next](Lecture-17.md)

**Executed 18 October 2026:** LangGraph 1.2.16 and `langgraph-checkpoint-sqlite`3.1.2 in the [local workflow lab](../Local-RAG-Workflow-Lab/Guide.md). Learn the actual state and recovery semantics before inserting model calls. Allow60–90 minutes.

## 1. A graph makes dependencies explicit

A node reads state and returns an update. An edge determines what can run next. Conditional edges select a path; a bounded graph or recursion limit prevents an accidental endless loop. A graph framework does not itself evaluate whether a model's answer is correct.

The executed example retrieves a source quotation, runs two independent checks, waits for both, pauses for review, and saves a local artifact only after approval. Its deterministic nodes are useful for learning orchestration; they are not described as autonomous model agents.

## 2. State updates are partial, and parallel writes need a rule

```python
from typing import Annotated, TypedDict
import operator

class ReviewState(TypedDict):
    query: str
    reviews: Annotated[list[str], operator.add]
    approved: bool

# A node returns only what it changes:
def citation_check(state: ReviewState):
    return {'reviews': ['citations:pass']}
```

The reducer concatenates updates for `reviews`. Without a reducer, concurrent writes to the same ordinary state field can be invalid. Returning `{**state, ...}` from every branch can write stale copies of unrelated fields and replay accumulated lists. Return the intended changes and choose a reducer whose behavior is appropriate, including ordering and duplicate handling.

For message histories, use the framework's message-aware reducer when its replacement/ID semantics are needed; ordinary list concatenation is not interchangeable with message merging. [State/reducer documentation](https://docs.langchain.com/oss/python/langgraph/graph-api).

## 3. Express an all-predecessors join

```python
from langgraph.graph import StateGraph, START, END

def scope_check(state: ReviewState):
    return {'reviews': ['scope:pass']}

def join(state: ReviewState):
    assert sorted(state['reviews']) == ['citations:pass', 'scope:pass']
    return {'approved': False}

builder = StateGraph(ReviewState)
builder.add_node('citations', citation_check)
builder.add_node('scope', scope_check)
builder.add_node('join', join)
builder.add_edge(START, 'citations')
builder.add_edge(START, 'scope')
builder.add_edge(['citations', 'scope'], 'join')
builder.add_edge('join', END)
graph = builder.compile()
result = graph.invoke({'query': 'fixture', 'reviews': [], 'approved': False})
```

The explicit list edge waits for both named predecessors. Separate ordinary edges can have different scheduling implications as branches gain more steps; write the dependency you mean. The complete lab verifies this with real framework execution. [add_edge contract](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_edge).

## 4. Keep the checkpoint resource alive

`SqliteSaver.from_conn_string` returns a context manager. Passing that context manager itself to `compile` is not the same as opening the saver:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

config = {'configurable': {'thread_id': 'example-1'}}
with SqliteSaver.from_conn_string('/tmp/example-checkpoints.db') as saver:
    persistent_graph = builder.compile(checkpointer=saver)
    result = persistent_graph.invoke(
        {'query': 'fixture', 'reviews': [], 'approved': False}, config=config
    )
    print(persistent_graph.get_state(config).next)
```

Use an in-memory saver for ephemeral tests and a persistence backend suited to the workload for durable runs. SQLite is useful for a local exercise; declaring it “production” does not solve concurrency, backups, retention or service isolation. A thread ID selects state; it is not authentication. Verify ownership before loading another user's thread. [Persistence reference](https://docs.langchain.com/oss/python/langgraph/persistence).

## 5. Distinguish dynamic interrupts from ordinary retry

A dynamic `interrupt(payload)` pauses with a reviewable value. Resume it with `Command(resume=decision)` using the same thread/checkpointer. The node restarts from its beginning, and the corresponding interrupt receives the supplied value. Put non-idempotent effects after review or protect them separately.

The lab's CLI separates `approve`, `reject` and `retry`. Retry while approval is pending fails. An ordinary retry after a failed node uses saved state; it does not silently grant an approval. A completed graph, a failed node and an interrupt are different states; “invoke None always resumes” is an inadequate explanation.

An approval decision must bind to the concrete proposed content. If the proposal changes, its earlier approval is not automatically valid. The lab stores a content hash with its local artifact ID. [Interrupt and resume rules](https://docs.langchain.com/oss/python/langgraph/interrupts).

## 6. A checkpoint does not roll back an external action

Consider this order:

1. Save a note or send a request to an external service.
2. The action succeeds.
3. The process crashes before the graph records success.
4. The node runs again after restart.

Without receiver-side idempotency or reconciliation, step 4 can repeat the action. The lab intentionally crashes at this boundary and uses a local SQLite unique key plus payload hash to keep one artifact row. Tests restart actual Python processes; no model or external service is involved.

Do not label a checkpointed workflow “exactly once” without specifying the effect and transaction boundary. For robotics, a planning checkpoint does not undo movement or make a repeated actuator command acceptable.

## 7. Completion

Run the [workflow lab](../Local-RAG-Workflow-Lab/Guide.md), approve one quotation, reject another, and recover one injected crash. Submit the persisted review payload, final status and artifact count.

**Diagnostic answers:** reducers define how updates combine; a join defines dependencies; a checkpoint persists graph state; an idempotency contract controls repeated effects. None of these establishes that the original answer is factually correct.
