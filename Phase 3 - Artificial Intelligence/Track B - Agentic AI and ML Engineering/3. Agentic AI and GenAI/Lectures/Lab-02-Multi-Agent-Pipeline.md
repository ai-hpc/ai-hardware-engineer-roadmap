# Lab 02 — Review workflows, parallel checks and restart recovery

**Track B · Agentic AI & GenAI** | [Previous](Lab-01-Research-Agent.md) | [Index](README.md) | [Next](Lab-03-Production-RAG.md)

**Time:** 60–90 minutes. Build the actual LangGraph extension in [the local workflow lab](../Local-RAG-Workflow-Lab/Guide.md), sections 4–6 and 8. Its nodes are deterministic checks; they are not presented as independent AI agents. Adding model-backed reviewers is a later experiment.

## 1. Draw the dependency graph

A retrieval node creates a proposed quotation. Two checks can run independently: one validates citation content/revision/span; the other validates that cited IDs belong to this reader's index. Review waits for both. A rejected or unsupported draft ends without publishing a local artifact.

Read [workflow_lab.py](../Local-RAG-Workflow-Lab/labs/workflow_lab.py). Nodes return **partial updates**. The `reviews` list has an explicit reducer so parallel updates are combined. Returning an entire previous state from every branch can replay old values or cause concurrent-update errors.

The join uses `add_edge(["citations", "scope"], "review")`. Two unrelated edges are not a clear substitute for an explicit all-predecessors dependency when the graph later develops branches of different depth. [LangGraph edge contract](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_edge).

## 2. Review a concrete artifact

Start a run and inspect its persisted interrupt payload. The quotation and citations are visible before approval. Exit the process, then invoke `approve` or `reject` in a new process using the same directory and run ID.

Approval is a typed value, not a substring found in a review paragraph. A response saying “NOT APPROVED” must never pass because it contains “APPROVED”. The supplied implementation accepts a real boolean through `Command(resume=...)` and keeps the review decision separate from prose.

The reviewer here is the learner. Neither automatic source check verifies all aspects of relevance, so do not automatically promote a successful citation check into semantic approval.

## 3. Test the crash window

Use `--crash-after-save` when approving a fresh run. The local artifact commits, then the node fails before its completion checkpoint. Retry in another process.

The artifact store's unique run ID and payload hash prevent duplicate rows and detect changed content under the same ID. Observe one final row. Explain why a checkpoint by itself would not provide this property. The tested transaction is local SQLite; no email, payment, robot command or external service is involved.

A remote mutation needs a receiver-supported idempotency key, transactional outbox, or a reconciliation workflow. Network timeout can mean “the response was lost after the action happened”, as demonstrated in the [localhost transport exercise](../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/Localhost-Network-Lab.md).

## 4. Add model roles only when they earn their cost

A planner, coder and reviewer can share the same underlying model and therefore share failure modes. Different role prompts do not make their judgments independent. Keep objective checks—schema validation, tests, source spans and policy—outside their prose.

For an optional multi-agent variant, hold the tasks, tools and budgets fixed and compare:

| System | Record |
|---|---|
| Deterministic workflow | Contract failures, relevance failures, elapsed time |
| One model with tools | Task success, tool errors, total calls/tokens, latency |
| Two model roles | Additional errors caught, new errors introduced, extra cost/latency |

Generated code is untrusted input. Do not execute it in the course workspace merely to make a reviewer say it passed. An execution extension needs a disposable, bounded environment and meaningful tests. Keep human review of the concrete diff/output where the application requires it.

## 5. Exercises with answers

1. Remove the `reviews` reducer in a temporary copy. **Answer:** parallel writes to that key lose their defined merge behavior; the framework test should expose the invalid update.
2. Put a counter increment before `interrupt()`. **Answer:** resuming restarts that node; a non-idempotent side effect can repeat.
3. Change the payload but reuse its artifact ID. **Answer:** the store rejects the conflict instead of silently treating different work as a retry.
4. Add a reviewer that agrees with the first model. **Answer:** agreement alone is not evidence of correctness; evaluate against an independent labelled/tested outcome.

**Submit:** graph diagram, approval/rejection traces, injected-crash trace and one artifact-count query. Report optional model evaluation separately from the executed deterministic checks.
