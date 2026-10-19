# Lecture 10 — Memory Systems That Preserve Evidence

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 09](Lecture-09.md) | **Next:** [Lecture 11](Lecture-11.md)

**Outcome:** choose what to remember, preserve a valid tool transcript, and distinguish retrieval similarity from truth and task success.

## 1. Four useful meanings of memory

| Memory | Example | Main failure |
|---|---|---|
| Working context | Current task and tool results | Token overflow or lost constraints |
| User preferences | An explicitly saved language preference | Wrong user, stale preference, excessive retention |
| Knowledge retrieval | Versioned manual passages | Wrong version, unauthorized document, missing evidence |
| Episodic history | Prior task, actions and independently checked result | Mistaking a fluent answer for a successful action |

External storage does not automatically update model weights. Adding an episode is retrieval-based adaptation, not training. A vector index, SQL table and file each have different consistency and access-control properties.

## 2. Compact complete turns, not arbitrary message slices

A sliding window must respect the provider protocol. Keeping the last six message objects can drop the assistant's tool call while retaining its result, or separate one result from the other calls in the same turn. Responses reasoning/continuation items may also need to travel with a turn.

Maintain records for complete logical turns: user request, assistant blocks/items, calls, results and terminal state. A pending call is not a completed turn. Keep the active task, user constraints, authoritative result IDs and unresolved actions in structured state outside a generated summary.

**Worked example:** a robot diagnostic agent calls `read_temperature` with ID `c7`. The result arrives after a summary is created. If the summary says “sensor is safe” before consuming the result, the next action uses invented state. Keep `c7` pending, then attach its timestamped result and evaluate the threshold in code. A summary is never a substitute for that measurement.

Summaries are lossy model outputs. Store their source turn range and distinguish quoted external instructions from authorized user instructions. Do not promote a retrieved instruction into a system message. Test constraints that appear early in long sessions and contradictions introduced later.

## 3. Persistent vector retrieval: an explicit metric

This is a **Chroma API template**, not a measured embedding benchmark. It requires Chroma and a supplied embedding function. The local [RAG workflow lab](../Local-RAG-Workflow-Lab/Guide.md) provides the fully executed CPU retrieval path.

```python
import uuid
from datetime import datetime, timezone
import chromadb

class LongTermMemory:
    def __init__(self, embedding_function, path="./memory-data"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="agent_memory_cosine_v1",
            embedding_function=embedding_function,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def remember(self, content, source_id):
        identifier = str(uuid.uuid4())
        self.collection.add(
            ids=[identifier], documents=[content],
            metadatas=[{"source_id": source_id,
                        "saved_at": datetime.now(timezone.utc).isoformat()}],
        )
        return identifier

    def recall(self, query, k=3):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_texts=[query], n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [{"text": doc, "source": meta, "cosine_similarity": 1.0 - dist}
                for doc, meta, dist in zip(result["documents"][0],
                                           result["metadatas"][0],
                                           result["distances"][0])]
```

`chromadb.Client()` is ephemeral; use `PersistentClient` when persistence is intended. `1 - distance` is cosine similarity only with cosine distance, not with the default squared L2 metric. Cosine similarity is not a calibrated probability of relevance or truth. For unit vectors, squared L2 distance is `2 - 2*cosine`; a squared L2 distance of 0.4 means cosine 0.8, not 0.6.

An existing collection retains its established metric and embedding dimension; passing a different configuration to `get_or_create_collection` does not migrate stored vectors. Version the collection and rebuild when changing either. Evaluate any threshold on a development split, then freeze it for held-out evaluation. The example assumes a single local user; a collection name is not an authorization boundary for a multi-user service.

Sources: [Chroma client reference](https://docs.trychroma.com/reference/python/client), [collection configuration and distance semantics](https://cookbook.chromadb.dev/core/collections/).

## 4. Save an episode only with a declared outcome

A useful episode contains task ID, input/data versions, action IDs, result artifacts, start/end times and an outcome such as `verified_success`, `verified_failure` or `unknown`. Record who or what verified it.

Do not set `success = "error" not in answer.lower()`. “The test did not pass” contains no `error`, and “no error occurred” contains it. Instead, use the actual test exit status plus expected assertions, a receipt checked against the requested action, or an independently labeled outcome. Save failed and unknown episodes too: filtering to successes hides important failure modes.

If SQLite is authoritative and the vector index is derived, a crash can occur after committing SQL but before indexing. Save an indexing-outbox record in the same SQL transaction, process it idempotently using the episode ID, and reconcile after restart. Two separate `.commit()`/`.add()` calls are not one atomic transaction. Keep deletion/version updates consistent across both stores.

The [local workflow lab](../Local-RAG-Workflow-Lab/Guide.md) teaches durable state, retries and replay. Use that executed implementation before adding a second database.

## 5. Choosing what to retain

Save information that supports the user's task and is authorized for the relevant user. Attach provenance, version, freshness and scope. Store document IDs and evidence spans when possible; avoid creating “facts” from an unverified generated answer. Retrieved content remains data, even if it contains instructions.

For changing facts, retain validity time and supersession information. For sensor measurements, distinguish observation time from arrival time. For preferences, give the user a way to correct or remove them. For embeddings, document model revision, pooling, normalization and tokenizer: a new encoder generally requires re-embedding the corpus.

## Practice and worked answers

1. Create a persistent collection, add one authorized fixture, restart the process and retrieve it. Repeat with an ephemeral client. **Expected:** persistence depends on the client/path, not the word “memory” in the class name.
2. A distance is 0.4. Can you call it 60% confidence? **No:** identify the metric first; even cosine 0.6 is not a probability.
3. A transcript contains two tool calls and only one returned result. Can you summarize and discard that turn? **No:** preserve pending IDs and current provider continuation state until the turn is resolved or explicitly canceled.
4. SQLite contains episode 27 but the index lacks it after a crash. **Recovery:** replay the durable outbox idempotently; do not rerun the external action.
5. A remembered successful response is retrieved for a new firmware revision. **Decision:** revalidate its applicability and current evidence; similarity does not transfer correctness across versions.

Submit a small trace showing one retrieved hit, one deliberately irrelevant hit, one stale fact and one incomplete turn. Explain how your system distinguishes them.
