# Lecture 10 — Memory Systems

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 09](Lecture-09.md) | **Next:** [Lecture 11](Lecture-11.md)

---

## Learning Objectives

- Distinguish the four types of agent memory
- Implement in-context, external, and episodic memory
- Choose the right memory strategy for a given use case
- Manage context window growth in long-running agents

---

## 1. The Four Memory Types

| Type | Where stored | Lifespan | Retrieval |
|------|-------------|----------|-----------|
| **In-context (working)** | LLM context window | Current session | Automatic (LLM sees it) |
| **External (long-term)** | Vector DB, SQL, files | Persistent | Semantic search or lookup |
| **Episodic** | Database of past interactions | Persistent | Similarity search |
| **Semantic (knowledge)** | Vector DB of facts/docs | Persistent until updated | RAG retrieval |

---

## 2. In-Context Memory (Working Memory)

The simplest form — everything in the current **message list**.

```python
import anthropic
from collections import deque

client = anthropic.Anthropic()

class ConversationAgent:
    def __init__(self, system: str, max_history: int = 20):
        self.system = system
        self.messages = []
        self.max_history = max_history  # sliding window

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        # Sliding window: keep only recent messages
        if len(self.messages) > self.max_history:
            # Always keep first message (task context) + recent messages
            self.messages = self.messages[:2] + self.messages[-(self.max_history-2):]

        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            system=self.system,
            messages=self.messages
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply
```

### Context Compression

When context grows too large, **summarize old turns**:

```python
def compress_history(messages: list, keep_recent: int = 6) -> list:
    """Summarize old messages, keep recent ones verbatim."""
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # Summarize old messages
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[tool call]'}"
        for m in old_messages
    )

    summary_response = client.messages.create(
        model="your-fast-model-id",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Summarize this conversation history concisely, preserving key facts and decisions:\n\n{history_text}"
        }]
    )

    summary = summary_response.content[0].text
    summary_message = {
        "role": "user",
        "content": f"[Conversation summary — {len(old_messages)} earlier messages]\n{summary}"
    }

    return [summary_message] + recent_messages
```

---

## 3. External Long-Term Memory (Vector Store)

Store facts, user preferences, and past results in a **vector database** for retrieval across sessions.

```python
import chromadb
from chromadb.utils import embedding_functions
import json
from datetime import datetime

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.client = chromadb.Client()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )

    def remember(self, content: str, metadata: dict = None) -> str:
        """Store a memory with optional metadata."""
        memory_id = f"mem_{datetime.now().timestamp()}"
        self.collection.add(
            documents=[content],
            ids=[memory_id],
            metadatas=[metadata or {"timestamp": datetime.now().isoformat()}]
        )
        return memory_id

    def recall(self, query: str, n_results: int = 5) -> list[dict]:
        """Retrieve relevant memories by semantic similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            memories.append({
                "content": doc,
                "metadata": meta,
                "relevance": 1 - dist  # convert distance to similarity
            })
        return memories

    def forget(self, memory_id: str):
        """Remove a specific memory."""
        self.collection.delete(ids=[memory_id])


class MemoryAugmentedAgent:
    def __init__(self):
        self.memory = LongTermMemory()
        self.messages = []

    def chat(self, user_input: str) -> str:
        # Retrieve relevant memories
        memories = self.memory.recall(user_input, n_results=3)
        memory_context = ""
        if memories:
            memory_context = "Relevant context from memory:\n" + "\n".join(
                f"- {m['content']} (relevance: {m['relevance']:.2f})"
                for m in memories if m['relevance'] > 0.5
            )

        # Build augmented message
        augmented_input = user_input
        if memory_context:
            augmented_input = f"{memory_context}\n\nUser: {user_input}"

        self.messages.append({"role": "user", "content": augmented_input})

        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=1024,
            system="You are a helpful assistant with access to long-term memory.",
            messages=self.messages
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})

        # Store important information from this exchange
        self._extract_and_store(user_input, reply)

        return reply

    def _extract_and_store(self, user_input: str, reply: str):
        """Extract facts worth remembering."""
        extract_response = client.messages.create(
            model="your-fast-model-id",
            max_tokens=256,
            system="""Extract facts worth remembering from this exchange.
Return a JSON array of strings. Return [] if nothing notable.
Focus on: user preferences, decisions made, facts learned.""",
            messages=[{
                "role": "user",
                "content": f"User said: {user_input}\nAssistant replied: {reply[:500]}"
            }]
        )
        try:
            facts = json.loads(extract_response.content[0].text)
            for fact in facts:
                self.memory.remember(fact, {"source": "conversation", "type": "fact"})
        except json.JSONDecodeError:
            pass  # No facts to store
```

---

## 4. Episodic Memory (Past Interactions)

Store and retrieve complete **past task executions** — useful for **learning from experience**.

```python
import sqlite3
from dataclasses import dataclass

@dataclass
class Episode:
    task: str
    approach: str
    result: str
    success: bool
    timestamp: str

class EpisodicMemory:
    def __init__(self, db_path: str = "episodes.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()
        self.vector_index = LongTermMemory("episodes")

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                approach TEXT,
                result TEXT,
                success INTEGER,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def store_episode(self, episode: Episode) -> int:
        cursor = self.conn.execute(
            "INSERT INTO episodes VALUES (NULL,?,?,?,?,?)",
            (episode.task, episode.approach, episode.result,
             int(episode.success), episode.timestamp)
        )
        self.conn.commit()
        episode_id = cursor.lastrowid

        # Also index in vector store for semantic search
        self.vector_index.remember(
            f"Task: {episode.task}\nApproach: {episode.approach}\nSuccess: {episode.success}",
            metadata={"episode_id": str(episode_id), "success": str(episode.success)}
        )
        return episode_id

    def recall_similar_episodes(self, task: str, n: int = 3) -> list[Episode]:
        """Find similar past tasks and what worked."""
        memories = self.vector_index.recall(task, n_results=n)
        episodes = []
        for mem in memories:
            if mem["relevance"] > 0.4:
                episode_id = int(mem["metadata"]["episode_id"])
                row = self.conn.execute(
                    "SELECT * FROM episodes WHERE id=?", (episode_id,)
                ).fetchone()
                if row:
                    episodes.append(Episode(
                        task=row[1], approach=row[2], result=row[3],
                        success=bool(row[4]), timestamp=row[5]
                    ))
        return episodes


def agent_with_episodic_memory(task: str) -> str:
    episodic = EpisodicMemory()

    # Check if we've done something similar before
    similar = episodic.recall_similar_episodes(task)
    prior_context = ""
    if similar:
        successful = [e for e in similar if e.success]
        if successful:
            prior_context = f"\nSimilar past successes:\n" + "\n".join(
                f"- Task: {e.task}\n  Approach: {e.approach}"
                for e in successful[:2]
            )

    response = client.messages.create(
        model="your-agent-model-id",
        max_tokens=1024,
        system=f"You are a task executor.{prior_context}",
        messages=[{"role": "user", "content": task}]
    )

    result = response.content[0].text
    success = "error" not in result.lower() and "failed" not in result.lower()

    episodic.store_episode(Episode(
        task=task,
        approach="Direct LLM response",
        result=result[:500],
        success=success,
        timestamp=datetime.now().isoformat()
    ))

    return result
```

---

## 5. Memory Strategy Selection Guide

```
Does the agent need to remember things across sessions?
├── No → In-context memory only (sliding window)
└── Yes → External memory needed
    │
    ├── Remember facts/preferences about the user?
    │   → Long-term vector store (semantic recall)
    │
    ├── Remember how to do tasks (what worked)?
    │   → Episodic memory (SQLite + vector index)
    │
    └── Answer questions from documents/knowledge?
        → RAG / semantic memory (covered in Lectures 09-10)
```

---

## Key Takeaways

1. Use **sliding window** to prevent context overflow in long conversations
2. **Compress old history** with a cheap model (Haiku) before it fills the window
3. **Vector stores** enable semantic recall across sessions — use `all-MiniLM-L6-v2` for cost efficiency
4. **Episodic memory** lets agents learn from past successes/failures
5. Extract and store facts selectively — storing everything creates noise at retrieval time

---

## Exercises

1. Build a `ConversationAgent` with auto-compression that triggers when context exceeds 80% of the model's limit.
2. Implement a user profile system using `LongTermMemory` that persists preferences (communication style, expertise level, interests).
3. Create an episodic memory that tracks which tools succeeded vs. failed for given task types, and uses that to guide tool selection.

---

**Previous:** [Lecture 09](Lecture-09.md) | **Next:** [Lecture 11](Lecture-11.md)
