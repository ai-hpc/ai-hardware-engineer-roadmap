# Lecture 12 — RAG: Retrieval & Reranking

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 11](Lecture-11.md) | **Next:** [Lecture 13](Lecture-13.md)

---

## Learning Objectives

By the end of this lecture you will be able to:

1. Explain the difference between dense retrieval and hybrid (BM25 + dense) retrieval.
2. Apply MMR to produce diverse, non-redundant result sets.
3. Implement a cross-encoder reranker to reorder initial retrieval results.
4. Use query expansion and HyDE to improve recall on ambiguous queries.
5. Build a multi-query retriever to cover multiple phrasings of the same question.
6. Assemble a complete, production-grade RAG chain using LangChain LCEL.

---

## 1. Dense vs Hybrid Search

### 1.1 Dense Retrieval (Semantic Search)

**Dense retrieval** embeds the query and all documents into a shared **vector space**. Documents are ranked by **cosine similarity** (or dot product) to the query vector. It handles semantic equivalence well: "memory bandwidth" and "data transfer rate" may retrieve the same documents.

```python
# pip install langchain langchain-chroma sentence-transformers

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(embedding_function=embedding_fn, collection_name="demo")
docs = [
    Document(page_content="H100 GPU delivers 3.35 TB/s HBM3 bandwidth.", metadata={"id": "1"}),
    Document(page_content="The A100 accelerator provides 2 TB/s memory throughput.", metadata={"id": "2"}),
    Document(page_content="Python 3.12 introduced significant interpreter speedups.", metadata={"id": "3"}),
    Document(page_content="NVLink 4.0 connects GPUs at 900 GB/s.", metadata={"id": "4"}),
]
vectorstore.add_documents(docs)

results = vectorstore.similarity_search_with_score("GPU memory data transfer rate", k=3)
print("Dense retrieval results:")
for doc, score in results:
    print(f"  score={score:.4f}  {doc.page_content}")
```

**Weakness:** Dense retrieval misses exact keyword matches. If you search for "NVLink 4.0" but no document contains that exact phrase, it may fail.

### 1.2 BM25 (Sparse/Keyword Search)

**BM25** is a classical TF-IDF-derived ranking function. It excels at **exact keyword matching**.

```python
# pip install rank-bm25

from rank_bm25 import BM25Okapi
import re

corpus = [
    "H100 GPU delivers 3.35 TB/s HBM3 bandwidth.",
    "The A100 accelerator provides 2 TB/s memory throughput.",
    "Python 3.12 introduced significant interpreter speedups.",
    "NVLink 4.0 connects GPUs at 900 GB/s.",
]

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()

tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "NVLink 4.0"
scores = bm25.get_scores(tokenize(query))
ranked = sorted(zip(scores, corpus), reverse=True)

print("BM25 results:")
for score, doc in ranked:
    print(f"  score={score:.3f}  {doc}")
```

### 1.3 Hybrid Search (BM25 + Dense)

**Hybrid search** combines both scores using **Reciprocal Rank Fusion (RRF)**:

```python
# pip install langchain-community rank-bm25 sentence-transformers

import re
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class HybridRetriever:
    """Combines BM25 and dense vector search via Reciprocal Rank Fusion."""

    def __init__(self, documents: list[Document], embedding_model: str = "all-MiniLM-L6-v2", k: int = 60):
        self.documents = documents
        self.k = k  # RRF constant

        # BM25 index
        tokenized = [self._tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Dense embeddings
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        texts = [d.page_content for d in documents]
        self.doc_vectors = np.array(self.embedder.embed_documents(texts))

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def _dense_ranks(self, query: str) -> list[int]:
        q_vec = np.array(self.embedder.embed_query(query))
        scores = self.doc_vectors @ q_vec
        return list(np.argsort(-scores))  # descending

    def _bm25_ranks(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(self._tokenize(query))
        return list(np.argsort(-scores))

    def retrieve(self, query: str, top_k: int = 4) -> list[Document]:
        dense_ranks = self._dense_ranks(query)
        sparse_ranks = self._bm25_ranks(query)

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (self.k + rank + 1)
        for rank, idx in enumerate(sparse_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (self.k + rank + 1)

        top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices]


# Demo
docs = [
    Document(page_content="H100 GPU delivers 3.35 TB/s HBM3 bandwidth."),
    Document(page_content="The A100 accelerator provides 2 TB/s memory throughput."),
    Document(page_content="NVLink 4.0 connects GPUs at 900 GB/s."),
    Document(page_content="Python 3.12 introduced significant interpreter speedups."),
    Document(page_content="CUDA 12 supports the Hopper architecture."),
]
retriever = HybridRetriever(docs)
for doc in retriever.retrieve("NVLink 4.0 bandwidth"):
    print(f"  {doc.page_content}")
```

---

## 2. Maximal Marginal Relevance (MMR)

**MMR** selects documents that are relevant to the query AND **diverse** from each other. It prevents retrieving five near-identical chunks.

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = Chroma(embedding_function=embedding_fn)

# Add some redundant documents
vs.add_documents([
    Document(page_content="H100 has 80 GB HBM3 memory with 3.35 TB/s bandwidth."),
    Document(page_content="H100 GPU memory is 80 GB HBM3 at 3.35 terabytes per second."),
    Document(page_content="The H100 SXM5 uses HBM3 providing 3.35 TB/s throughput."),
    Document(page_content="NVLink 4.0 delivers 900 GB/s between H100 GPUs."),
    Document(page_content="H100 Tensor Cores achieve 3958 TFLOPS for FP8 inference."),
])

# Standard retrieval — will return 3 near-identical H100 memory docs
standard = vs.similarity_search("H100 memory bandwidth", k=3)
print("Standard retrieval:")
for d in standard:
    print(f"  {d.page_content}")

# MMR retrieval — lambda_mult=0.5 balances relevance vs diversity
mmr = vs.max_marginal_relevance_search(
    "H100 memory bandwidth",
    k=3,
    fetch_k=10,       # candidate pool size
    lambda_mult=0.5,  # 0=pure diversity, 1=pure relevance
)
print("\nMMR retrieval:")
for d in mmr:
    print(f"  {d.page_content}")
```

**lambda_mult tuning:**
- `0.0` — maximally diverse (ignores relevance)
- `0.5` — balanced default
- `1.0` — equivalent to standard similarity search

---

## 3. Cross-Encoder Reranking

**Two-stage retrieval**: first retrieve a broad candidate set (e.g., top-20 with fast dense search), then rerank with a slower but more accurate **cross-encoder** that reads query + document together.

```python
# pip install sentence-transformers

from sentence_transformers.cross_encoder import CrossEncoder
from langchain_core.documents import Document

# Load once at startup (model is ~130 MB)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[Document], top_k: int = 3) -> list[Document]:
    """Rerank candidate documents using a cross-encoder."""
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:top_k]]


# Simulate a first-stage retrieval returning 6 candidates
candidates = [
    Document(page_content="H100 SXM5 memory bandwidth is 3.35 TB/s using HBM3."),
    Document(page_content="Python lists are dynamically resizable arrays."),
    Document(page_content="The GPU memory subsystem uses high bandwidth memory."),
    Document(page_content="H100 PCIe version has 80 GB HBM2e at 2 TB/s."),
    Document(page_content="NVLink connects multiple H100 GPUs in a node."),
    Document(page_content="Memory bandwidth determines LLM inference throughput."),
]

query = "What is the memory bandwidth of the H100?"
reranked = rerank(query, candidates, top_k=3)
print(f"Top 3 after reranking for: '{query}'")
for i, doc in enumerate(reranked, 1):
    print(f"  {i}. {doc.page_content}")
```

**Why reranking helps:** The **bi-encoder** (dense retrieval) encodes query and document independently. The **cross-encoder** attends to every token of both simultaneously, giving much better relevance scores at the cost of O(candidates) inference calls.

---

## 4. Query Expansion and HyDE

### 4.1 Query Expansion

Simple expansion generates **multiple phrasings** of the query and merges results.

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="your-fast-model-id", temperature=0)

expand_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a search query optimizer. Generate {n} diverse rephrasing of the user's question. Output one per line, no numbering."),
    ("human", "{query}"),
])

expand_chain = expand_prompt | llm | StrOutputParser()

def expand_query(query: str, n: int = 3) -> list[str]:
    result = expand_chain.invoke({"query": query, "n": n})
    variants = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return [query] + variants  # original + expansions

# Usage
queries = expand_query("What is the memory bandwidth of H100?")
for q in queries:
    print(f"  - {q}")
```

### 4.2 HyDE (Hypothetical Document Embeddings)

Instead of embedding the short query, generate a **hypothetical document** that would answer the question, then embed that. The hypothesis is much closer in vector space to actual documents.

```python
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

llm = ChatOpenAI(model="your-fast-model-id", temperature=0.3)
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def hyde_retrieve(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
    """
    HyDE: embed a generated hypothetical answer rather than the raw query.
    """
    # Step 1: generate a hypothetical document
    hypothesis = llm.invoke(
        f"Write a short technical paragraph (3-4 sentences) that would answer: {query}"
    ).content

    print(f"Hypothesis: {hypothesis[:100]}...")

    # Step 2: embed the hypothesis
    hyp_vec = np.array(embedder.embed_query(hypothesis))

    # Step 3: embed all documents and rank by similarity to hypothesis
    doc_vecs = np.array(embedder.embed_documents([d.page_content for d in documents]))
    scores = doc_vecs @ hyp_vec

    ranked_idx = np.argsort(-scores)[:top_k]
    return [documents[i] for i in ranked_idx]


corpus = [
    Document(page_content="H100 SXM5 uses HBM3 with 3.35 TB/s memory bandwidth."),
    Document(page_content="A100 features HBM2e memory delivering 2 TB/s bandwidth."),
    Document(page_content="NVLink 4.0 provides 900 GB/s GPU-to-GPU bandwidth."),
    Document(page_content="Python asyncio enables non-blocking I/O operations."),
]

results = hyde_retrieve("What is the fastest GPU memory bandwidth available?", corpus)
for doc in results:
    print(f"  {doc.page_content}")
```

---

## 5. Multi-Query Retrieval

LangChain's `MultiQueryRetriever` issues multiple rephrasings of the original query, deduplicates results, and returns the union.

```python
import os
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = Chroma(embedding_function=embedding_fn)
vs.add_documents([
    Document(page_content="H100 SXM5 has 3.35 TB/s HBM3 bandwidth."),
    Document(page_content="A100 has 2 TB/s HBM2e bandwidth."),
    Document(page_content="GPU memory bandwidth limits LLM inference speed."),
    Document(page_content="NVLink 4.0 enables 900 GB/s inter-GPU communication."),
    Document(page_content="The H100 PCIe variant has lower bandwidth than SXM5."),
])

llm = ChatOpenAI(model="your-fast-model-id", temperature=0)
base_retriever = vs.as_retriever(search_kwargs={"k": 2})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# Enable verbose logging to see generated queries
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

docs = multi_retriever.invoke("How fast is H100 GPU memory?")
print(f"\nRetrieved {len(docs)} unique documents:")
for doc in docs:
    print(f"  {doc.page_content}")
```

---

## 6. Complete RAG Chain with LCEL

Putting retrieval, reranking, and generation together:

```python
# pip install langchain langchain-openai langchain-chroma sentence-transformers

import os
from typing import Any
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from sentence_transformers.cross_encoder import CrossEncoder


# ── Setup ─────────────────────────────────────────────────────────────────────
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model="your-fast-model-id", temperature=0)

# Build vector store
vs = Chroma(embedding_function=embedding_fn)
vs.add_documents([
    Document(page_content="H100 SXM5 provides 3.35 TB/s HBM3 memory bandwidth.", metadata={"source": "h100_spec.md"}),
    Document(page_content="A100 provides 2 TB/s HBM2e memory bandwidth.", metadata={"source": "a100_spec.md"}),
    Document(page_content="NVLink 4.0 delivers 900 GB/s bidirectional GPU-to-GPU bandwidth.", metadata={"source": "nvlink.md"}),
    Document(page_content="H100 PCIe has lower bandwidth (2 TB/s) compared to SXM5.", metadata={"source": "h100_spec.md"}),
    Document(page_content="GPU memory bandwidth is critical for LLM inference throughput.", metadata={"source": "guide.md"}),
    Document(page_content="HBM3 uses a stacked DRAM design to achieve high bandwidth density.", metadata={"source": "hbm.md"}),
])

base_retriever = vs.as_retriever(search_kwargs={"k": 6})  # broad first stage


# ── Reranking step ────────────────────────────────────────────────────────────
def rerank_docs(inputs: dict) -> list[Document]:
    query = inputs["question"]
    candidates = inputs["documents"]
    if not candidates:
        return []
    pairs = [(query, d.page_content) for d in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in ranked[:3]]  # top-3 after reranking


# ── Format context ────────────────────────────────────────────────────────────
def format_context(docs: list[Document]) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        sections.append(f"[{i}] ({src})\n{doc.page_content}")
    return "\n\n".join(sections)


# ── Prompt ────────────────────────────────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a technical assistant. Answer the question using ONLY the provided context. "
        "Cite sources using [N] notation. If the context does not contain the answer, say so."
    )),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# ── LCEL Pipeline ─────────────────────────────────────────────────────────────
rag_chain = (
    RunnablePassthrough.assign(
        documents=itemgetter("question") | base_retriever
    )
    | RunnablePassthrough.assign(
        documents=RunnableLambda(rerank_docs),
    )
    | RunnablePassthrough.assign(
        context=itemgetter("documents") | RunnableLambda(format_context)
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)


def ask(question: str) -> str:
    return rag_chain.invoke({"question": question})


if __name__ == "__main__":
    questions = [
        "What is the memory bandwidth of the H100?",
        "How does HBM3 achieve high bandwidth?",
        "What is NVLink 4.0 bandwidth?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
```

### 6.1 Streaming the Chain

```python
# Replace the final .invoke() call with .stream() for token-by-token output
def ask_streaming(question: str):
    print(f"Q: {question}")
    print("A: ", end="", flush=True)
    for chunk in rag_chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    print()  # newline

ask_streaming("Which GPU has faster memory, H100 SXM5 or A100?")
```

---

## Key Takeaways

- **Hybrid search** combines BM25 (exact keyword matching) with dense retrieval (semantic matching) via RRF, covering the weaknesses of each.
- **MMR** is a simple but effective way to prevent duplicate chunks from dominating retrieval results.
- **Two-stage retrieval** (dense top-K → cross-encoder rerank) gives the quality of cross-encoder scoring without the latency of scoring every document.
- **HyDE** often beats raw query embedding for sparse or short queries by generating a more "document-like" embedding.
- **Multi-query retrieval** improves recall at the cost of extra LLM calls — worth it for high-stakes questions.
- **LCEL** (`|` operator) composes retrieval and generation steps into a clean, inspectable pipeline that supports streaming out of the box.

---

## Exercises

### Exercise 1 — Hybrid vs Dense Ablation

Build a small corpus of 15 documents (mix of exact-keyword docs and paraphrase-heavy docs). Create five test queries where you expect keyword matching to win, and five where you expect semantic matching to win. Run both pure BM25 and pure dense retrieval, then hybrid. Report which strategy wins for each query and why.

### Exercise 2 — Reranker Threshold

Implement a `rerank_with_threshold(query, candidates, min_score)` function that uses the `CrossEncoder` but only returns documents whose raw score exceeds `min_score`. Experiment with values between -5 and 5 on a 10-document corpus. Plot the number of documents returned vs the threshold value and identify a sensible default for a technical documentation use case.

### Exercise 3 — Full Pipeline Benchmark

Build the complete LCEL RAG chain from this lecture. Add a wrapper that measures total latency broken down into three parts: (a) retrieval time, (b) reranking time, (c) LLM generation time. Run 10 queries and report the mean and p95 latency for each stage. Which stage dominates? What would you optimize first in a production system?

---

*Next: [Lecture 13](Lecture-13.md)*
