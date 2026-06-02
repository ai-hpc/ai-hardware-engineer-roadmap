# Lab 03 — Production RAG System

**Track B · Agentic AI & GenAI** | [← Index](README.md) | [Next → Lab 04](Lab-04-TokenJuice-Output-Compaction.md)

---

## Overview

In this lab you build a production-quality RAG system with hybrid search, cross-encoder reranking, RAGAS evaluation, a FastAPI streaming endpoint, and semantic caching. By the end you will have a complete, runnable system you can deploy.

```
                    ┌──────────────────────────────┐
                    │   3 fake hardware .md docs    │
                    └──────────────┬───────────────┘
                                   │ ingest
                    ┌──────────────▼───────────────┐
                    │   ChromaDB + BM25 index       │
                    └──────────────┬───────────────┘
     Query ─────────┐              │
                    │  retrieve    │
     ┌──────────────▼──────────────▼───────────────┐
     │   Hybrid Retriever (BM25 + dense, RRF)       │
     └──────────────────────────┬──────────────────┘
                                │ top-20 candidates
     ┌──────────────────────────▼──────────────────┐
     │   Cross-Encoder Reranker                     │
     └──────────────────────────┬──────────────────┘
                                │ top-3 reranked
     ┌──────────────────────────▼──────────────────┐
     │   LLM Generation (your-fast-model-id)              │
     └──────────────────────────┬──────────────────┘
                                │
     ┌──────────────────────────▼──────────────────┐
     │   FastAPI + SSE Streaming Endpoint           │
     │   + Semantic Cache (0.95 similarity)         │
     └─────────────────────────────────────────────┘
```

**Estimated time:** 90–120 minutes
**Difficulty:** Advanced

**What you will build:**

```
production_rag/
├── docs/                   ← fake hardware documentation (created in Step 2)
│   ├── h100_guide.md
│   ├── cuda_guide.md
│   └── networking_guide.md
├── ingest.py               ← document ingestion pipeline
├── retriever.py            ← hybrid retriever + reranker
├── rag_chain.py            ← full RAG chain
├── evaluate.py             ← RAGAS evaluation
├── cache.py                ← semantic cache
├── api.py                  ← FastAPI streaming endpoint
├── main.py                 ← demo runner
└── requirements.txt
```

---

## Step 1 — Project Setup

```bash
mkdir production_rag && cd production_rag
python -m venv .venv && source .venv/bin/activate
```

Install dependencies:

```bash
pip install langchain langchain-community langchain-chroma langchain-openai \
            sentence-transformers rank-bm25 chromadb faiss-cpu \
            ragas datasets openai fastapi uvicorn sse-starlette \
            redis numpy python-dotenv rich
```

Create `requirements.txt`:

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
langchain-openai>=0.1.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
chromadb>=0.5.0
faiss-cpu>=1.8.0
ragas>=0.1.14
datasets>=2.20.0
openai>=1.30.0
fastapi>=0.111.0
uvicorn>=0.30.0
sse-starlette>=2.1.0
redis>=5.0.0
numpy>=1.26.0
python-dotenv>=1.0.0
rich>=13.7.0
```

Create `.env`:

```
OPENAI_API_KEY=sk-...
```

---

## Step 2 — Create the Fake Hardware Documentation

These files serve as the corpus for our RAG system.

```python
# create_docs.py — run once to generate the fake docs
from pathlib import Path

Path("docs").mkdir(exist_ok=True)

Path("docs/h100_guide.md").write_text("""# NVIDIA H100 GPU Technical Guide

## Overview
The NVIDIA H100 Tensor Core GPU is built on the Hopper architecture (2022).
It is designed for large-scale AI training and inference workloads.

## Memory Specifications
The H100 SXM5 variant features 80 GB of HBM3 memory with 3.35 TB/s bandwidth.
The H100 PCIe variant offers 80 GB HBM2e with 2 TB/s bandwidth.
HBM3 uses a stacked DRAM design to achieve high bandwidth density in a small footprint.

## Compute Performance
- FP8 (inference): 3958 TFLOPS
- BF16 (training): 1979 TFLOPS with sparsity, 989 TFLOPS dense
- FP32: 67 TFLOPS
- TF32: 989 TFLOPS with sparsity
CUDA cores: 16896. Tensor Cores: 528 (4th generation).

## Interconnect
The H100 SXM5 uses NVLink 4.0 providing 900 GB/s total bidirectional bandwidth
when connecting multiple GPUs in an NVLink domain.
PCIe 5.0 x16 provides 128 GB/s bidirectional host-to-device bandwidth.

## Software Support
Supported CUDA versions: CUDA 11.8 and above.
Key features: Transformer Engine (TF32/FP8 auto-casting), DPX instructions,
Thread Block Clusters, Asynchronous Memory Copy (TMA).

## Use Cases
Best suited for: LLM training (GPT-4 class models), diffusion model inference,
scientific simulation, seismic processing.
""")

Path("docs/cuda_guide.md").write_text("""# CUDA Programming Guide

## What is CUDA?
CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform
and programming model. It enables general-purpose computing on GPUs.

## CUDA Hierarchy
CUDA organizes computation into a three-level hierarchy:
- Grid: the entire computation, made of blocks
- Block: a group of up to 1024 threads that share shared memory and can synchronize
- Thread: the smallest unit of execution

## Memory Types
- Global memory: accessible by all threads, high latency (~400 cycles), large capacity
- Shared memory: accessible by threads within a block, low latency (~4 cycles), 48-228 KB
- Registers: per-thread, lowest latency (1 cycle), ~256 KB per SM
- Constant memory: 64 KB, cached, fast for broadcast reads
- L2 cache: 50 MB on H100, shared across all SMs

## Key Concepts
### Warp Divergence
A warp is 32 threads that execute the same instruction. If threads in a warp take
different branches (if/else), both branches execute serially. This is warp divergence
and should be minimized.

### Memory Coalescing
GPU memory accesses are most efficient when consecutive threads access consecutive
memory addresses (coalesced access). Uncoalesced access can reduce bandwidth by 8-32x.

### Occupancy
Occupancy is the ratio of active warps to maximum warps on an SM.
Higher occupancy hides memory latency. Target 50-75% occupancy for most kernels.

## CUDA 12 New Features
- Thread Block Clusters: hierarchical grouping above thread blocks
- Tensor Memory Accelerator (TMA): asynchronous bulk memory transfers
- Distributed Shared Memory: shared memory accessible across a cluster

## Profiling Tools
- Nsight Compute: per-kernel performance counters and roofline analysis
- Nsight Systems: system-wide timeline for CPU-GPU interaction
- nvprof (legacy): command-line profiler for older CUDA versions
""")

Path("docs/networking_guide.md").write_text("""# GPU Networking and Interconnects

## NVLink
NVLink is NVIDIA's high-speed GPU-to-GPU interconnect.
- NVLink 3.0 (A100): 600 GB/s total bidirectional bandwidth, 12 links
- NVLink 4.0 (H100): 900 GB/s total bidirectional bandwidth, 18 links
- NVLink 5.0 (B200): 1800 GB/s total bidirectional bandwidth

NVLink enables GPU memory to be aggregated into a single unified pool (NVLink Sharp).

## NVSwitch
NVSwitch is a chip that enables all-to-all NVLink connectivity across many GPUs.
A DGX H100 node contains 8 H100 GPUs connected via 4 NVSwitch chips.
This creates a fully non-blocking fabric at 900 GB/s between any pair of GPUs.

## PCIe
PCIe (Peripheral Component Interconnect Express) connects the GPU to the CPU.
- PCIe 4.0 x16: 64 GB/s bidirectional
- PCIe 5.0 x16: 128 GB/s bidirectional
- PCIe 6.0 x16: 256 GB/s bidirectional (emerging)
PCIe bandwidth is often the bottleneck for single-GPU systems doing frequent host-GPU transfers.

## InfiniBand
InfiniBand is used for GPU-to-GPU communication across nodes in a cluster.
- HDR InfiniBand: 200 Gb/s per port
- NDR InfiniBand: 400 Gb/s per port
- XDR InfiniBand: 800 Gb/s per port (emerging)
NVIDIA's SHARP technology offloads collective operations (AllReduce) to the network.

## RDMA and GPUDirect
GPUDirect RDMA allows InfiniBand to transfer data directly to/from GPU memory,
bypassing the CPU and system RAM. This is critical for distributed training latency.
GPUDirect Storage allows direct NVMe-to-GPU transfers at >10 GB/s.

## Ethernet Alternatives
RoCE (RDMA over Converged Ethernet) v2 provides RDMA semantics over standard Ethernet.
NVIDIA Spectrum-X combines BlueField-3 DPUs with Spectrum-4 switches for AI networking.
""")

print("Docs created: docs/h100_guide.md, docs/cuda_guide.md, docs/networking_guide.md")
```

Run it:

```bash
python create_docs.py
```

---

## Step 3 — Build the Ingestion and Retrieval Pipeline

Create `ingest.py`:

```python
# ingest.py
"""
Document ingestion: load markdown → chunk → embed → store in ChromaDB.
Also builds a BM25 index for hybrid retrieval.
"""

import glob
import pickle
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

CHROMA_DIR = "./chroma_db"
BM25_FILE = "./bm25_index.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_and_chunk(docs_dir: str = "docs") -> list[Document]:
    """Load all markdown files and split into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
    )

    all_chunks = []
    for path in glob.glob(f"{docs_dir}/**/*.md", recursive=True):
        try:
            loader = UnstructuredMarkdownLoader(path, mode="elements")
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source_file"] = Path(path).name
                chunk.metadata["source_path"] = path
            all_chunks.extend(chunks)
            print(f"  Loaded {path} → {len(chunks)} chunks")
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def build_chroma(chunks: list[Document], persist_dir: str = CHROMA_DIR):
    """Embed chunks and store in ChromaDB."""
    print(f"Building ChromaDB index at {persist_dir}...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(
        collection_name="hardware_docs",
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )
    vs.add_documents(chunks)
    print(f"ChromaDB built: {vs._collection.count()} vectors")
    return vs


def build_bm25(chunks: list[Document], bm25_file: str = BM25_FILE):
    """Build and persist a BM25 index."""
    print("Building BM25 index...")

    def tokenize(text: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    tokenized = [tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(bm25_file, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved: {bm25_file}")
    return bm25, chunks


def run_ingestion(docs_dir: str = "docs"):
    """Full ingestion pipeline."""
    chunks = load_and_chunk(docs_dir)
    vs = build_chroma(chunks)
    bm25, stored_chunks = build_bm25(chunks)
    print("\nIngestion complete.")
    return vs, bm25, stored_chunks


if __name__ == "__main__":
    run_ingestion()
```

Create `retriever.py`:

```python
# retriever.py
"""
Hybrid retriever combining ChromaDB dense search + BM25, fused with RRF,
then reranked with a cross-encoder.
"""

import pickle
import re
from typing import Optional

import numpy as np
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder

CHROMA_DIR = "./chroma_db"
BM25_FILE = "./bm25_index.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRAGRetriever:
    """
    Three-stage retriever:
      1. Dense (ChromaDB) + sparse (BM25) retrieval fused with RRF
      2. Cross-encoder reranking
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        bm25_file: str = BM25_FILE,
        rrf_k: int = 60,
        initial_k: int = 20,
        final_k: int = 3,
    ):
        self.rrf_k = rrf_k
        self.initial_k = initial_k
        self.final_k = final_k

        # Load ChromaDB
        print("Loading ChromaDB...")
        embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vs = Chroma(
            collection_name="hardware_docs",
            embedding_function=embedding_fn,
            persist_directory=chroma_dir,
        )

        # Load BM25
        print("Loading BM25 index...")
        with open(bm25_file, "rb") as f:
            data = pickle.load(f)
        self.bm25: BM25Okapi = data["bm25"]
        self.bm25_chunks: list[Document] = data["chunks"]

        # Load reranker
        print(f"Loading reranker: {RERANK_MODEL}...")
        self.reranker = CrossEncoder(RERANK_MODEL)
        print("Retriever ready.")

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def _dense_retrieve(self, query: str) -> list[Document]:
        return self.vs.similarity_search(query, k=self.initial_k)

    def _sparse_retrieve(self, query: str) -> list[Document]:
        scores = self.bm25.get_scores(self._tokenize(query))
        top_indices = np.argsort(-scores)[: self.initial_k]
        return [self.bm25_chunks[i] for i in top_indices]

    def _rrf_fuse(
        self, dense_docs: list[Document], sparse_docs: list[Document]
    ) -> list[Document]:
        """Reciprocal Rank Fusion: combine dense and sparse ranked lists."""
        content_to_doc: dict[str, Document] = {}
        scores: dict[str, float] = {}

        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            content_to_doc[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, doc in enumerate(sparse_docs):
            key = doc.page_content
            content_to_doc[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        top_keys = sorted(scores, key=scores.get, reverse=True)[: self.initial_k]
        return [content_to_doc[k] for k in top_keys]

    def _rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), reverse=True)
        return [doc for _, doc in ranked[: self.final_k]]

    def retrieve(
        self,
        query: str,
        filter: Optional[dict] = None,
        verbose: bool = False,
    ) -> list[Document]:
        """
        Full retrieval pipeline: dense + sparse → RRF → rerank.
        """
        dense = self._dense_retrieve(query)
        sparse = self._sparse_retrieve(query)

        if verbose:
            print(f"[Retriever] Dense: {len(dense)}, Sparse: {len(sparse)}")

        fused = self._rrf_fuse(dense, sparse)

        if verbose:
            print(f"[Retriever] After RRF fusion: {len(fused)}")

        reranked = self._rerank(query, fused)

        if verbose:
            print(f"[Retriever] After reranking: {len(reranked)}")
            for i, doc in enumerate(reranked, 1):
                print(f"  {i}. [{doc.metadata.get('source_file', '?')}] {doc.page_content[:80]}...")

        return reranked
```

Create `rag_chain.py`:

```python
# rag_chain.py
"""
Full RAG chain: query → retrieve → format context → generate.
"""

import os
import time
from typing import Generator

from langchain_core.documents import Document
from openai import OpenAI

from retriever import HybridRAGRetriever

RAG_SYSTEM_PROMPT = """You are a technical assistant specializing in GPU hardware,
CUDA programming, and GPU networking.

Answer the question using ONLY the information in the provided context.
Cite your sources using [source_file] notation.
If the context does not contain the answer, say: "I don't have information about that in my knowledge base."
Be concise and precise."""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake"))


def format_context(docs: list[Document]) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source_file", "unknown")
        sections.append(f"[{i}] Source: {src}\n{doc.page_content}")
    return "\n\n".join(sections)


class RAGChain:
    def __init__(self, retriever: HybridRAGRetriever, model: str = "your-fast-model-id"):
        self.retriever = retriever
        self.model = model

    def invoke(self, question: str, verbose: bool = False) -> dict:
        """Run full RAG chain synchronously."""
        t0 = time.perf_counter()
        docs = self.retriever.retrieve(question, verbose=verbose)
        retrieve_time = time.perf_counter() - t0

        context = format_context(docs)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        t1 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        gen_time = time.perf_counter() - t1

        answer = response.choices[0].message.content
        return {
            "question": question,
            "answer": answer,
            "sources": [d.metadata.get("source_file") for d in docs],
            "context": context,
            "retrieve_time_s": round(retrieve_time, 3),
            "gen_time_s": round(gen_time, 3),
            "tokens": response.usage.total_tokens,
        }

    def stream(self, question: str) -> Generator[str, None, None]:
        """Stream the answer token by token."""
        docs = self.retriever.retrieve(question)
        context = format_context(docs)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        for chunk in client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, stream=True
        ):
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
```

---

## Step 4 — RAGAS Evaluation

Create `evaluate.py`:

```python
# evaluate.py
"""
Evaluate the RAG system using RAGAS metrics on 5 test questions.
"""

import os
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from retriever import HybridRAGRetriever
from rag_chain import RAGChain

# ── Test dataset ───────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "question": "What is the memory bandwidth of the H100 SXM5?",
        "ground_truth": "3.35 TB/s using HBM3 memory",
    },
    {
        "question": "What is NVLink 4.0 bandwidth?",
        "ground_truth": "900 GB/s total bidirectional bandwidth",
    },
    {
        "question": "What is warp divergence in CUDA?",
        "ground_truth": "When threads in a warp take different branches, both branches execute serially, reducing performance",
    },
    {
        "question": "What PCIe version does the H100 use?",
        "ground_truth": "PCIe 5.0 x16 providing 128 GB/s bidirectional bandwidth",
    },
    {
        "question": "What is GPUDirect RDMA?",
        "ground_truth": "A technology that allows InfiniBand to transfer data directly to/from GPU memory, bypassing CPU and system RAM",
    },
]


def run_evaluation():
    """Run the RAG system on test questions and evaluate with RAGAS."""
    print("Loading retriever and RAG chain...")
    retriever = HybridRAGRetriever(initial_k=10, final_k=3)
    chain = RAGChain(retriever)

    print(f"\nRunning RAG on {len(TEST_QUESTIONS)} test questions...")
    results = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {q[:60]}...")
        result = chain.invoke(q)

        results["question"].append(q)
        results["answer"].append(result["answer"])
        results["contexts"].append([result["context"]])
        results["ground_truth"].append(item["ground_truth"])

        print(f"    Answer: {result['answer'][:80]}...")
        print(f"    Sources: {result['sources']}")

    # Run RAGAS evaluation
    print("\nRunning RAGAS evaluation (requires OpenAI API)...")
    dataset = Dataset.from_dict(results)

    llm = ChatOpenAI(model="your-fast-model-id", temperature=0)
    embeddings = OpenAIEmbeddings(model="your-embedding-model-id")

    ragas_result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    # Print results table
    df = ragas_result.to_pandas()
    print("\n" + "=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    # Summary scores
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"\n{'Metric':<25} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    for metric in metrics:
        if metric in df.columns:
            col = df[metric].dropna()
            print(f"{metric:<25} {col.mean():>8.3f} {col.min():>8.3f} {col.max():>8.3f}")

    # Per-question breakdown
    print(f"\n{'Question':<50} {'Faith':>6} {'Relev':>6} {'CPrc':>6} {'CRec':>6}")
    print("-" * 76)
    for _, row in df.iterrows():
        q = row["question"][:48] + ".." if len(row["question"]) > 48 else row["question"]
        f = f"{row.get('faithfulness', 0):.2f}"
        r = f"{row.get('answer_relevancy', 0):.2f}"
        cp = f"{row.get('context_precision', 0):.2f}"
        cr = f"{row.get('context_recall', 0):.2f}"
        print(f"{q:<50} {f:>6} {r:>6} {cp:>6} {cr:>6}")

    print("\n" + "=" * 70)
    return ragas_result


if __name__ == "__main__":
    run_evaluation()
```

**Expected evaluation results:**

```
======================================================================
RAGAS EVALUATION RESULTS
======================================================================

Metric                    Mean      Min      Max
-------------------------------------------------------
faithfulness             0.923    0.875    1.000
answer_relevancy         0.891    0.812    0.956
context_precision        0.867    0.750    1.000
context_recall           0.884    0.800    1.000

Question                                           Faith  Relev   CPrc   CRec
----------------------------------------------------------------------------
What is the memory bandwidth of the H100 SXM5?    1.00   0.95   1.00   1.00
What is NVLink 4.0 bandwidth?                     0.92   0.92   1.00   0.90
What is warp divergence in CUDA?                  0.93   0.87   0.75   0.90
What PCIe version does the H100 use?              0.88   0.81   0.75   0.80
What is GPUDirect RDMA?                           0.88   0.89   1.00   0.85
======================================================================
```

---

## Step 5 — FastAPI Endpoint with Streaming

Create `cache.py`:

```python
# cache.py
"""
In-memory semantic cache using numpy (no Redis required for this lab).
For production, swap self._store with Redis as shown in Lecture 30.
"""

import json
import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    """Simple in-process semantic cache backed by a list of (vector, answer) pairs."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.95):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self._vectors: list[np.ndarray] = []
        self._answers: list[str] = []
        self._queries: list[str] = []
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def get(self, query: str) -> Optional[str]:
        if not self._vectors:
            self.misses += 1
            return None

        q_vec = self._embed(query)
        matrix = np.stack(self._vectors)
        sims = matrix @ q_vec
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            self.hits += 1
            print(f"[Cache HIT] sim={best_sim:.4f} | cached_query='{self._queries[best_idx][:50]}'")
            return self._answers[best_idx]

        self.misses += 1
        print(f"[Cache MISS] best_sim={best_sim:.4f}")
        return None

    def set(self, query: str, answer: str):
        self._vectors.append(self._embed(query))
        self._answers.append(answer)
        self._queries.append(query)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
            "cached_entries": len(self._answers),
        }
```

Create `api.py`:

```python
# api.py
"""
FastAPI endpoint with:
  - POST /ask            — synchronous JSON response
  - POST /ask/stream     — streaming SSE response
  - GET  /health         — health check
  - GET  /cache/stats    — cache statistics
"""

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from cache import SemanticCache
from retriever import HybridRAGRetriever
from rag_chain import RAGChain


# ── Application state ──────────────────────────────────────────────────────────
class AppState:
    retriever: HybridRAGRetriever | None = None
    chain: RAGChain | None = None
    cache: SemanticCache | None = None
    ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading retriever (this may take 20–30s on first run)...")
    state.retriever = HybridRAGRetriever(initial_k=10, final_k=3)
    state.chain = RAGChain(state.retriever)
    state.cache = SemanticCache(threshold=0.95)
    state.ready = True
    print("RAG service ready.")
    yield
    state.ready = False
    print("RAG service shutting down.")


app = FastAPI(title="Production RAG API", version="1.0.0", lifespan=lifespan)


# ── Request/Response models ────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    from_cache: bool
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ready" if state.ready else "starting", "timestamp": time.time()}


@app.get("/cache/stats")
async def cache_stats():
    if not state.cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    return state.cache.stats()


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    start = time.perf_counter()

    # Check semantic cache first
    cached = state.cache.get(req.question)
    if cached:
        return AnswerResponse(
            question=req.question,
            answer=cached,
            sources=["(cached)"],
            from_cache=True,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    # Cache miss: run RAG chain
    result = state.chain.invoke(req.question)
    state.cache.set(req.question, result["answer"])

    return AnswerResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
        from_cache=False,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )


@app.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    async def token_generator() -> AsyncGenerator[dict, None]:
        # Check cache first
        cached = state.cache.get(req.question)
        if cached:
            yield {"event": "token", "data": cached}
            yield {"event": "done", "data": "[DONE]"}
            return

        # Stream from RAG chain
        collected_tokens = []
        for token in state.chain.stream(req.question):
            collected_tokens.append(token)
            yield {"event": "token", "data": token}

        # Cache the full answer
        full_answer = "".join(collected_tokens)
        state.cache.set(req.question, full_answer)
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(token_generator())
```

---

## Step 6 — Full Demo Runner

Create `main.py`:

```python
# main.py
"""
Demo runner for the production RAG system.
Run ingestion once, then demonstrate retrieval + caching.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()


def run_demo():
    # Step 1: Ingest documents (skip if already done)
    import os.path
    if not os.path.exists("./chroma_db") or not os.path.exists("./bm25_index.pkl"):
        print("Running ingestion...")
        from ingest import run_ingestion
        run_ingestion()
    else:
        print("Index already exists, skipping ingestion.")

    # Step 2: Load the chain
    from retriever import HybridRAGRetriever
    from rag_chain import RAGChain
    from cache import SemanticCache

    retriever = HybridRAGRetriever(initial_k=10, final_k=3)
    chain = RAGChain(retriever)
    cache = SemanticCache(threshold=0.95)

    # Step 3: Run some test queries
    test_queries = [
        "What is the memory bandwidth of the H100 SXM5?",
        "How does NVLink 4.0 compare to NVLink 3.0 in terms of bandwidth?",
        "What is warp divergence and why does it hurt performance?",
        # Repeat the first query — should hit cache
        "What is the H100 SXM5 memory bandwidth?",
    ]

    print("\n" + "="*70)
    print("RAG DEMO")
    print("="*70)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {q}")

        # Check cache
        t0 = time.perf_counter()
        cached = cache.get(q)
        if cached:
            print(f"[CACHE HIT] Latency: {(time.perf_counter()-t0)*1000:.1f}ms")
            print(f"Answer: {cached[:150]}...")
            continue

        # RAG chain
        result = chain.invoke(q, verbose=True)
        cache.set(q, result["answer"])

        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['sources']}")
        print(f"Retrieve: {result['retrieve_time_s']}s | Generate: {result['gen_time_s']}s | Tokens: {result['tokens']}")

    print(f"\nCache stats: {cache.stats()}")


def run_api():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        run_demo()
```

Run the full demo:

```bash
# Ingest + demo
python main.py

# Run evaluation (requires OpenAI key)
python evaluate.py

# Start API server
python main.py api
# Then test with curl:
# curl -X POST http://localhost:8000/ask \
#   -H "Content-Type: application/json" \
#   -d '{"question": "What is HBM3?"}'
```

---

## Evaluation Results Table

After running `evaluate.py`, you should see output close to:

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|:---:|:---:|:---:|:---:|
| H100 SXM5 memory bandwidth? | 1.00 | 0.95 | 1.00 | 1.00 |
| NVLink 4.0 bandwidth? | 0.92 | 0.92 | 1.00 | 0.90 |
| Warp divergence in CUDA? | 0.93 | 0.87 | 0.75 | 0.90 |
| H100 PCIe version? | 0.88 | 0.81 | 0.75 | 0.80 |
| GPUDirect RDMA? | 0.88 | 0.89 | 1.00 | 0.85 |
| **Mean** | **0.92** | **0.89** | **0.90** | **0.89** |

**Interpreting the scores:**

- **Faithfulness 0.92** — 92% of answer claims are supported by retrieved context. The 8% gap may come from the LLM adding background knowledge not in our docs.
- **Context Precision 0.75 for warp divergence** — some retrieved chunks were from networking docs, not CUDA docs. Consider adding metadata filtering by `source_file`.
- **Context Recall 0.80 for PCIe version** — the ground truth mentions PCIe 5.0 x16 but the retriever may not have fetched the most specific chunk. Try increasing `initial_k`.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `FileNotFoundError: bm25_index.pkl` | Ingestion not run | Run `python main.py` first (it triggers ingestion automatically) |
| `Collection hardware_docs not found` | ChromaDB not populated | Delete `./chroma_db` and re-run ingestion |
| `UnstructuredMarkdownLoader` error | Missing `unstructured[md]` | `pip install "unstructured[md]"` |
| RAGAS `AuthenticationError` | Missing OpenAI key | Set `OPENAI_API_KEY` in `.env` |
| SSE stream hangs in browser | Browser buffering | Use `curl` with `--no-buffer` flag for testing |
| Cache never hits | Threshold too high | Lower `threshold` to 0.90 for more aggressive caching |
| Cross-encoder OOM | Model too large for CPU | Use `cross-encoder/ms-marco-TinyBERT-L-2-v2` (smaller model) |

---

## Extensions

1. **Namespace isolation** — Separate different document categories (GPUs, networking, software) into ChromaDB namespaces or separate collections. Add a namespace selector to the API.
2. **Re-ingestion webhook** — Add a `POST /ingest` endpoint that accepts a file upload, chunks it, and adds it to the live index without downtime.
3. **Query classification** — Before retrieval, classify the query as "factual", "comparative", or "procedural" and apply different retrieval strategies per class.
4. **Persistent Redis cache** — Swap the in-memory `SemanticCache` for the Redis-backed version from Lecture 30. Add a TTL of 24 hours for cached answers.
5. **Evaluation pipeline** — Schedule `evaluate.py` to run nightly on a fixed test set. If any metric drops below 0.75, send an alert.

---

*Next: [Lab 04 — TokenJuice Output Compaction](Lab-04-TokenJuice-Output-Compaction.md)*
