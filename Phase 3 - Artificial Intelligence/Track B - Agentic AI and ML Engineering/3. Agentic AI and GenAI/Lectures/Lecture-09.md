# Lecture 09 — RAG: Ingestion & Embeddings

**Track B · Agentic AI & GenAI** | [← Lecture 08](Lecture-08.md) | [Next → Lecture 10](Lecture-10.md)

---

## Learning Objectives

By the end of this lecture you will be able to:

1. Load documents from PDF, HTML, and Markdown sources using LangChain loaders.
2. Apply fixed-size, recursive, and semantic chunking strategies and explain the tradeoffs.
3. Generate embeddings with both API-based (OpenAI) and local (sentence-transformers) models.
4. Store and query vectors in Chroma, FAISS, and Pinecone.
5. Filter retrieval results using metadata and namespaces.
6. Assemble a complete, reusable indexing pipeline class.

---

## 1. Document Loading

The first stage of any **RAG system** is getting raw content into a usable format. LangChain provides `BaseLoader` implementations for dozens of file types. All loaders return a list of `Document` objects — each with `.page_content` (string) and `.metadata` (dict).

### 1.1 PDF Loading

```python
# pip install langchain langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()           # one Document per page
    print(f"Loaded {len(docs)} pages from {path}")
    for doc in docs[:2]:
        print(f"  Page {doc.metadata['page']}: {len(doc.page_content)} chars")
    return docs

# Usage
# docs = load_pdf("hardware_manual.pdf")
```

For scanned PDFs that need OCR, swap in `UnstructuredPDFLoader`:

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("scanned.pdf", mode="elements")
docs = loader.load()
# Each element (Title, NarrativeText, Table) becomes a separate Document
```

### 1.2 HTML Loading

```python
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

urls = [
    "https://example.com/docs/intro",
    "https://example.com/docs/api",
]

# Load raw HTML
loader = AsyncHtmlLoader(urls)
raw_docs = loader.load()

# Strip tags, keep only meaningful text
bs_transformer = BeautifulSoupTransformer()
docs = bs_transformer.transform_documents(
    raw_docs,
    tags_to_extract=["p", "li", "h1", "h2", "h3", "code"],
    remove_unwanted_tags=["script", "style", "nav", "footer"],
)
print(f"Extracted {len(docs)} documents")
```

### 1.3 Markdown Loading

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import glob

def load_markdown_directory(directory: str):
    all_docs = []
    for path in glob.glob(f"{directory}/**/*.md", recursive=True):
        loader = UnstructuredMarkdownLoader(path, mode="elements")
        docs = loader.load()
        # Enrich metadata
        for doc in docs:
            doc.metadata["source_file"] = path
            doc.metadata["file_type"] = "markdown"
        all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} elements from {directory}")
    return all_docs
```

---

## 2. Chunking Strategies

Raw documents are almost always too long to embed as-is. **Chunking** splits them into pieces small enough for an embedding model while **preserving enough context** for retrieval.

### 2.1 Fixed-Size Chunking

The simplest approach: split every N characters with an **overlap** to prevent context from being cut at boundaries.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50,     # overlap between consecutive chunks
    separator="\n",       # split on newlines first
)

text = """
GPUs are massively parallel processors. They contain thousands of small cores
optimized for floating-point arithmetic. Modern AI accelerators like the H100
push this further with dedicated Tensor Cores for matrix multiplication.

DRAM bandwidth is the primary bottleneck for large language model inference.
A100 provides 2 TB/s HBM2e bandwidth. H100 SXM5 delivers 3.35 TB/s HBM3.
""".strip()

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars | '{chunk[:60]}...'")
```

**When to use:** Homogeneous text (logs, emails) where sentence boundaries do not matter much.

### 2.2 Recursive Character Chunking

Uses a hierarchy of separators (`\n\n`, `\n`, ` `, `""`) so it tries to break on paragraph boundaries first, then sentences, then words.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("docs/architecture.md")
docs = loader.load()
chunks = splitter.split_documents(docs)

print(f"Original: {len(docs)} docs → {len(chunks)} chunks")
print(f"Chunk 0 metadata: {chunks[0].metadata}")
print(f"Chunk 0 preview: {chunks[0].page_content[:120]}")
```

**When to use:** The default choice for most unstructured text. Respects natural language structure.

### 2.3 Semantic Chunking

Instead of splitting by character count, **semantic chunking** embeds consecutive sentences and splits whenever the **cosine similarity** between adjacent sentences drops below a threshold. This keeps topically coherent content together.

```python
# pip install langchain-experimental sentence-transformers

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",   # "standard_deviation" | "interquartile"
    breakpoint_threshold_amount=90,           # split when similarity drops to 90th percentile
)

text = open("docs/long_article.txt").read()
chunks = splitter.create_documents([text])
print(f"Created {len(chunks)} semantic chunks")
for c in chunks[:3]:
    print(f"  {len(c.page_content)} chars: {c.page_content[:80]}...")
```

**When to use:** Technical documentation, research papers, or any content where topic shifts should define chunk boundaries. Costs more compute than fixed-size splitting.

---

## 3. Embedding Models

An **embedding model** converts text into a **dense vector** (list of floats). Semantically similar texts produce vectors that are close in **cosine distance**.

### 3.1 OpenAI your-embedding-model-id (API)

```python
# pip install openai langchain-openai

import os
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="your-embedding-model-id",  # 1536 dims, cheap & fast
    api_key=os.environ["OPENAI_API_KEY"],
)

texts = [
    "CUDA cores execute floating-point operations in parallel.",
    "Tensor Cores accelerate matrix multiplication for AI workloads.",
    "The weather in Paris is often cloudy in November.",
]

vectors = embeddings.embed_documents(texts)
print(f"Embedding shape: {len(vectors)} x {len(vectors[0])}")

# Single query embedding
query_vec = embeddings.embed_query("What are Tensor Cores?")
print(f"Query vector dim: {len(query_vec)}")

# Compute cosine similarity manually
import numpy as np

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

for i, text in enumerate(texts):
    sim = cosine_sim(query_vec, vectors[i])
    print(f"  sim({text[:40]}...) = {sim:.3f}")
```

**Cost note:** your-embedding-model-id costs $0.02 per million tokens. For a 100 MB corpus (~25M tokens), that is ~$0.50.

### 3.2 Local Embeddings with sentence-transformers

```python
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims, ~22 MB

texts = [
    "CUDA cores execute floating-point operations in parallel.",
    "Tensor Cores accelerate matrix multiplication for AI workloads.",
    "The weather in Paris is often cloudy in November.",
]

# Batch encode
vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
print(f"Shape: {vectors.shape}")  # (3, 384)

query_vec = model.encode("What are Tensor Cores?", normalize_embeddings=True)

# With normalized vectors, dot product == cosine similarity
similarities = vectors @ query_vec
for text, sim in zip(texts, similarities):
    print(f"  {sim:.3f}  {text[:50]}")
```

### 3.3 Comparison Table

| Model | Dims | Speed | Cost | Quality |
|---|---|---|---|---|
| your-embedding-model-id | 1536 | ~500 docs/s (API) | $0.02/1M tokens | Very good |
| text-embedding-3-large | 3072 | ~200 docs/s (API) | $0.13/1M tokens | Best OpenAI |
| all-MiniLM-L6-v2 | 384 | ~10k docs/s (CPU) | Free | Good |
| all-mpnet-base-v2 | 768 | ~2k docs/s (CPU) | Free | Very good |
| bge-large-en-v1.5 | 1024 | ~1k docs/s (CPU) | Free | Excellent |

**Rule of thumb:** Use local models for prototyping and high-volume offline indexing. Use OpenAI embeddings when query latency and accuracy are critical.

---

## 4. Vector Stores

### 4.1 Chroma (Local, No Setup)

```python
# pip install chromadb langchain-chroma

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create / open a persistent store
vectorstore = Chroma(
    collection_name="hardware_docs",
    embedding_function=embedding_fn,
    persist_directory="./chroma_db",   # omit for in-memory
)

# Add documents
docs = [
    Document(page_content="H100 SXM5 has 80 GB HBM3 memory.", metadata={"source": "h100.md", "section": "memory"}),
    Document(page_content="A100 offers 312 TFLOPS of BF16 performance.", metadata={"source": "a100.md", "section": "compute"}),
    Document(page_content="NVLink 4.0 provides 900 GB/s bidirectional bandwidth.", metadata={"source": "nvlink.md", "section": "interconnect"}),
    Document(page_content="CUDA 12.0 introduced TMA (Tensor Memory Accelerator).", metadata={"source": "cuda.md", "section": "software"}),
]

vectorstore.add_documents(docs)
print(f"Collection size: {vectorstore._collection.count()}")

# Basic similarity search
results = vectorstore.similarity_search("GPU memory bandwidth", k=2)
for r in results:
    print(f"  [{r.metadata['source']}] {r.page_content}")

# With scores (lower L2 distance = more similar)
results_with_scores = vectorstore.similarity_search_with_score("GPU memory bandwidth", k=2)
for doc, score in results_with_scores:
    print(f"  score={score:.4f}  {doc.page_content}")
```

### 4.2 FAISS (High-Performance Local)

```python
# pip install faiss-cpu langchain-community

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts = [
    "H100 SXM5 has 80 GB HBM3 memory.",
    "A100 offers 312 TFLOPS of BF16 performance.",
    "NVLink 4.0 provides 900 GB/s bidirectional bandwidth.",
    "CUDA 12.0 introduced TMA (Tensor Memory Accelerator).",
]
metadatas = [
    {"source": "h100.md"},
    {"source": "a100.md"},
    {"source": "nvlink.md"},
    {"source": "cuda.md"},
]

vectorstore = FAISS.from_texts(texts, embedding_fn, metadatas=metadatas)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index", embedding_fn, allow_dangerous_deserialization=True
)

results = vectorstore.similarity_search("interconnect bandwidth", k=2)
for r in results:
    print(f"  [{r.metadata['source']}] {r.page_content}")
```

### 4.3 Pinecone (Managed Cloud)

```python
# pip install pinecone-client langchain-pinecone

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "hardware-docs"
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

embedding_fn = OpenAIEmbeddings(model="your-embedding-model-id")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_fn)

from langchain_core.documents import Document
docs = [
    Document(page_content="H100 SXM5 has 80 GB HBM3.", metadata={"source": "h100", "type": "gpu"}),
    Document(page_content="A100 BF16 312 TFLOPS.", metadata={"source": "a100", "type": "gpu"}),
]
vectorstore.add_documents(docs)

results = vectorstore.similarity_search("GPU memory", k=2)
for r in results:
    print(r.page_content)
```

---

## 5. Metadata Filtering and Namespacing

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="filtered_demo", embedding_function=embedding_fn)

docs = [
    Document(page_content="H100 memory bandwidth is 3.35 TB/s.", metadata={"product": "H100", "year": 2023, "category": "memory"}),
    Document(page_content="A100 memory bandwidth is 2 TB/s.", metadata={"product": "A100", "year": 2020, "category": "memory"}),
    Document(page_content="H100 uses NVLink 4.0.", metadata={"product": "H100", "year": 2023, "category": "interconnect"}),
    Document(page_content="A100 uses NVLink 3.0.", metadata={"product": "A100", "year": 2020, "category": "interconnect"}),
]
vectorstore.add_documents(docs)

# Filter to only H100 documents
results = vectorstore.similarity_search(
    "bandwidth",
    k=5,
    filter={"product": "H100"},
)
print("H100 only:")
for r in results:
    print(f"  {r.page_content}")

# Compound filter (Chroma supports $and, $or, $eq, $ne, $gt, $gte, $lt, $lte, $in)
results = vectorstore.similarity_search(
    "bandwidth",
    k=5,
    filter={"$and": [{"year": {"$gte": 2023}}, {"category": "memory"}]},
)
print("\nH100 memory docs (year >= 2023):")
for r in results:
    print(f"  {r.page_content}")
```

**Pinecone namespacing** allows **tenant isolation** — each namespace is a separate partition:

```python
# Pinecone: use namespace parameter to isolate tenants
vectorstore.add_documents(docs, namespace="customer_acme")
results = vectorstore.similarity_search("bandwidth", k=2, namespace="customer_acme")
```

---

## 6. Complete Indexing Pipeline

Putting it all together into a **reusable class**:

```python
# pip install langchain langchain-community langchain-chroma
# pip install sentence-transformers pypdf unstructured[md]

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    AsyncHtmlLoader,
)
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class IndexingPipeline:
    """
    End-to-end RAG ingestion pipeline.

    Usage:
        pipeline = IndexingPipeline(persist_dir="./my_index")
        pipeline.ingest_directory("./docs", file_types=["*.md", "*.pdf"])
        results = pipeline.search("What is HBM3?", k=3)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Track ingested file hashes to avoid re-indexing unchanged files
        self._hash_file = Path(persist_dir) / "ingested_hashes.json"
        self._hashes: dict = self._load_hashes()

    def _load_hashes(self) -> dict:
        if self._hash_file.exists():
            return json.loads(self._hash_file.read_text())
        return {}

    def _save_hashes(self):
        self._hash_file.parent.mkdir(parents=True, exist_ok=True)
        self._hash_file.write_text(json.dumps(self._hashes, indent=2))

    def _file_hash(self, path: str) -> str:
        content = Path(path).read_bytes()
        return hashlib.md5(content).hexdigest()

    def _load_file(self, path: str) -> List[Document]:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        elif ext in (".md", ".markdown"):
            return UnstructuredMarkdownLoader(path, mode="elements").load()
        elif ext in (".txt",):
            from langchain_community.document_loaders import TextLoader
            return TextLoader(path).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def ingest_file(self, path: str, extra_metadata: Optional[dict] = None) -> int:
        """Ingest a single file. Returns number of chunks added."""
        current_hash = self._file_hash(path)
        if self._hashes.get(path) == current_hash:
            print(f"  Skipping (unchanged): {path}")
            return 0

        docs = self._load_file(path)
        chunks = self.splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source_path"] = path
            chunk.metadata["file_name"] = Path(path).name
            if extra_metadata:
                chunk.metadata.update(extra_metadata)

        self.vectorstore.add_documents(chunks)
        self._hashes[path] = current_hash
        self._save_hashes()

        print(f"  Ingested: {path} → {len(chunks)} chunks")
        return len(chunks)

    def ingest_directory(
        self,
        directory: str,
        file_types: Optional[List[str]] = None,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        """Ingest all matching files in a directory tree."""
        if file_types is None:
            file_types = ["*.md", "*.pdf", "*.txt"]

        total = 0
        for pattern in file_types:
            for path in glob.glob(f"{directory}/**/{pattern}", recursive=True):
                try:
                    total += self.ingest_file(path, extra_metadata)
                except Exception as e:
                    print(f"  ERROR ingesting {path}: {e}")

        print(f"\nTotal chunks added: {total}")
        print(f"Collection size: {self.vectorstore._collection.count()}")
        return total

    def ingest_urls(self, urls: List[str]) -> int:
        """Ingest content from a list of URLs."""
        loader = AsyncHtmlLoader(urls)
        raw_docs = loader.load()
        transformer = BeautifulSoupTransformer()
        clean_docs = transformer.transform_documents(
            raw_docs,
            tags_to_extract=["p", "li", "h1", "h2", "h3", "code"],
        )
        chunks = self.splitter.split_documents(clean_docs)
        self.vectorstore.add_documents(chunks)
        print(f"Ingested {len(urls)} URLs → {len(chunks)} chunks")
        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """Similarity search with optional metadata filter."""
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def get_retriever(self, k: int = 4, filter: Optional[dict] = None):
        """Return a LangChain retriever for use in chains."""
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def stats(self) -> dict:
        count = self.vectorstore._collection.count()
        return {
            "total_chunks": count,
            "indexed_files": len(self._hashes),
            "persist_dir": self.persist_dir,
        }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    # Create some fake docs
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(f"{tmpdir}/h100.md").write_text(
            "# H100\nThe H100 GPU has 80 GB HBM3. Bandwidth is 3.35 TB/s.\n"
            "It contains 16896 CUDA cores and 528 Tensor Cores.\n"
        )
        Path(f"{tmpdir}/a100.md").write_text(
            "# A100\nThe A100 GPU has 80 GB HBM2e. Bandwidth is 2 TB/s.\n"
            "BF16 performance reaches 312 TFLOPS.\n"
        )

        pipeline = IndexingPipeline(
            persist_dir=f"{tmpdir}/index",
            chunk_size=200,
            chunk_overlap=40,
        )
        pipeline.ingest_directory(tmpdir, file_types=["*.md"])
        print("\nStats:", pipeline.stats())

        print("\nSearch results for 'memory bandwidth':")
        for doc in pipeline.search("memory bandwidth", k=2):
            print(f"  [{doc.metadata.get('file_name')}] {doc.page_content.strip()}")
```

---

## Key Takeaways

- **Loading**: Use `PyPDFLoader` for PDFs, `UnstructuredMarkdownLoader` for Markdown, `AsyncHtmlLoader` + `BeautifulSoupTransformer` for web content.
- **Chunking**: Start with `RecursiveCharacterTextSplitter` (chunk_size=400, overlap=80). Graduate to semantic chunking for research-grade quality.
- **Embeddings**: Local `all-MiniLM-L6-v2` is free and fast enough for millions of docs. Use OpenAI embeddings when query precision matters.
- **Vector stores**: Chroma for local development, FAISS for high-performance offline search, Pinecone for production multi-tenant systems.
- **Metadata**: Always store `source`, `section`, and any filterable fields at ingest time — retroactively adding metadata is painful.
- **Incremental indexing**: Hash files at ingest to skip re-processing unchanged documents.

---

## Exercises

### Exercise 1 — Chunking Comparison

Take any long text file (at least 2000 words). Split it with all three strategies: `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, and `SemanticChunker`. For each strategy, print the number of chunks, the mean chunk length, and the standard deviation of chunk lengths. Which strategy produces the most uniform chunks? Which produces the most semantically coherent ones? Write a one-paragraph explanation.

### Exercise 2 — Embedding Distance Analysis

Embed the following five sentences using `all-MiniLM-L6-v2`. Build the full 5×5 cosine similarity matrix and display it as a formatted table. Identify the pair of sentences that is most similar and the pair that is most dissimilar. Explain whether the results match your intuition.

```
"The H100 GPU achieves 3.35 TB/s memory bandwidth."
"HBM3 provides high-bandwidth memory for AI accelerators."
"Python is a dynamically typed programming language."
"CUDA enables general-purpose computing on NVIDIA GPUs."
"Renewable energy sources include solar and wind power."
```

### Exercise 3 — Pipeline Extension

Extend the `IndexingPipeline` class with two new methods:

1. `delete_file(path: str)` — removes all chunks from the vector store that came from `path` and removes its hash entry.
2. `search_with_sources(query: str, k: int) -> list[dict]` — returns results as `[{"content": ..., "source": ..., "score": ...}]`.

Test both methods in a short `main` block using at least three markdown files.

---

*Next: [Lecture 10 — RAG: Retrieval & Reranking](Lecture-10.md)*
