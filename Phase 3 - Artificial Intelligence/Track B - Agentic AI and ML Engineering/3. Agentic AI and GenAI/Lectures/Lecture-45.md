# Lecture 45 - Qdrant, pgvector, and Embedding Model Selection

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 44](Lecture-44.md) | **Next:** [Lab 01](Lab-01-Research-Agent.md)

---

Qdrant and pgvector solve the same broad problem:

```text
Given a query embedding, find the stored vectors that are most similar.
```

They are **not the same kind of system**.

Qdrant is a **dedicated vector search engine**.

pgvector is a **PostgreSQL extension** that adds vector search to Postgres.

The practical rule:

```text
Use pgvector when SQL integration is the main constraint.
Use Qdrant when retrieval performance and vector-search features are the main constraint.
```

The same logic applies to embedding models.

There is **no universal "best embedding model."**

There is a best model for:

- your corpus
- your languages
- your chunk size
- your latency budget
- your memory budget
- your deployment target
- your query distribution
- your relevance metric

For local and edge RAG, the winning design is usually:

```text
compact embedding model
  + strong chunking
  + good metadata filters
  + hybrid retrieval when needed
  + reranking
  + small grounded generator
```

not:

```text
giant embedding model
  + unfiltered top-k
  + huge prompt
  + hope the LLM fixes retrieval mistakes
```

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain what a vector database does in a RAG system.
2. Explain the difference between Qdrant and pgvector.
3. Describe dense, sparse, and multi-vector retrieval.
4. Explain HNSW and IVFFlat at a practical systems level.
5. Choose Qdrant or pgvector based on architecture, scale, filters, and operations.
6. Compare Granite embeddings with BGE, E5, OpenAI, Cohere, Voyage, and other alternatives.
7. Design an embedding evaluation plan instead of trusting generic leaderboards.
8. Estimate vector storage pressure from embedding dimension and data type.
9. Plan an embedding migration without corrupting retrieval quality.

---

## 1. The core RAG storage problem

A RAG system has two kinds of data:

```text
source data:
  docs, markdown, PDFs, code, tickets, emails, manuals

retrieval data:
  chunks, embeddings, metadata, indexes, scores, citations
```

The LLM does **not search your raw documents directly**.

A typical pipeline is:

```text
document
  -> chunk
  -> embed each chunk
  -> store vector + chunk text + metadata
  -> embed user query
  -> search nearest vectors
  -> rerank/filter
  -> send selected evidence to LLM
```

The vector store owns this part:

```text
stored vectors + metadata + nearest-neighbor index + query API
```

Each stored item is usually a "point" or row:

```json
{
  "id": "doc-17#chunk-03",
  "vector": [0.012, -0.044, 0.331],
  "payload": {
    "document_id": "doc-17",
    "path": "manuals/orin/power.md",
    "section": "Thermals",
    "language": "en",
    "created_at": "2026-05-28",
    "source_hash": "..."
  },
  "text": "The chunk text may live here or in another store."
}
```

The **metadata matters as much as the vector**.

Example query:

```text
"How do I reduce Jetson Orin power draw during idle?"
```

Good retrieval does not just ask:

```text
Which chunks are semantically close?
```

It asks:

```text
Which chunks are semantically close,
inside the right product docs,
in the right version,
in the right language,
from trusted sources,
and recent enough to answer safely?
```

That is why **vector search and filtering need to be designed together**.

---

## 2. What is Qdrant?

Qdrant is a **standalone vector database and search engine**.

It is designed around collections of points:

```text
collection
  -> points
      -> id
      -> vector or named vectors
      -> payload metadata
```

The key design idea:

```text
Qdrant is optimized for vector-native retrieval first.
```

Useful features:

- dense vector search
- sparse vector search
- named vectors
- multi-vector retrieval patterns
- payload metadata filtering
- payload indexes
- HNSW indexing
- quantization options
- horizontal scaling through sharding and replication
- HTTP/gRPC APIs
- client libraries
- local, cloud, private-cloud, and edge deployment patterns

Qdrant is a good fit when retrieval is a **product-critical path**.

Examples:

- local RAG server
- semantic search API
- AI coding assistant over repositories
- recommendation systems
- hybrid search over technical docs
- multi-tenant knowledge retrieval
- edge assistant with a local vector service

The important distinction:

```text
Qdrant is not your relational database.
It is your vector retrieval engine.
```

You may still keep canonical business data in Postgres, SQLite, object storage, or a document store.

In that design, Qdrant stores:

```text
id + vector + retrieval metadata + optional text snippet
```

The source system stores:

```text
full document + permissions + owner + audit history + business state
```

That separation is **normal**.

---

## 3. What is pgvector?

pgvector is an **extension for PostgreSQL**.

It adds **vector types, distance operators, and approximate indexes** to Postgres.

The key design idea:

```text
pgvector brings vector search into an existing SQL database.
```

A minimal table:

```sql
CREATE EXTENSION vector;

CREATE TABLE document_chunks (
  id bigserial PRIMARY KEY,
  document_id text NOT NULL,
  path text NOT NULL,
  chunk text NOT NULL,
  embedding vector(384)
);
```

A basic nearest-neighbor query:

```sql
SELECT id, path, chunk
FROM document_chunks
ORDER BY embedding <=> $1
LIMIT 8;
```

`<=>` is cosine distance.

The biggest advantage is **architectural simplicity**:

```text
same database
same backup path
same SQL permissions
same transactions
same app connection pool
same operational team
```

This is valuable if your application is already centered on Postgres.

pgvector is a good fit when:

- you already use PostgreSQL
- your vector corpus is moderate
- you want SQL joins with vector results
- relational consistency matters
- operational simplicity matters more than specialized retrieval features
- vector search is a feature, not the whole product

Example:

```sql
SELECT c.id, c.path, c.chunk, p.owner_id
FROM document_chunks c
JOIN projects p ON p.id = c.project_id
WHERE p.organization_id = $org_id
ORDER BY c.embedding <=> $query_embedding
LIMIT 8;
```

That query is the reason pgvector exists.

You can combine **vector similarity with normal relational conditions** in one SQL path.

---

## 4. Qdrant vs pgvector: the real comparison

| Dimension | Qdrant | pgvector |
|---|---|---|
| System type | Dedicated vector database | PostgreSQL extension |
| Best default use | Retrieval-heavy AI systems | SQL-first apps adding vector search |
| API | HTTP/gRPC/client libraries | SQL |
| Data model | Collections, points, vectors, payloads | Tables, rows, vector columns |
| Scaling model | Standalone service or cluster | Scale PostgreSQL |
| Filtering | Payload filtering and payload indexes | SQL `WHERE`, partial indexes, partitioning |
| Hybrid search | Dense + sparse + multi-representation patterns | Combine with Postgres full-text search and rank fusion |
| Operations | Extra service to deploy and monitor | Uses existing Postgres operations |
| Strength | Search features and vector-native performance | Simplicity and relational integration |
| Risk | Data sync with source DB | Postgres can become overloaded |

The decision is **not ideological**.

Ask:

```text
Is vector retrieval a side feature of my SQL app,
or is it a core runtime service?
```

If it is a side feature:

```text
pgvector is often enough.
```

If it is a core runtime service:

```text
Qdrant is usually the cleaner architecture.
```

---

## 5. Dense, sparse, and multi-vector retrieval

Vector search is **not one thing**.

There are **several retrieval representations**.

### Dense vectors

**Dense vectors** are fixed-length arrays of floats.

Example:

```text
384 dimensions
768 dimensions
1024 dimensions
1536 dimensions
3072 dimensions
```

They capture semantic similarity.

Good at:

- paraphrases
- conceptual similarity
- multilingual semantic search
- fuzzy document retrieval
- "meaning" rather than exact words

Bad at:

- exact identifiers
- error codes
- part numbers
- rare API names
- very precise keyword constraints

Example failure:

```text
Query: "NV_ERR_INVALID_STATE"
```

A dense model might retrieve generic "invalid state" content and miss the exact error-code page.

### Sparse vectors

**Sparse vectors** are high-dimensional vectors where most values are zero.

They are closer to **keyword and lexical retrieval**.

Examples:

- BM25
- SPLADE
- learned sparse retrievers

Good at:

- exact keywords
- identifiers
- names
- rare terms
- technical symbols

Bad at:

- paraphrase-only queries
- cross-lingual semantic matching
- conceptual retrieval when words do not overlap

### Multi-vector retrieval

**Multi-vector systems** store multiple vectors for one document or chunk.

Examples:

- one vector per passage segment
- ColBERT-style late interaction vectors
- image + text vectors
- title vector + body vector
- query-specific representations

Good at:

- precise matching inside long documents
- retrieval where one single pooled vector loses details
- high-quality search over complex docs

Cost:

- more storage
- more compute
- more complicated ranking
- more operational complexity

### Hybrid retrieval

**Hybrid retrieval** combines dense and sparse signals.

Simple pattern:

```text
dense top 20
+ sparse/BM25 top 20
-> reciprocal rank fusion
-> rerank top 10
-> keep top 3
```

Why this works:

```text
dense catches meaning
sparse catches exact terms
reranker chooses final evidence
```

For technical docs, codebases, and enterprise manuals, hybrid retrieval is **often better than dense-only retrieval**.

---

## 6. HNSW and IVFFlat

Nearest-neighbor search has two broad modes:

```text
exact search:
  compare query against every vector

approximate search:
  use an index to find likely nearest neighbors faster
```

Exact search has perfect recall but becomes expensive as the corpus grows.

Approximate nearest neighbor search **trades some recall for speed**.

### HNSW

HNSW means **Hierarchical Navigable Small World**.

At a practical level:

```text
Build a graph where nearby vectors are connected.
Search by walking the graph toward better candidates.
```

Strengths:

- strong speed/recall tradeoff
- works well for many production search workloads
- no separate training step
- can be built before or as data arrives

Costs:

- more memory than simpler indexes
- slower index build than IVFFlat
- parameters matter

Important knobs:

```text
m:
  graph connectivity

ef_construction:
  candidate list size during build

ef_search:
  candidate list size during query
```

Increasing `ef_search` usually **improves recall but increases latency**.

### IVFFlat

IVFFlat means **inverted file flat**.

At a practical level:

```text
Cluster vectors into lists.
At query time, search only the closest lists.
```

Strengths:

- faster build than HNSW
- lower memory pressure
- useful when index size matters

Costs:

- usually weaker speed/recall tradeoff than HNSW
- requires representative data before index creation
- needs tuning of lists/probes

Important knobs:

```text
lists:
  number of partitions

probes:
  number of lists searched per query
```

Increasing `probes` improves recall but increases latency.

### Practical guidance

For most app-level RAG:

```text
Start with HNSW.
Measure recall@k and latency.
Tune search width before changing database.
```

Use IVFFlat when:

- memory is tighter
- build time matters
- data is mostly static
- you know how to tune list/probe tradeoffs

---

## 7. Filtering is where systems diverge

RAG **needs filters**.

Examples:

```text
only docs user can access
only version 2.1 docs
only English docs
only source = official_manual
only product = Jetson Orin
only updated after 2026-01-01
```

**Bad filtering can break retrieval.**

The hard case:

```text
Find nearest neighbors, but only among 2% of the corpus.
```

There are three common strategies:

```text
pre-filter:
  reduce candidate set first, then vector search

post-filter:
  vector search first, then filter results

integrated filtered ANN:
  search the vector index with filter awareness
```

Post-filtering can **silently reduce recall**.

Example:

```text
top_k = 10
filter matches 10% of rows
approximate index returns 10 candidates
after filtering, only 1 candidate remains
```

That is **not a language-model problem**.

That is a **retrieval planning problem**.

Qdrant's payload indexes are designed to make filtered vector search a first-class retrieval path.

pgvector uses SQL filtering, partial indexes, partitioning, iterative scans, and planner behavior to manage this problem.

Both can work.

But when your product depends heavily on filtered ANN retrieval, **test this explicitly**.

---

## 8. Qdrant architecture patterns

### Pattern A: local sidecar

Good for local agents and edge RAG.

```text
OpenClaw / local app
  -> Qdrant on localhost
  -> local embedding model
  -> local generator
```

Benefits:

- simple network boundary
- local data
- good retrieval performance
- easy replacement of app database

Risks:

- another process to supervise
- data sync if canonical docs live elsewhere

### Pattern B: retrieval service

Good for production AI apps.

```text
agent runtime
  -> retrieval API
      -> Qdrant
      -> reranker
      -> citation formatter
```

Benefits:

- one retrieval contract
- service-level caching
- centralized logging
- easier A/B tests

Risks:

- more service architecture
- need clear access-control enforcement

### Pattern C: edge cache of central index

Good for remote/local-first systems.

```text
central corpus
  -> sync selected docs
  -> edge Qdrant collection
  -> local agent queries edge index
```

Benefits:

- lower latency
- private/offline operation
- reduced cloud dependency

Risks:

- sync correctness
- stale docs
- permission drift

---

## 9. pgvector architecture patterns

### Pattern A: SQL app with vector search

Good when Postgres is already the source of truth.

```text
web app
  -> Postgres
      -> relational tables
      -> pgvector columns
      -> full-text search
```

Benefits:

- simplest architecture
- easy joins
- one backup/restore path
- one permission model

Risks:

- vector workload competes with OLTP workload
- index tuning can affect database resources
- horizontal vector scaling is not as clean as a dedicated vector service

### Pattern B: hybrid SQL retrieval

Combine full-text search and vector search.

```sql
-- Dense candidates
SELECT id, 1.0 / (60 + row_number() OVER ()) AS dense_score
FROM document_chunks
ORDER BY embedding <=> $query_embedding
LIMIT 50;

-- Text candidates
SELECT id, ts_rank_cd(textsearch, query) AS text_score
FROM document_chunks, plainto_tsquery($query_text) query
WHERE textsearch @@ query
LIMIT 50;
```

Then fuse in SQL or app code:

```text
reciprocal rank fusion
cross-encoder rerank
weighted score fusion
```

Benefits:

- no separate search engine
- strong for SQL-heavy apps
- easy to filter by relational state

Risks:

- more query complexity
- planner/index tuning matters
- recall must be measured

---

## 10. Embedding model selection

An embedding model **maps text to a vector**.

Different models optimize for different tradeoffs:

```text
quality
latency
dimension
context length
license
language coverage
code retrieval
domain retrieval
image/document support
local deployability
cloud API convenience
```

The **first mistake** is asking:

```text
What is the best embedding model?
```

Ask instead:

```text
What is the best embedding model for this corpus and deployment budget?
```

---

## 11. Granite embeddings

Granite embeddings are **IBM embedding models** for retrieval and search.

The useful local/edge target from Lecture 44:

```text
ibm-granite/granite-embedding-97m-multilingual-r2
```

Why it is attractive:

- compact 97M-class model
- multilingual retrieval
- code retrieval support
- long context for an embedding model
- Apache-2.0 license
- practical deployment paths
- good memory/latency fit for edge RAG

Use Granite 97M when:

- local/private RAG matters
- multilingual retrieval matters
- you want permissive licensing
- memory is constrained
- you want a compact default for Jetson/edge

Use Granite 311M when:

- server resources are available
- retrieval quality is more important than low memory
- the corpus is harder or more multilingual
- latency budget allows a larger encoder

The important point:

```text
Granite 97M is a strong default for efficient local RAG,
not a universal winner for every retrieval task.
```

---

## 12. Open-source alternatives to Granite

### BGE-M3

`BAAI/bge-m3` is a **strong open model** when you want one model that supports:

- dense retrieval
- sparse retrieval
- multi-vector retrieval
- multilingual retrieval
- long-ish input up to 8192 tokens

Use BGE-M3 when:

- hybrid retrieval matters
- you want dense + sparse from one model family
- multilingual search matters
- you can afford more compute than a tiny edge model

Tradeoff:

```text
more retrieval capability
usually more runtime cost than compact embedding models
```

### Multilingual E5

`intfloat/multilingual-e5-large` is a **mature multilingual dense embedding model**.

Use E5 when:

- multilingual text retrieval is central
- you want a widely used baseline
- 512-token truncation is acceptable for your chunks
- you can afford a larger encoder

Tradeoff:

```text
strong baseline, but shorter context than long-context embedding models
```

### Nomic Embed

Nomic embedding models are useful **open-weight baselines**, especially for local development and reproducible experiments.

Use Nomic-style models when:

- you want local inference
- English retrieval is enough
- you need simple open-weight deployment

### Jina embeddings

Jina embedding models are useful when you care about:

- multilingual retrieval
- multimodal retrieval in newer model families
- code/doc retrieval tasks
- deployment flexibility

Use Jina when the corpus includes varied web, code, or multimodal-ish documents and you are willing to test model-specific behavior.

### Snowflake Arctic Embed

Snowflake Arctic Embed models are another useful open retrieval family.

Use them when:

- you want strong open retrieval baselines
- English or multilingual enterprise retrieval is the target
- you are comparing several open models under the same evaluation harness

---

## 13. API-based embedding alternatives

API embeddings are useful when you want **quality and simplicity more than local control**.

### OpenAI embeddings

OpenAI's embedding models are **simple to operate** through an API.

Use them when:

- cloud API use is acceptable
- you want strong general-purpose retrieval quality
- you do not want to host embedding infrastructure
- latency and cost are acceptable

Tradeoffs:

- external API dependency
- token cost
- privacy/compliance review
- provider lock-in

### Cohere Embed

Cohere Embed v4 is useful for:

- multilingual search
- business documents
- image/document screenshot embeddings
- configurable output dimensions
- compressed output types

Use Cohere when:

- enterprise document retrieval matters
- multimodal document surfaces matter
- you want managed embedding infrastructure

### Voyage embeddings

Voyage models are useful for:

- high-quality managed retrieval
- code retrieval
- finance/law/domain-specific retrieval
- configurable dimensions and output dtypes in newer model families

Use Voyage when:

- retrieval quality is critical
- cloud API is acceptable
- your domain matches one of their specialized models

---

## 14. Embedding recommendation matrix

Use this as a starting point, not a law.

| Use case | Good starting model |
|---|---|
| Jetson/local multilingual RAG | Granite 97M Multilingual R2 |
| Local hybrid retrieval | BGE-M3 |
| Mature multilingual dense baseline | Multilingual E5 |
| Server-side IBM/open enterprise stack | Granite 311M Multilingual R2 |
| Cloud general-purpose retrieval | OpenAI embedding model or Voyage general model |
| Cloud enterprise document search | Cohere Embed v4 |
| Code-heavy retrieval | BGE-M3, Granite, or Voyage code model |
| Image-rich PDFs/slides/screenshots | Cohere Embed v4 or a dedicated multimodal document retriever |
| SQL-only prototype | Any embedding model + pgvector |
| Retrieval product/API | Embedding model + Qdrant + reranker |

The real answer should come from your **evaluation set**.

---

## 15. Vector database alternatives

Qdrant and pgvector are not the only options.

| Tool | Best fit |
|---|---|
| Milvus | large-scale vector infrastructure, distributed retrieval |
| Weaviate | semantic app layer, hybrid search, GraphQL/module ecosystem |
| Pinecone | managed vector DB with low operational overhead |
| Elasticsearch/OpenSearch | text search first, vector search added to existing search stack |
| LanceDB | embedded/serverless-style vector storage, data/AI workflows |
| Chroma | local development and prototypes |
| FAISS | library-level vector indexing, not a full database |

Practical guidance:

```text
Prototype:
  Chroma, LanceDB, pgvector, or local Qdrant

SQL app:
  pgvector

Production retrieval service:
  Qdrant, Milvus, Weaviate, Pinecone

Text-search-heavy app:
  Elasticsearch/OpenSearch or hybrid Qdrant

Lowest-level custom indexing:
  FAISS
```

For this roadmap, focus on Qdrant and pgvector first because they represent the two most common architecture choices:

```text
dedicated vector service vs SQL-integrated vector search
```

---

## 16. Storage and memory math

Embedding dimension **affects storage directly**.

Approximate raw vector storage:

```text
float32 bytes = vector_count * dimension * 4
float16 bytes = vector_count * dimension * 2
int8 bytes    = vector_count * dimension * 1
binary bytes  = vector_count * dimension / 8
```

Example for 1 million chunks:

| Dimension | float32 raw vectors | float16 raw vectors |
|---:|---:|---:|
| 384 | ~1.5 GB | ~0.75 GB |
| 768 | ~3.1 GB | ~1.5 GB |
| 1024 | ~4.1 GB | ~2.0 GB |
| 1536 | ~6.1 GB | ~3.1 GB |
| 3072 | ~12.3 GB | ~6.1 GB |

This excludes:

- HNSW graph memory
- payload metadata
- text/chunk storage
- database overhead
- WAL/replication
- indexes
- cache
- snapshots/backups

The lesson:

```text
embedding dimension is an infrastructure decision,
not just a model-card detail.
```

For edge RAG:

```text
384-dimensional embeddings can be a major advantage.
```

For max-quality server retrieval:

```text
larger embeddings may be worth the storage and memory cost.
```

**Measure both.**

---

## 17. How to evaluate embedding models

Do **not choose from vibes**.

Build a **retrieval evaluation set**.

Minimum dataset:

```text
100-300 representative questions
ground-truth relevant chunk ids or document ids
query language labels
query type labels
expected citation requirements
```

Query type labels:

```text
conceptual
exact keyword
API name
error code
code search
cross-lingual
long-document
ambiguous
permission-sensitive
```

Metrics:

| Metric | Meaning |
|---|---|
| recall@k | Did the relevant chunk appear in top-k? |
| MRR | How high was the first relevant result? |
| nDCG | Did the ranking quality match graded relevance? |
| answer faithfulness | Did generation stay grounded in retrieved context? |
| citation accuracy | Did cited chunks actually support the answer? |
| p95 latency | Is retrieval fast enough under load? |
| memory/RAM/VRAM | Does it fit deployment constraints? |
| index build time | Can you refresh the corpus operationally? |

Evaluation loop:

```text
for each embedding model:
  ingest same chunks
  use same metadata
  build index
  run same queries
  measure recall@3, recall@8, MRR, latency
  rerank same candidate count
  run final answer eval
```

Important:

```text
Changing chunking changes the benchmark.
Changing embedding model changes the benchmark.
Changing top-k changes the benchmark.
Changing filters changes the benchmark.
```

Only compare **one major variable at a time**.

---

## 18. Reranking

Embedding retrieval is **first-stage retrieval**.

Reranking is **second-stage retrieval**.

Pattern:

```text
vector search top 30
  -> reranker scores query + candidate text
  -> keep top 3 to 5
  -> send to LLM
```

Why reranking helps:

- dense embeddings are coarse
- chunks can be semantically close but not answer the question
- cross-encoders inspect query and candidate together
- rerankers reduce prompt waste

Common reranker choices:

- BGE reranker family
- Granite reranker
- Jina reranker
- Cohere Rerank
- custom cross-encoder for domain-specific search

When to add reranking:

```text
if recall@20 is good but answer quality is weak,
add reranking before changing the generator.
```

If recall@20 is bad, reranking **will not save you**.

Fix:

- chunking
- embedding model
- hybrid retrieval
- metadata filters
- corpus coverage

---

## 19. Migration rule: never mix embeddings casually

Vectors from different embedding models do **not live in the same comparable space**.

Bad migration:

```text
old chunks embedded with Model A
new chunks embedded with Model B
same vector column
same index
same distance metric
```

This **corrupts retrieval**.

Correct migration:

```text
create new collection or new vector column
backfill all chunks with new model
dual-write new chunks during migration
run retrieval eval against old and new
switch traffic gradually
keep rollback path
delete old index after confidence
```

Qdrant pattern:

```text
collection_docs_v1_granite97
collection_docs_v2_bge_m3
```

pgvector pattern:

```sql
ALTER TABLE document_chunks ADD COLUMN embedding_v2 vector(1024);
CREATE INDEX document_chunks_embedding_v2_hnsw
ON document_chunks USING hnsw (embedding_v2 vector_cosine_ops);
```

Keep model metadata:

```text
embedding_model
embedding_dimension
embedding_normalization
embedding_created_at
chunker_version
source_hash
```

Without this, debugging retrieval regressions becomes **guesswork**.

---

## 20. Security and permissions

Vector databases can **leak data if filtering is wrong**.

Common failure:

```text
query embeds user request
vector search retrieves private chunks
LLM summarizes private chunks to unauthorized user
```

Do **not rely on the LLM to enforce access control**.

Access control belongs **before generation**:

```text
authorized document ids
  -> retrieval filter
  -> rerank only authorized candidates
  -> prompt only authorized context
```

Minimum metadata:

```text
tenant_id
organization_id
project_id
visibility
source
document_id
version
deleted_at
```

For Qdrant:

```text
use payload filters and payload indexes
```

For pgvector:

```text
use SQL WHERE clauses, row-level security if appropriate,
partial indexes, and partitioning where needed
```

**Never send unauthorized chunks** to the model and expect a prompt to save you.

---

## 21. Decision framework

### Choose Qdrant if:

- vector retrieval is central to the product
- you need high-performance filtered vector search
- you want dense + sparse + hybrid retrieval
- you want a dedicated retrieval service
- you need horizontal scaling options
- you are building edge/local RAG as a service
- you expect heavy retrieval traffic
- you want to evolve retrieval independently from the app database

### Choose pgvector if:

- your app already uses PostgreSQL
- vectors are attached to relational entities
- SQL joins and transactions matter
- you want one operational system
- the corpus is modest or moderate
- vector search is not the dominant workload
- your team is stronger in Postgres than vector DB operations

### Choose Granite 97M if:

- local/edge inference matters
- memory is constrained
- multilingual retrieval matters
- Apache-2.0 licensing matters
- compact embeddings are a strategic advantage

### Choose BGE-M3 if:

- hybrid dense/sparse retrieval matters
- you want one open model for multiple retrieval modes
- multilingual and long-ish documents matter
- you can afford more retrieval compute

### Choose API embeddings if:

- you want minimal hosting work
- cloud data flow is acceptable
- quality and speed-to-market matter more than local control
- provider cost is acceptable

---

## 22. Recommended defaults

### Local Jetson-style RAG

```text
embedding:
  Granite 97M Multilingual R2

vector DB:
  Qdrant local service

retrieval:
  dense top 8
  rerank top 3 if latency allows

generator:
  Qwen3.5-4B INT4 or similar 4B-class model

reason:
  compact, private, low memory, good enough to iterate
```

### Existing SaaS app on Postgres

```text
embedding:
  OpenAI / Cohere / Voyage / Granite depending on policy

vector DB:
  pgvector

retrieval:
  SQL WHERE filters
  HNSW index
  optional Postgres full-text search + rank fusion

reason:
  one database and simple app integration
```

### Search-heavy AI product

```text
embedding:
  evaluate Granite, BGE-M3, OpenAI, Cohere, Voyage

vector DB:
  Qdrant

retrieval:
  dense + sparse hybrid
  metadata filters
  reranker
  retrieval telemetry

reason:
  retrieval quality and latency are product features
```

### Codebase assistant

```text
embedding:
  BGE-M3, Granite, Voyage code model, or a code-specialized model

vector DB:
  Qdrant if repo search is a service
  pgvector if it is part of a Postgres-backed app

retrieval:
  hybrid search
  path/language filters
  symbol-aware chunking
  reranking
```

---

## Mini-lab: choose a vector store and embedding model

Design a RAG stack for one of these:

- Jetson local assistant over hardware manuals
- coding assistant over a monorepo
- internal company knowledge base
- multilingual support bot
- PDF-heavy enterprise search tool

Fill this out:

```text
Corpus:
Languages:
Chunk types:
Estimated chunks:
Average chunk tokens:
Strict metadata filters:
Permission model:
Latency target:
Memory target:
Embedding candidates:
Vector DB candidates:
Reranker candidates:
Evaluation query count:
Primary metric:
Secondary metric:
```

Then answer:

```text
I choose Qdrant/pgvector because:
I choose this embedding model because:
I reject the alternatives because:
My first recall@k target is:
My p95 retrieval latency target is:
My migration plan is:
```

If you cannot justify the choice with measurements, you are **still guessing**.

---

## Key takeaways

- Qdrant is a dedicated vector retrieval engine; pgvector is vector search inside PostgreSQL.
- Qdrant is usually better when retrieval is the product path; pgvector is usually better when SQL integration is the product path.
- Dense retrieval captures meaning; sparse retrieval captures exact terms; hybrid retrieval often wins for technical docs.
- HNSW is the common default ANN index; IVFFlat can be useful when memory/build-time tradeoffs matter.
- Filtering is not a detail. Permission and metadata filters are core retrieval correctness.
- Granite 97M is a strong compact local/edge embedding model, but BGE-M3, E5, OpenAI, Cohere, Voyage, Jina, and others can win depending on corpus and constraints.
- Embedding dimension directly affects storage, RAM, index size, and edge viability.
- Do not mix embeddings from different models in one vector space without a controlled migration.
- The only reliable answer to "what is best?" is a retrieval eval on your own data.

---

## References

- Qdrant overview: [https://qdrant.tech/documentation/overview/](https://qdrant.tech/documentation/overview/)
- Qdrant indexing: [https://qdrant.tech/documentation/concepts/indexing/](https://qdrant.tech/documentation/concepts/indexing/)
- Qdrant search docs: [https://qdrant.tech/documentation/search/](https://qdrant.tech/documentation/search/)
- pgvector README: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- IBM Granite Embedding docs: [https://www.ibm.com/granite/docs/models/embedding](https://www.ibm.com/granite/docs/models/embedding)
- Granite 97M Multilingual R2 model card: [https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
- BGE-M3 model card: [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- Multilingual E5 model card: [https://huggingface.co/intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- OpenAI embedding model docs: [https://developers.openai.com/api/docs/models/text-embedding-3-large](https://developers.openai.com/api/docs/models/text-embedding-3-large)
- Cohere embeddings docs: [https://docs.cohere.com/docs/embeddings](https://docs.cohere.com/docs/embeddings)
- Voyage embeddings docs: [https://docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings)
- Lecture 44 - Efficient Local RAG Stack: [Lecture-44.md](Lecture-44.md)

---

*Next: [Lab 01 - Research Agent](Lab-01-Research-Agent.md)*
