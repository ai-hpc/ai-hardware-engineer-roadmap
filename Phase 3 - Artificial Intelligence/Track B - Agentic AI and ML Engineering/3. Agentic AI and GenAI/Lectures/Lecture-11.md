# Lecture 11 — RAG ingestion, chunking and embeddings

**Course:** [AI Agent Development](../Guide.md) | [Previous](Lecture-10.md) | [Next](Lecture-12.md)

**Reviewed 18 October 2026.** Turn a source into retrievable evidence without losing its identity, meaning or permissions. Start with the executed [local retrieval lab](../Local-RAG-Workflow-Lab/Guide.md); its SQLite baseline needs no model. The text-splitter behavior below was executed with `langchain-text-splitters==1.1.3`. Embedding/cloud integrations are optional and are not claimed as local inference results.

**Time:** 60 minutes plus the lab. **Outcome:** an ingestion manifest, a defensible chunking choice and a small labelled retrieval experiment.

## 1. Ingestion is a data contract

```text
original source → extraction → normalization → chunks → index
        identity / revision / permissions / provenance travel with every stage
```

For each source retain its stable ID, acquisition URI/path, content hash, revision/effective date, permission scope, extraction method and relevant page/section locations. A chunk also needs a unique chunk ID and a mapping back to its source. The same paragraph in two documents is not necessarily the same evidence: ownership, revision and permissions can differ.

A PDF loader extracts what it can from a document; it does not certify table order or OCR accuracy. A scanned image needs OCR and a quality check. HTML extraction can drop code, tables, negative signs or footnotes. Markdown headings help preserve context but do not turn every paragraph into a self-contained fact. Inspect a representative sample before embedding thousands of chunks.

**Worked failure:** a table says “maximum temperature” in its header and “40” in one row. Indexing only the row “40” loses the unit and condition. Preserve a meaningful table/header representation and its source location. A larger model cannot reliably reconstruct missing evidence.

Treat imported content as data. Access labels come from trusted application metadata, not a line in the document saying “this is public”. Validate file size/type and network scope before extraction. Record failed documents rather than silently replacing them with fabricated text.

## 2. Choose boundaries before choosing a vector database

Small chunks can match a narrow phrase but omit its conditions. Large chunks preserve context but dilute the query match and consume more context budget. Overlap can preserve a boundary fact while producing redundant retrieved evidence. There is no universal best chunk size.

| Strategy | Useful starting case | Failure to test |
|---|---|---|
| Document/section boundaries | Short manuals with meaningful headings | Oversized sections |
| Recursive character splitting | General text with a hard character fallback | Split tables, identifiers and non-space languages |
| Token-aware splitting | Model token-budget constraints | Tokenizer mismatch and lost structure |
| Semantic boundary detection | Topical transitions | Extra embedding cost and unstable/oversized groups |

Current imports use `langchain_text_splitters`, not `langchain.text_splitter`. This executed example reveals a subtle distinction:

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = 'x' * 31
simple = CharacterTextSplitter(chunk_size=10, chunk_overlap=0, separator='\n')
recursive = RecursiveCharacterTextSplitter(
    chunk_size=10, chunk_overlap=0, separators=['\n', ' ', '']
)
print([len(x) for x in simple.split_text(text)])     # [31]
print([len(x) for x in recursive.split_text(text)])  # [10, 10, 10, 1]
```

`CharacterTextSplitter` merges separator-delimited pieces; an indivisible piece can exceed the target. The recursive splitter's final empty-string fallback permits splitting that piece. A character limit is still not a token limit, and neither proves semantic completeness. [Official recursive splitter contract](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

Semantic chunkers commonly use distances between neighboring sentence groups. A high percentile of **distance** means a large semantic change, not the same numerical percentile of similarity. Inspect the selected implementation, enforce a model input limit, and compare retrieval against a simpler boundary strategy.

## 3. Understand what an embedding score means

An embedding maps an input to a numeric vector. Its geometry is learned from a training objective; closeness can be useful for retrieval but does not prove truth, entailment, access permission or answer equivalence.

For nonzero vectors, cosine similarity is:

`cos(a,b) = dot(a,b) / (norm(a) * norm(b))`.

For `a=(3,4)` and `b=(0,5)`, the dot product is 20, both norms are 5, and cosine is 0.8. After unit normalization, the dot product is also 0.8. Without normalization, a larger magnitude can change dot-product rank. A zero vector needs an explicit policy; division by zero is not meaningful similarity.

A backend may return cosine similarity, cosine distance, L2 distance or a transformed relevance score. Keep the configured metric with the index and verify its sort direction. SQLite FTS5 BM25 scores use a different convention from a normalized embedding dot product. [Sentence Transformers similarity definitions](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).

## 4. Select and record an embedding contract

Record the exact model/revision, tokenizer, input template or query/document prefix, pooling, maximum input length, normalization, output dimensions and dtype. The query and document paths must use the compatible contract. Changing one requires a deliberate reindex or a verified compatibility strategy.

The familiar `sentence-transformers/all-MiniLM-L6-v2` is a 384-dimensional reference encoder. Its card documents truncation of long input; **roughly 22 million parameters is not a 22 MB full-precision weight file**. Weight storage alone scales with parameters times bytes per value, before runtime overhead. Check actual artifact files and the [official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), rather than copying an invented CPU documents-per-second table.

For a local Sentence Transformers extension, pin an immutable repository revision and package environment, encode both document and query inputs according to the model card, and measure on the target CPU. Current LangChain integration uses `from langchain_huggingface import HuggingFaceEmbeddings`; it is a separate package. [Current integration](https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub).

A hosted embedding extension additionally records the selected accessible model, returned dimensionality, request limits, measured usage and current price source. An arbitrary placeholder model name cannot be assigned a fixed price, dimensionality or speed. A 100 MB file is not reliably 25 million tokens: encoding, language and extraction affect token count. Prefer measured token usage and batch latency.

## 5. Match storage to the application

| Option | What it provides | What you still must define |
|---|---|---|
| SQLite FTS5 | Local lexical index and BM25 ranking | Tokenization, scope, versioning and relevance labels |
| FAISS | Vector similarity search/index structures | Metadata, persistence trust, permissions and service behavior |
| Chroma or another vector database | Vector storage/query plus metadata features | Model contract, score semantics and lifecycle |
| Managed service such as Pinecone | Hosted index/service operations | Trusted tenant mapping, cost, consistency and access policy |

A namespace helps partition data; it is not authorization if the caller can select any namespace. Derive the tenant/scope from the authenticated application context and enforce it in every retrieval route. Recheck source validity before returning cached or persisted evidence.

Some FAISS wrappers persist Python metadata with pickle. Loading it with `allow_dangerous_deserialization=True` is an explicit trust decision about the artifact, not a standard fix for any downloaded index. Rebuild untrusted indexes from validated source data. A plain vector file also needs shape, type, integrity and model-contract validation.

## 6. Build ingestion that can be updated

Use stable IDs and an explicit replace/upsert/delete policy. Re-running ingestion must not silently duplicate every chunk. If document revision B supersedesA, test that old chunks leave the active index. Keep a revisioned manifest so an answer can be traced to the snapshot used at retrieval time.

The local lab builds a physically scoped current index from a fixed corpus and records its hash. It intentionally leaves a known relevance/abstention failure visible. A successful ingestion job is a transport result; retrieval quality requires labelled queries.

For larger corpora, measure extraction failures, chunk count/length distribution, duplicates, embedding failures, index size, build time and query latency. Split evaluation by source family/version where appropriate so near-duplicate chunks do not leak between development and evaluation.

## 7. Exercises and diagnostic answers

1. Why does the simple splitter return31 characters despite a 10-character target? **Answer:** its separator never appears; the unsplit piece is not recursively divided.
2. Why retain the source revision in a citation? **Answer:** identical filenames can contain changed facts; a revision/hash binds the evidence.
3. A semantic score is 0.98 for “stop at 0.5 m” and “stop at 5 m”. Can the answer be reused? **Answer:** no; a small numeric change can change the requested action despite similar wording.
4. Does more overlap always help? **Answer:** it may preserve boundaries but increase redundant candidates and storage; measure answer-source recall and diversity.
5. Does a valid quote prove correctness? **Answer:** it proves a source span exists. Relevance, completeness and source truth are separate checks.

**Submit:** the local lab report plus one changed chunking strategy evaluated on development questions. Preserve original sources and labels; do not manufacture a perfect benchmark from the documents being indexed.
