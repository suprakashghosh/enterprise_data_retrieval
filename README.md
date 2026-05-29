# Enterprise Data Retrieval (Mugen)

A production-grade document ingestion, structured chunking, and embedding pipeline for
complex, visually rich documents — financial reports, research papers, corporate
filings, and beyond.

---

## Overview

The pipeline ingests a PDF, extracts all structural and visual elements via
**Docling 2.x**, produces typed ``ChunkMetadata`` instances (the single source of
truth), enriches visual elements with LLM-generated descriptions, resolves
cross-references (e.g. "see Figure 3"), computes multimodal embeddings, and
  populates ``relates_to`` via top-k cosine similarity.  Outputs are structured for
  direct loading into Weaviate (vector DB) and Neo4j / Apache AGE (graph DB).

  ---

## Problem Statement

Traditional RAG pipelines treat documents as flat text — they chunk paragraphs,
embed them, and retrieve by semantic similarity. This breaks for enterprise
documents containing:

- **Tables** with structured data
- **Charts and graphs** conveying trends and relationships
- **Images and diagrams** that complement or replace text
- **Cross-references** ("see Figure 3", "as shown in Table IV")
- **Multi-column / complex layouts** where reading order and proximity matter

This project addresses those shortcomings through:

1. **Multi-modal understanding** — text, tables, charts, and images are each
   embedded with the appropriate encoder.
2. **Structured relationships** — explicit `refers_to` (cross-references) and
   `relates_to` (semantic similarity) edges link chunks into a graph.
3. **Deterministic, auditable IDs** — UUIDv5 chunk IDs are stable across re-runs,
   enabling idempotent indexing and graph construction.

---

## Pipeline Architecture

```
PDF (raw bytes)
  │  [ingestion/ingestor.py]
  ▼
Immutable storage + manifest
  │  [extraction/docling_extractor.py]
  ▼
DoclingDocument (pages, layout items, tables, text)
  │  [chunking/docling_chunker.py — HybridChunker]
  ▼
ChunkMetadata[]  ◄── single source of truth (19 fields, frozen Pydantic v2)
  │
  ├─ Embedding dispatch
  │    text  ──► chunk_text encoded
  │    image ──► image_uri loaded & encoded by multimodal model
  │    textual_description ──► LLM-generated description embedded
  │
  ├─ [visual_enricher.py]        ──► +2 chunks per picture/table (image + description)
  ├─ [cross_reference_resolver]  ──► populates refers_to[]
  ├─ [embedding_pipeline.py]     ──► vectors + attach_embeddings → Weaviate-ready dicts
  └─ [similarity.py]             ──► top-3 cosine → populates relates_to[]
```

### ChunkMetadata — The Core Data Model

Every chunk carries 19 fields in a frozen Pydantic v2 model:

| Category | Fields |
|----------|--------|
| **Identity** | `chunk_id`, `document_name`, `document_type`, `document_hash` |
| **Embedding** | `embedding_type` (`text` / `image` / `textual_description`), `chunk_text` |
| **Structure** | `chunk_types`, `section_path`, `section_headings`, `page_numbers`, `sequence_number` |
| **Visual** | `image_type`, `image_uri`, `caption_text`, `caption_number` (parallel lists) |
| **Graph linking** | `element_self_refs`, `refers_to`, `relates_to` |
| **Quality** | `token_count` |

See `src/chunking/models.py` for the full model.

### Export / Output Files

Each pipeline stage can optionally export its output to disk:

| Stage | File pattern | Content |
|-------|-------------|---------|
| Chunking | `{doc_stem}_chunk_metadata.json` | All ChunkMetadata as JSON |
| Embedding | `{doc_stem}_embeddings.json` | chunk_id + embedding + metadata for Weaviate |
| Relates-to | `{doc_stem}_chunks_metadata_with_relates_to.json` | ChunkMetadata with populated `relates_to` |

---

## Getting Started

### Prerequisites

- Python 3.12+
- A virtual environment is recommended.

### Installation

```bash
git clone <repo-url>
cd enterprise_data_retrieval

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

Key dependencies:

| Library | Purpose |
|---------|---------|
| `docling>=2.0` | PDF extraction engine + HybridChunker |
| `pydantic>=2.0` | Data models (ChunkMetadata, schemas) |
| `sentence-transformers[image]>=5.5.1` | Multimodal embeddings (text + image) |
| `numpy>=1.24` | Similarity matrix computation |
| `instructor>=1.15.1` | Structured LLM calls (image descriptions) |
| `openai>=2.38.0` | LLM API access |
| `orjson>=3.0` | Fast JSON serialization |
| `pyarrow>=10.0` | Parquet output (planned) |
| `pdfminer.six[image]` | Page count before Docling extraction |

### Running Tests

```bash
pytest tests/ -q
```

Current: **596 pass**, 7 pre-existing failures in `test_extraction.py::TestRunExtractionPipeline`.

### Running the Chunking Pipeline (Manual)

```python
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.chunking import build_chunk_metadata_list, enrich_visual_chunks, resolve_cross_references
from src.retrieval import build_encode_items, encode_batch, attach_embeddings, populate_relates_to

# 1. Extract with Docling
pipeline_opts = PdfPipelineOptions(
    do_table_structure=True,
    table_structure_mode=TableFormerMode.ACCURATE,
)
conv = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
})
result = conv.convert("document.pdf")

# 2. Chunk + extract metadata
chunks = build_chunk_metadata_list(
    result,
    max_tokens=300,
    output_dir=Path("outputs/doc_abc/"),  # optional export
)

# 3. Enrich visual elements (LLM descriptions)
enriched = enrich_visual_chunks(chunks, result)

# 4. Resolve cross-references ("see Figure 3" → refers_to)
resolved = resolve_cross_references(enriched, result)

# 5. Embed
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("clip-ViT-B-32")
items = build_encode_items(resolved)
embeddings = encode_batch(items, model)

# 6. Attach embeddings + export
docs = attach_embeddings(
    resolved, embeddings,
    document_name="document.pdf",
    output_dir=Path("outputs/doc_abc/"),
)

# 7. Populate relates_to
related = populate_relates_to(
    resolved, embeddings,
    document_name="document.pdf",
    output_dir=Path("outputs/doc_abc/"),
)
```

---

## Key Design Decisions

### Deterministic Chunk IDs
``chunk_id = UUIDv5(namespace, "{doc_hash}|{seq:06d}|{sorted_chunk_types}")``.
Excludes `chunk_text` — IDs are stable across Docling version upgrades.

### Immutable Data Model
``ChunkMetadata`` is frozen. Downstream stages return new instances via `model_copy(update={...})` — never mutate in place.

### Concurrent LLM Calls
Image descriptions via `asyncio.gather` + `Semaphore(max_concurrent=5)`.
Failures don't halt the pipeline — individual errors are logged and the chunk is created with an empty description.

### Caption Normalization
Both `extract_caption_label()` and `find_all_caption_refs()` normalize type names through the same `VALID_IMAGE_TYPES` map (e.g. "Fig. 3" → `"figure 3"`). This guarantees cross-reference matches are consistent.

### Sibling-Aware Relates-to
Chunks sharing any `element_self_ref` (e.g. an image chunk and its textual-description variant) are excluded from each other's `relates_to` — a chunk should not "relate to" itself through another embedding modality.

### Pure Numpy Similarity
No HDBSCAN, no sklearn. Cosine similarity via `einsum`, top-k via `argpartition` (O(n) partial sort). `min_similarity=0.75` threshold keeps edges sparse and meaningful.

---

## Roadmap

### Short-Term

- [ ] Weaviate vector DB integration — collection schema, batch ingestion with all filterable properties
- [ ] Graph DB construction — Neo4j/AGE nodes (document, chunk, section) + edges (follows, belongs_to, refers_to, relates_to)
- [ ] Pipeline orchestrator — single CLI entry point, JSONL/Parquet export, coverage validation

### Medium-Term

- [ ] Hybrid search (vector + BM25 + graph traversal)
- [ ] Graph-based context expansion at query time
- [ ] Query decomposition and expansion
- [ ] Agentic reasoning with structured knowledge

### Long-Term

- [ ] REST API (document upload, indexing, query)
- [ ] Incremental / streaming document updates
- [ ] Dashboard for graph visualization and document exploration
- [ ] Support for DOCX, HTML, scanned-image PDFs with OCR

---

## Status

> **Active Development — Production Pipeline In Progress**

The chunking, cross-reference resolution, multimodal embedding, and similarity
pipeline is fully functional for single-document processing. Weaviate vector DB
integration, graph DB construction, and the pipeline orchestrator are in progress.

The original notebook-based prototypes in `src/notebooks/` are retained for
reference but the production pipeline in `src/chunking/` and `src/retrieval/` is
the canonical implementation.

---

## License

MIT
