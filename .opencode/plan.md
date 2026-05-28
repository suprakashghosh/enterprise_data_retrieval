# Plan: Extraction and Structured Chunking Stage

## Objective

Build a production-grade extraction and structured chunking pipeline that ingests complex documents (PDFs with text, tables, charts, images, formulas, captions, footnotes, and rich layouts), extracts all elements via Docling, normalizes them into a well-defined internal schema, reconstructs document hierarchy, generates typed chunks with relationship metadata, and exports final artifacts (JSONL, Parquet, reports). This plan covers the full ingestion-to-chunks pipeline and **stops before** graph database loading, vector indexing, hybrid search, query decomposition, context expansion, and answer generation.

## Requirements Snapshot

- **R1 (Docling Extraction):** Use Docling as the primary extraction engine. Convert PDFs into Docling document format with full layout, table structure, OCR, and page-level detail.
- **R2 (Internal Schema):** Define and implement target schemas/data models for Document and Page (minimal — ingestion only) and ChunkMetadata (the retrieval-layer model). Use Docling's native models for extraction; no custom element registry.
- **R3 (ID Strategy):** Define deterministic UUIDv5 IDs for documents and chunks enabling cross-referencing and idempotent reprocessing.
- **R4 (Project Structure):** Establish modular project structure with separate packages for ingestion, extraction, chunking, schemas, validation, retrieval, and utilities.
- **R5 (Raw Document Storage):** Implement ingestion that stores raw PDFs immutably and associates them with a document record.
- **R6 (Docling Output Persistence):** Persist raw Docling outputs (JSON, markdown, page images, table images) alongside the document record for audit and re-processing.
- **R7 (Output Validation):** Validate Docling outputs for completeness, correct page counts, non-empty content, and structural consistency.
- **R8 (ChunkMetadata Model):** Define ``ChunkMetadata`` — the single source of truth for every chunk. Must carry identity (ID, document hash), structural fields (page numbers, section path, chunk types), visual-element fields (image_uri, caption_text, caption_number), embedding-type dispatch (text / image / textual_description), graph-linking fields (element_self_refs, refers_to, relates_to), and quality signals (token_count). The model is frozen and Pydantic v2.
- **R9 (Docling HybridChunker Wrapper):** Wrap Docling's built-in ``HybridChunker``. Extract all metadata fields directly from ``DocChunk.meta`` — page numbers from provenance, section headings, element labels, captions, and element self_refs. No intermediate normalization step.
- **R10 (Visual Element Enrichment):** For every picture and table in the document, generate two additional chunks: one with ``embedding_type="image"`` (the image itself is embedded; caption stored for reference), and one with ``embedding_type="textual_description"`` (an LLM-generated textual description is embedded). LLM calls must be concurrent via ``asyncio.gather`` + semaphore.
- **R11 (Cross-Reference Resolution):** Scan chunk text for patterns like "see Figure 3", "Table IV", "Eq. 5". Resolve to target chunk IDs via ``caption_number`` matching and populate ``refers_to`` fields. Creates the foundation for graph-DB ``refers_to`` edges.
- **R12 (Multimodal Batch Embedding):** Build ``items_to_encode`` from chunk metadata (dispatched by ``embedding_type``). Embed text chunks, image URIs, and textual descriptions using a multimodal embedding model (e.g., ``Qwen/Qwen3-VL-Embedding-2B`` or similar). Store embeddings alongside metadata.
- **R12b (Relates-to via Top-k Similarity):** For each chunk, compute top-k most similar chunks by cosine similarity on embeddings and populate ``relates_to``. Sparse edges only (top_k=3), with a minimum similarity threshold (0.75) to filter noise. Sibling chunks (same element_self_refs) are excluded — a visual chunk should not "relate to" its own textual-description variant. Pure numpy, no external clustering dependencies. Chunks below threshold get empty ``relates_to``.
- **R13 (Weaviate Vector DB Integration):** Load chunks and their embeddings into Weaviate. All filterable fields (page_numbers, section_path, caption_number, token_count, chunk_types) must be stored as Weaviate properties for hybrid (vector + filter) retrieval.
- **R14 (Graph DB Construction):** Build a graph from chunk metadata: document nodes, chunk nodes, section nodes. Create edges for sequential order (``follows``), section membership (``belongs_to``), cross-references (``refers_to``), and semantic relatedness (``relates_to``). Load into Neo4j or Apache AGE.
- **R15 (Pipeline Orchestrator, Export & Validation):** Chain all stages into a single callable entry point. Export chunk metadata as JSON/JSONL. Add validation checks for chunk coverage, ID uniqueness, and embedding completeness.
- **R16 (Out of Scope — Retrieval / RAG):** Hybrid retrieval, query decomposition, graph-based context expansion, answer generation, and agentic reasoning are explicitly out of scope for this plan.

## Scope

- Building the ingestion-to-retrieval pipeline: ingestion, Docling conversion, output persistence, validation, HybridChunker wrapper with metadata extraction, visual enrichment (LLM image descriptions), cross-reference resolution, batch multimodal embedding, Weaviate vector DB loading, and graph DB construction.
- Defining the ``ChunkMetadata`` model (Pydantic v2) in ``src/chunking/models.py`` — the single source of truth for every chunk.  Legacy schemas in ``src/schemas/`` are retained but not extended.
- Modularizing into ``ingestion/``, ``extraction/``, ``chunking/``, ``retrieval/``, ``validation/``, ``schemas/``, and ``utils/`` packages under ``src/``.
- Handling PDF documents with Docling; no other input formats in this phase.
- Supporting all document element types: text, tables, images — captured naturally by Docling chunks.
- No separate formula or code handling; these are captured within Docling's chunk boundaries.

## Assumptions and Constraints

- **Python 3.10+** is the target runtime; all implementation must be compatible.
- **Docling** (the `docling` PyPI package) is the extraction and chunking engine.  ``HybridChunker`` is used directly — no custom chunking strategies.  Docling version 2.x or later required.
- **GPU is not required** but acceleration (CUDA) should be leveraged when available for layout analysis and embedding-based chunking.
- The project currently has no test infrastructure; tests must be introduced alongside the implementation (pytest).
- All modules should be importable without a running GPU or external service; graceful fallbacks where sensible.
- Output directories (`data/`, `outputs/`) are gitignored already.
- The existing notebook code is prototype quality and should serve as reference, not as code to copy directly.
- Docling's built-in `HybridChunker` can be used as a foundation but must be wrapped and extended for the project's custom chunking needs.
- No data is sensitive/confidential in the sample documents; security controls (PII redaction, access control) are out of scope for this stage.

## Risks and Areas Requiring Care

- **Docling API volatility:** Docling is under active development. Pin the `docling` version in `pyproject.toml` and isolate Docling interactions behind thin adapter/wrapper functions.
- **Large documents:** Documents can be 300+ pages with hundreds of figures. The chunking pipeline must not load all page images or chunk JSON into memory simultaneously.
- **Multi-page table spanning:** Docling's ``HybridChunker`` may split or merge spanning tables unpredictably.  The metadata extraction should record which pages a chunk touches (``page_numbers``) but not attempt to re-merge.
- **Broken PDFs / OCR failures:** PDFs may be scanned images with corrupted streams. Docling may produce incomplete output. Validation (R7) must catch these cases.
- **ID determinism:** Chunk IDs must be identical across re-runs for the same document + chunker parameters.  Use UUIDv5 with ``document_hash`` + ``sequence_number`` + ``chunk_types`` — do NOT include ``chunk_text`` in the ID.
- **LLM API failures:** The instructor API for image descriptions may be rate-limited or unavailable.  ``asyncio.gather(return_exceptions=True)`` ensures one failure doesn't halt all descriptions.
- **False-positive cross-references:** Pattern matching for "see Figure 3" may match non-reference mentions.  Require both a type keyword AND a number after extraction; skip when no target chunk is found.

## Core Concepts

### Internal Data Model Overview

The pipeline transforms raw PDFs through several stages. The core data model evolves as follows:

```
PDF (raw bytes)
  │  [Docling conversion]
  ▼
DoclingDocument (Docling's native representation)
  │  [HybridChunker]
  ▼
DocChunk[] (Docling chunk objects with meta.headings, meta.doc_items, etc.)
  │  [Metadata extraction]
  ▼
ChunkMetadata[] (our Pydantic model — single source of truth)
  │
  ├── Text chunks:          embedding_type = "text"
  ├── Image chunks:         embedding_type = "image"          (picture/table visual)
  └── Description chunks:   embedding_type = "textual_description"  (LLM output)
  │
  ├── [Cross-reference resolver]  → populates refers_to[]
  ├── [Batch embedding]          → vectors stored in Weaviate
  └── [Graph construction]       → nodes + edges in Neo4j / AGE
```

### ID Strategy

Each document and chunk gets a deterministic UUIDv5 generated from stable inputs:

| Entity | UUID Namespace | Input to Hash |
|--------|---------------|----------------|
| Document | `docling-project-doc` | SHA256 of raw file content |
| Chunk | `docling-project-chunk` | `{document_hash}\|{sequence_number:06d}\|{sorted_chunk_types}` |

See ``src/chunking/models.py::make_chunk_id()`` for the exact implementation.

### Chunk Embedding Types

| embedding_type | What is embedded | chunk_text stores |
|---|---|---|
| ``"text"`` | ``chunk_text`` (raw document content) | The text |
| ``"image"`` | ``image_uri`` (image bytes via multimodal encoder) | Caption — for reference / display only |
| ``"textual_description"`` | ``chunk_text`` (LLM-generated description) | The description |

### Relationship Types (for Graph DB)

| Relationship | Direction | Meaning |
|-------------|-----------|---------|
| ``follows`` | A → B | A appears immediately before B in reading order |
| ``belongs_to`` | chunk → section | The chunk belongs to a given section path |
| ``refers_to`` | A → B | Chunk A references chunk B (e.g., "see Figure 3") |
| ``relates_to`` | A ↔ B | General semantic / structural relatedness |
| ``same_section_as`` | A ↔ B | Chunks share the same section path |

## Sub-Tasks

---

### Sub-Task 0: Create .opencode directory and plan.md

- **Status:** Completed
- **Objective:** Ensure the `.opencode/` directory exists and this `plan.md` is the single source of truth for the implementation.
- **Related Requirements:** (meta — enabling all others)
- **Dependencies and Preconditions:** None.
- **Instructions:** Directory already created; plan written.
- **Done When:** This file exists at `.opencode/plan.md`.

---

### Sub-Task 1: Define Target Schemas / Data Models (R2, R3)

- **Status:** Completed
- **Objective:** Define all Pydantic data models in a `src/schemas/` package. These models are the internal representation that every subsequent step produces and consumes.
- **Related Requirements:** R2 (Internal Schema), R3 (ID Strategy)
- **Dependencies and Preconditions:** Sub-Task 0 (plan exists). No code dependencies.
- **In Scope for This Sub-Task:**
  - Create `src/schemas/__init__.py`
  - Create `src/schemas/document.py` — `DocumentSchema` with fields: `doc_id: UUID`, `title: str`, `source_path: str`, `file_hash: str`, `page_count: int`, `created_at: datetime`, `pages: List[PageSchema]`, `elements: Dict[str, ElementSchema]`, `chunks: List[ChunkSchema]`, `relationships: List[RelationshipSchema]`, `metadata: DocumentMetadata`.
  - Create `src/schemas/elements.py` — Base `ElementSchema` with: `element_id: UUID`, `doc_id: UUID`, `page_num: int`, `bbox: BoundingBox`, `reading_order: int`, `section_path: str`, `element_type: Literal`, `content: str`, `metadata: ElementMetadata`. Subclasses: `TextBlockSchema`, `TableSchema`, `ImageSchema`, `ChartSchema`, `GraphSchema`, `FormulaSchema`, `CaptionSchema`, `FootnoteSchema`, `HeaderSchema`, `FooterSchema`, `ListBlockSchema`, `SectionHeaderSchema`. Each subclass adds typed fields (e.g., `TableSchema` has `markdown: str`, `html: str`, `json: dict`, `row_count: int`, `col_count: int`, `headers: List[str]`).
  - Create `src/schemas/relationships.py` — `RelationshipSchema` with: `relationship_id: UUID`, `source_id: UUID`, `target_id: UUID`, `relationship_type: Literal` (all types from Core Concepts), `metadata: dict`, `weight: float`.
  - Create `src/schemas/chunks.py` — `ChunkSchema` with: `chunk_id: UUID`, `doc_id: UUID`, `chunk_type: Literal["hierarchical", "semantic", "cluster"]`, `content: str`, `element_refs: List[UUID]`, `section_path: str`, `page_range: Tuple[int, int]`, `relationships: List[RelationshipSchema]`, `metadata: ChunkMetadata`.
  - Create `src/schemas/metadata.py` — `DocumentMetadata`, `ElementMetadata`, `ChunkMetadata` with optional fields for source info, confidence scores, processing timestamps, extraction versions, and extensibility dicts.
  - Create `src/schemas/geometry.py` — `BoundingBox(left, top, right, bottom, coord_system: Literal["pdf", "image", "normalized"])`, `Size(width, height)`, `Point(x, y)`.
  - Implement UUID generation helpers in `src/schemas/id_gen.py` using `uuid.uuid5()` with the namespace constants and input strings described in Core Concepts.
- **Out of Scope for This Sub-Task:**
  - No processing logic, converters, or validators that operate on data beyond schema definitions.
  - No database/ORM models; these are pure Pydantic schemas.
  - No chunking or extraction logic.
- **Instructions:**
  1. Create the directory tree: `src/schemas/` with `__init__.py` that exports all public model classes.
  2. Define each model as a frozen (or at least immutable-where-sensible) Pydantic `BaseModel`.
  3. Use `Literal` types constrained to the approved relationship types and element types.
  4. Write UUID helpers that accept `(namespace, name_string)` and produce deterministic UUIDv5.
  5. Add docstrings to every model and field.
  6. Test the models import correctly.
- **Acceptance Criteria:**
  - `from src.schemas import DocumentSchema, ElementSchema, TableSchema, ChunkSchema, RelationshipSchema` works.
  - A `DocumentSchema` can be instantiated with all fields populated.
  - UUIDs generated with the same inputs are identical; different inputs produce different UUIDs.
  - All relationship types from the table in Core Concepts are valid values.
  - `pip install -e .` (once `pyproject.toml` exists) or direct import works.
- **Cautionary Points (Risks & Edge Cases):**
  - Avoid circular imports between schema modules (element referencing document, document containing elements). Use `ForwardRef` or late annotations.
  - Keep models simple; avoid inheritance trees deeper than 2 levels.
  - The geometry module should support all three coordinate systems (PDF points, image pixels, normalized 0-1) to avoid ambiguity.
- **Implementation Suggestions:**
  - Use Pydantic v2 (`BaseModel`, `field_validator`, `model_config`).
  - Use `uuid.NAMESPACE_DNS` or custom UUID constants for deterministic generation.
  - Use `typing.Literal` with `typing_extensions` if Python < 3.11 (but target 3.10+ so `Literal` from `typing` works).
- **Testing Suggestions:**
  - Create `tests/test_schemas.py` — instantiate every model, verify field types, test UUID determinism (same input → same UUID, different input → different UUID).
  - Model validators: test that invalid relationship types raise `ValueError`.
- **Done When:**
  - The entire `src/schemas/` package is implemented, imports cleanly, and `pytest tests/test_schemas.py` passes.

---

### Sub-Task 2: Establish Project Structure and Packaging (R4)

- **Status:** Completed
- **Objective:** Create the modular project skeleton with all top-level packages and a basic `pyproject.toml` that makes the project installable.
- **Related Requirements:** R4 (Project Structure)
- **Dependencies and Preconditions:** Sub-Task 1 (schemas must exist as the first package). Sub-Task 0 (plan exists).
- **In Scope for This Sub-Task:**
  - Create `pyproject.toml` with project name `enterprise_data_retrieval`, Python 3.10+, dependencies from `requirements.txt` plus `docling`, `pydantic`, `pytest`, `pyarrow` (for Parquet), `orjson` or `ujson` for fast JSON.
  - Create directory structure:
    ```
    src/
    ├── __init__.py
    ├── schemas/          (from Sub-Task 1)
    ├── ingestion/        (Sub-Task 3)
    ├── extraction/       (Sub-Tasks 4-5)
    ├── normalization/    (Sub-Task 7-9)
    ├── metadata/         (Sub-Tasks 11-13)
    ├── chunking/         (Sub-Task 14-15)
    ├── validation/       (Sub-Task 17)
    └── utils/            (shared helpers, logging, file I/O)
    ```
  - Create `__init__.py` in each package with appropriate exports.
  - Create `src/utils/logging.py` with a configured `structlog` or standard `logging` setup.
  - Create `src/utils/file_io.py` with helpers for JSONL writing, Parquet writing, asset path management, and atomic writes.
  - Create `src/utils/config.py` with a `Settings` class (pydantic-settings or dataclass) for pipeline configuration (model paths, chunk sizes, output dirs, etc.).
  - Set up `pytest` configuration in `pyproject.toml` and create `tests/` at project root with `__init__.py`.
- **Out of Scope for This Sub-Task:**
  - No business logic beyond the skeleton.
  - No CI/CD configuration.
- **Instructions:**
  1. Initialize `pyproject.toml` using standard PEP 621 format with `[project]`, `[build-system]`, and `[tool.pytest.ini_options]`.
  2. Create all directories and `__init__.py` files with module-level docstrings.
  3. In `src/utils/config.py`, define a `PipelineSettings` dataclass with sensible defaults for chunk token limit (512), overlap (64), embedding model ("BAAI/bge-m3"), and output directories.
  4. Verify the project is installable with `pip install -e .`
  5. Verify `pytest --collect-only` discovers tests (even if none exist yet).
- **Acceptance Criteria:**
  - `pip install -e .` succeeds.
  - `python -c "from src.schemas import DocumentSchema; print('OK')"` works.
  - `pytest --collect-only` shows the test directory.
- **Cautionary Points (Risks & Edge Cases):**
  - Avoid dependency version conflicts; docling may require specific versions of torch, transformers, etc. Pin aggressively in `pyproject.toml`.
  - Keep `pyproject.toml` minimal; do not add build-time deps beyond `setuptools`.
- **Implementation Suggestions:**
  - Use `setuptools` as the build backend (simplest).
  - For `src/utils/file_io.py`, use `pathlib.Path` everywhere, never raw strings.
- **Testing Suggestions:**
  - Test that `PipelineSettings` loads from environment variables (if using pydantic-settings) or has correct defaults.
  - Test `atomic_write` helper in `file_io.py` (write text, JSON, binary) and verify crash safety.
- **Done When:**
  - The project skeleton imports cleanly, `pip install -e .` succeeds, and `pytest --collect-only` passes.

---

### Sub-Task 3: Implement Document Ingestion and Raw Document Storage (R5)

- **Status:** Completed
- **Objective:** Create the ingestion module that accepts PDF files (or a directory of PDFs), creates a document record, stores the raw PDF immutably, and prepares for extraction.
- **Related Requirements:** R5 (Raw Document Storage)
- **Dependencies and Preconditions:** Sub-Task 2 (project structure, file I/O utils, config). Sub-Task 1 (schemas for document records).
- **In Scope for This Sub-Task:**
  - Create `src/ingestion/__init__.py` and `src/ingestion/ingestor.py`.
  - Define `IngestionSource` (single file path, directory path, list of paths, or URL).
  - Implement `ingest_pdf(source) -> DocumentSchema` that:
    1. Copies the PDF into a `data/raw/{doc_id}/` directory (immutable storage).
    2. Computes SHA-256 file hash.
    3. Uses PyPDF2 or pdfminer to get page count (quick, before Docling).
    4. Returns a minimal `DocumentSchema` with basic metadata (title from filename or PDF metadata, source path, file hash, page count, created_at).
  - Implement `IngestionManifest` — a JSON file per document at `data/raw/{doc_id}/manifest.json` recording original source, timestamp, file hash, and processing status.
  - Implement `batch_ingest(sources) -> List[DocumentSchema]` for processing multiple documents.
- **Out of Scope for This Sub-Task:**
  - No Docling conversion yet (that is Sub-Task 4).
  - No content extraction, analysis, or chunking.
  - No support for non-PDF formats.
- **Instructions:**
  1. In `ingestor.py`, write `ingest_pdf()` that uses `pathlib.Path` for all file operations.
  2. Use `shutil.copy2` to copy the source PDF to `data/raw/{doc_id}/source.pdf` (the immutable copy).
  3. Use `hashlib.sha256()` to compute the file hash.
  4. Write the ingestion manifest as JSON with `orjson` or standard `json`.
  5. Use the `PipelineSettings` config for the base data directory.
  6. Add a `get_document(doc_id) -> DocumentSchema` to load a previously ingested document.
- **Acceptance Criteria:**
  - Calling `ingest_pdf("path/to/test.pdf")` creates `data/raw/<uuid>/source.pdf` and `data/raw/<uuid>/manifest.json`.
  - The returned `DocumentSchema` has a valid UUID, the correct page count, and the correct file hash.
  - Re-ingesting the same file produces the same `doc_id` (due to content-hash-based UUID).
  - `batch_ingest` returns a list of `DocumentSchema` for multiple files.
  - Error handling: non-existent files raise `FileNotFoundError`; empty PDFs are handled gracefully.
- **Cautionary Points (Risks & Edge Cases):**
  - PDFs may be very large (hundreds of MB). Copying should stream in chunks, not read the entire file into memory. `shutil.copy2` handles this.
  - File hash should be computed during copy (streaming hash) to avoid reading the file twice. Consider a utility that copies and hashes simultaneously.
  - Windows vs. Linux path separator differences are handled by `pathlib.Path`.
- **Implementation Suggestions:**
  - Create a helper in `src/utils/file_io.py`: `def copy_with_hash(src: Path, dst: Path) -> str` that copies and returns the SHA-256 hex digest.
  - Use `uuid.uuid5(schema.DOC_NAMESPACE, file_hash)` for deterministic doc ID.
- **Testing Suggestions:**
  - Create `tests/test_ingestion.py` with a fixture that generates a minimal valid PDF (use `reportlab` or just copy a known test PDF).
  - Test: ingest, verify file exists, verify hash matches, verify doc_id is deterministic.
  - Test: ingest same file twice — IDs match, no duplicate copy.
  - Test: batch ingest of mixed valid/invalid files, verify error handling.
- **Done When:**
  - `pytest tests/test_ingestion.py` passes all tests, and manual ingestion of the included `2502.04644v1.pdf` works producing expected output.

---

### Sub-Task 4: Implement Docling Extraction and Persist Raw Outputs (R1, R6)

- **Status:** Completed
- **Objective:** Use Docling to convert ingested PDFs into Docling's native document format and persist all raw outputs (JSON, markdown, page images, table images, asset files) for audit and re-processing.
- **Related Requirements:** R1 (Docling Extraction), R6 (Docling Output Persistence)
- **Dependencies and Preconditions:** Sub-Task 3 (ingestion and raw storage). Sub-Task 2 (utils, config). Docling must be installed.
- **In Scope for This Sub-Task:**
  - Create `src/extraction/__init__.py` and `src/extraction/docling_extractor.py`.
  - Implement `extract_with_docling(doc: DocumentSchema) -> DoclingDocument` that:
    1. Loads the raw PDF from `data/raw/{doc_id}/source.pdf`.
    2. Configures `DocumentConverter` with `PdfPipelineOptions` (table structure enabled, `TableFormerMode.ACCURATE`, cell matching disabled — matching the pattern from notebook 1).
    3. Calls `converter.convert()`.
    4. Returns the Docling `DoclingDocument` object.
  - Implement `persist_docling_outputs(doc: DocumentSchema, dl_doc: DoclingDocument)` that:
    1. Saves `dl_doc.export_to_dict()` as JSON → `data/raw/{doc_id}/docling/output.json`.
    2. Saves `dl_doc.export_to_markdown()` → `data/raw/{doc_id}/docling/output.md`.
    3. Saves page-level rendered images (via Docling's page image export or PIL rendering) → `data/raw/{doc_id}/docling/pages/page_{n}.png`.
    4. Saves table-level images extracted by Docling → `data/raw/{doc_id}/docling/tables/table_{idx}.png`.
    5. Saves any other asset files (if Docling extracts embedded images) → `data/raw/{doc_id}/docling/assets/`.
  - Implement `run_extraction_pipeline(doc: DocumentSchema) -> DocumentSchema` that orchestrates extraction and persistence, returning the document with updated metadata (extraction timestamp, docling version).
  - Add a checkpoint/status field in the manifest: `"extraction_status": "completed"`.
- **Out of Scope for This Sub-Task:**
  - No normalization into internal schemas (that is Sub-Task 7).
  - No validation (that is Sub-Task 6 — but extraction must not crash on bad docs).
  - No chunking.
- **Instructions:**
  1. Create an abstraction layer: `class DoclingAdapter` that wraps `DocumentConverter` creation, making it possible to mock or swap Docling in tests.
  2. The adapter must handle Docling import errors gracefully (if docling is not installed, raise a clear `ImportError` with installation instructions).
  3. Follow Docling's documentation for best practices on pipeline options. The reference is notebook 1's code.
  4. For page images, if Docling's page image API is not available, use `pdf2image` or PyMuPDF as a fallback.
  5. Log progress (page number, table count, etc.) via the structured logger from `utils/logging.py`.
- **Acceptance Criteria:**
  - `extract_with_docling(doc)` returns a valid Docling document with pages, text content, tables, and layout items.
  - `persist_docling_outputs` writes all expected files (JSON, markdown, page images, table images) to the correct paths.
  - Re-extracting the same document overwrites the persisted outputs (or skips if content-hash did not change).
  - A document with no tables still produces correct output (no errors, empty tables list).
- **Cautionary Points (Risks & Edge Cases):**
  - Docling may throw errors on corrupted PDFs. The adapter should wrap extraction in a try/except and set status to `"failed"` with error details in the manifest.
  - Page image rendering is memory-intensive for large documents. Render images lazily or page-by-page; do not hold all page images in memory at once.
  - Docling's `export_to_dict()` output can be very large (tens of MB). Use `orjson` for fast serialization.
- **Implementation Suggestions:**
  - Use a `@lru_cache` or similar to avoid re-extracting if outputs already exist and the source hasn't changed.
  - Store the docling version in the output directory for debugging: `data/raw/{doc_id}/docling/version.txt`.
- **Testing Suggestions:**
  - Create a minimal PDF with known content (a title, a paragraph, a simple table, a simple image) using `reportlab`. Run extraction and verify outputs exist.
  - Create a PDF with no extractable content (scanned image without OCR) and verify graceful failure.
  - Run extraction on the sample `2502.04644v1.pdf` and measure time/memory.
- **Done When:**
  - `pytest tests/test_extraction.py` passes, and the sample document extracts fully with all persisted outputs.

---

### Sub-Task 5: Validate Docling Outputs (R7)

- **Status:** Completed
- **Objective:** Validate the raw Docling outputs for completeness, consistency, and structural integrity before normalization.
- **Related Requirements:** R7 (Output Validation)
- **Dependencies and Preconditions:** Sub-Task 4 (Docling extraction and persistence).
- **In Scope for This Sub-Task:**
  - Create `src/validation/__init__.py` and `src/validation/docling_validator.py`.
  - Implement `validate_docling_output(dl_doc: DoclingDocument, doc: DocumentSchema) -> ValidationReport` that checks:
    1. **Page count:** `dl_doc.page_count` matches `doc.page_count`.
    2. **Non-empty content:** At least one text item or table exists.
    3. **Structural consistency:** Every item references a valid page number (within range).
    4. **Item types:** All items have recognized types (text, table, figure, formula, caption, etc.).
    5. **Bounding boxes:** All items have valid bounding boxes (non-negative, within page dimensions).
    6. **Reading order:** Reading order indices are dense (no gaps) and monotonically increasing within each page.
    7. **Table check:** If any tables exist, they have at least one row and column.
  - Create `src/validation/models.py` — `ValidationReport` with: `is_valid: bool`, `doc_id: UUID`, `checks: List[ValidationCheck]`, `errors: List[str]`, `warnings: List[str]`, `summary: str`.
  - Implement `ValidationCheck` with: `check_name: str`, `passed: bool`, `details: str`.
  - If validation fails (non-critical), set status to `"extraction_complete_with_warnings"` in manifest and log warnings. Only set `"extraction_failed"` on critical failures.
- **Out of Scope for This Sub-Task:**
  - No content quality evaluation (e.g., OCR accuracy, table structure correctness).
  - No downstream impact from validation failures beyond status flags.
- **Instructions:**
  1. Design checks to be independent and additive (each check produces a `ValidationCheck`, failures do not stop subsequent checks).
  2. Critical failures are: page count mismatch, empty document, all items have invalid bboxes.
  3. Non-critical warnings: some items missing captions, minor reading order gaps.
- **Acceptance Criteria:**
  - `validate_docling_output` returns a `ValidationReport` for any input.
  - A well-formed Docling output passes all critical checks.
  - An intentionally broken output (e.g., manipulated to have wrong page count) fails the relevant check.
  - The report is serializable to JSON.
- **Cautionary Points (Risks & Edge Cases):**
  - Docling may return 0 items for a page that failed layout analysis; that should be a warning, not a failure.
  - Bounding boxes may use different coordinate systems; ensure the validator normalizes or accounts for this.
- **Implementation Suggestions:**
  - Use the `src/schemas/geometry.py` `BoundingBox` for coordinate validation.
  - Write each check as a separate method or function for testability.
- **Testing Suggestions:**
  - Unit-test each validation check with synthetic good/bad data.
  - Integration-test against the output of Sub-Task 4.
  - Verify `ValidationReport` serializes to JSON and back.
- **Done When:**
  - `pytest tests/test_validation.py` passes all validation tests.

---

### Sub-Task 6: Define ChunkMetadata Model and Chunk ID Strategy (R2, R3, R8)

- **Status:** Completed
- **Objective:** Define ``ChunkMetadata`` — the single source of truth for every chunk produced by the pipeline.  Also implement deterministic UUIDv5 generation for chunk IDs.
- **Related Requirements:** R2 (Internal Schema), R3 (ID Strategy), R8 (ChunkMetadata Model)
- **Dependencies and Preconditions:** Sub-Task 1 (schemas directory exists).  Sub-Task 2 (project structure).
- **In Scope for This Sub-Task:**
  - Create ``src/chunking/models.py`` with:
    1. ``CHUNK_ID_NAMESPACE`` — a fixed UUIDv4 used as the namespace for all chunk UUIDs.
    2. ``make_chunk_id(document_hash, sequence_number, chunk_types) -> str`` — deterministic UUIDv5 from ``{doc_hash}|{seq}|{sorted_types}``.
    3. ``ChunkMetadata`` — frozen Pydantic v2 ``BaseModel`` with all fields discussed:

       *Identity:* ``chunk_id``, ``document_name``, ``document_type``, ``document_hash``.
       *Embedding:* ``embedding_type`` (``"text"`` | ``"image"`` | ``"textual_description"``), ``chunk_text``.
       *Structure:* ``chunk_types``, ``section_path``, ``section_headings``, ``page_numbers``, ``sequence_number``.
       *Visual:* ``image_type``, ``image_uri``, ``caption_text``, ``caption_number``.
       *Graph linking:* ``element_self_refs``, ``refers_to``, ``relates_to``.
       *Quality:* ``token_count``.
- **Out of Scope for This Sub-Task:**
  - No chunking logic — just the data model.
  - No embedding generation — fields are populated but not embedded yet.
- **Instructions:**
  1. Use the ``make_chunk_id`` formula: ``uuid.uuid5(CHUNK_ID_NAMESPACE, f"doc:{hash}|seq:{seq:06d}|types:{sorted_types}")``.
  2. ``embedding_type`` is a ``Literal["text", "image", "textual_description"]``.
  3. ``image_type`` is ``Optional[Literal["picture", "table"]]``.
  4. All list fields default to ``[]``; ``token_count`` defaults to ``0``.
  5. Model must be ``frozen=True``.
- **Acceptance Criteria:**
  - ``from src.chunking import ChunkMetadata, make_chunk_id`` works.
  - ``make_chunk_id`` produces the same output for the same inputs; different for different inputs.
  - ``ChunkMetadata`` can be instantiated with all required fields and raises ``ValidationError`` for invalid ``embedding_type`` values.
  - Model serializes via ``.model_dump()`` and deserializes via ``.model_validate()``.
- **Cautionary Points (Risks & Edge Cases):**
  - The ``chunk_id`` must be a string (not UUID object) for Weaviate compatibility.  ``uuid.uuid5()`` returns a ``UUID``; cast to ``str``.
  - Don't include ``chunk_text`` in the ID — it changes across Docling versions, breaking determinism for the same logical chunk.
- **Implementation Suggestions:**
  - Use ``ConfigDict(frozen=True)`` — not the deprecated ``class Config`` style.
  - Add docstrings to every field describing its role in vector DB filtering or graph DB edge creation.
- **Testing Suggestions:**
  - Create ``tests/test_chunking_models.py``: test UUID determinism, field validation, serialization round-trip.
- **Done When:**
  - ``src/chunking/models.py`` is implemented and ``pytest tests/test_chunking_models.py`` passes.

---

### Sub-Task 7: Implement Docling HybridChunker Wrapper with Metadata Extraction (R9)

- **Status:** Completed
- **Objective:** Wrap Docling's ``HybridChunker``, extract all ``ChunkMetadata`` fields from ``DocChunk.meta``, and produce a list of text chunks.
- **Related Requirements:** R9 (HybridChunker Wrapper)
- **Dependencies and Preconditions:** Sub-Task 6 (ChunkMetadata model exists).  Sub-Task 4 (Docling extraction works).
- **In Scope for This Sub-Task:**
  - Create ``src/chunking/docling_chunker.py`` with:
    1. ``create_hybrid_chunker()`` — factory for a configured ``HybridChunker`` + ``HuggingFaceTokenizer``.
    2. ``_build_text_lookup()`` — maps ``self_ref → text`` for all text items in the document.
    3. ``_build_picture_table_lookup()`` — maps ``self_ref → item`` for all pictures and tables.
    4. ``_build_picture_table_section_lookup()`` — initialises per-ref section heading accumulators.
    5. ``_extract_caption_text()`` — resolves a picture/table item's caption text via ``captions[0].cref`` or ``children[0].cref``.
    6. ``extract_chunk_metadata()`` — produces a ``ChunkMetadata`` from a single ``DocChunk``:
       * ``chunk_types`` and ``element_self_refs`` from ``meta.doc_items``.
       * ``page_numbers`` from ``item.prov[*].page_no`` (deduplicated, sorted).
       * ``section_path`` = ``" > ".join(meta.headings)``.
       * ``image_type``, ``image_uri``, ``caption_text``, ``caption_number`` for visual elements.
       * ``chunk_text`` from ``chunker.contextualize(chunk)``.
       * ``token_count`` from ``tokenizer.count_tokens(text)``.
    7. ``build_chunk_metadata_list()`` — main orchestrator; chunks the document and returns ``List[ChunkMetadata]``.  Optionally writes raw chunk JSON and metadata JSON to ``output_dir``.
- **Out of Scope for This Sub-Task:**
  - No visual enrichment (image / textual_description chunks) — that is Sub-Task 8.
  - No cross-reference resolution.
  - No embedding.
- **Instructions:**
  1. The ``tokenizer`` argument to ``extract_chunk_metadata`` is the ``HuggingFaceTokenizer`` from the chunker (``chunker.tokenizer``).
  2. Page numbers: iterate ``item.prov`` (list of ``ProvenanceItem``), collect ``prov.page_no``, deduplicate via ``sorted(set(...))``.
  3. Caption number: use the existing ``src/utils/caption_extractor.extract_caption_label()``.
  4. Section accumulation: ``pic_table_section_lookup`` is mutated in-place by ``extract_chunk_metadata``; the caller passes the same dict to all calls so sections accumulate across chunks.
  5. The ``DocChunk.model_validate(chunk)`` call is needed because ``HybridChunker`` yields raw objects — validate to access typed ``meta`` fields.
- **Acceptance Criteria:**
  - ``build_chunk_metadata_list(conv_result)`` returns a list of ``ChunkMetadata``, one per chunk.
  - Every chunk has populated ``chunk_id``, ``document_name``, ``chunk_text``, ``page_numbers``, ``section_path``, ``token_count``, and ``element_self_refs``.
  - Chunks containing pictures/tables have ``image_type``, ``image_uri``, ``caption_text`` populated.
  - ``section_path`` is correct (verified against ``meta.headings`` of a known document).
  - Optional ``output_dir`` parameter writes both JSON files.
- **Cautionary Points (Risks & Edge Cases):**
  - ``meta.doc_items`` may be empty for some chunks (e.g., a chunk that Docling produces with no labelled items).  Handle gracefully — all lists default to empty, ``page_numbers`` stays empty, ``section_path`` may be ``""``.
  - Picture items may not have a ``captions`` list — the ``_extract_caption_text`` function handles ``None`` correctly.
  - ``chunk_text`` for visual-only chunks may be very short (just the caption or a few words).  This is expected — ``token_count`` will reflect it.
- **Implementation Suggestions:**
  - Keep all helper functions module-private (``_`` prefix) except the public API.
  - Use ``logging.getLogger(__name__)`` with ``INFO`` for chunk counts, ``DEBUG`` for per-chunk details.
- **Testing Suggestions:**
  - Create ``tests/test_docling_chunker.py`` with a fixture that runs ``build_chunk_metadata_list`` on the sample PDF.
  - Verify chunk count > 0, all fields present, page numbers valid, section paths parseable.
  - Test with ``output_dir`` — verify JSON files exist and are valid.
- **Done When:**
  - ``pytest tests/test_docling_chunker.py`` passes, and manual inspection of chunk metadata for ``2502.04644v1.pdf`` shows correct values.

---

### Sub-Task 8: Implement Visual Element Enrichment (R10)

- **Status:** Completed
- **Objective:** For each picture and table in the document, generate an LLM-powered textual description and create two additional ``ChunkMetadata`` entries: one ``embedding_type="image"`` and one ``embedding_type="textual_description"``.
- **Related Requirements:** R10 (Visual Element Enrichment)
- **Dependencies and Preconditions:** Sub-Task 7 (chunk metadata list produced, pic_table_section_lookup populated). Sub-Task 6 (ChunkMetadata model). ``src/utils/instructor_api_response.py`` must be functional.
- **In Scope for This Sub-Task:**
  - Create ``src/chunking/visual_enricher.py`` with:
    1. ``_get_caption_for_item()`` — extract caption text from a picture/table item.
    2. ``_describe_image_async()`` — async LLM call using ``get_llm_response_from_instructor`` with ``_ImageDescription`` pydantic response model.
    3. ``generate_all_image_descriptions()`` — async function that calls ``_describe_image_async`` for all pictures/tables concurrently via ``asyncio.gather`` + ``asyncio.Semaphore(max_concurrent)``.
    4. ``enrich_visual_chunks()`` — sync wrapper that:
       * Calls ``asyncio.run(generate_all_image_descriptions(...))`` once (NOT per item).
       * For each picture/table, creates two ``ChunkMetadata``:
         - ``embedding_type="image"`` — ``chunk_text`` = caption (reference only), ``image_uri`` is what gets embedded later.
         - ``embedding_type="textual_description"`` — ``chunk_text`` = LLM description (embedded later).
       * Sequence numbers continue from the last text chunk.
       * Returns the combined list: text chunks + visual chunks.
- **Out of Scope for This Sub-Task:**
  - No actual embedding of images or descriptions — that is Sub-Task 12.
  - No chunking strategy beyond what Docling provides.
- **Instructions:**
  1. The ``_ImageDescription`` pydantic model should have ``description: str`` and ``keywords: List[str]`` fields.
  2. Concatenate output as: ``f"Image Caption: {caption}\\nImage Description: {description}\\nKeywords: {keywords}"``.
  3. Failed LLM calls (``return_exceptions=True`` in ``asyncio.gather``) produce empty descriptions — log a warning per failure but don't halt.
  4. ``_label_value()`` normalises ``DocItemLabel`` enum values to lowercase strings (``"picture"``, ``"table"``) for the ``Literal`` type.
  5. Section info for visual chunks comes from ``pic_table_section_lookup`` (accumulated by Sub-Task 7's ``extract_chunk_metadata``).
- **Acceptance Criteria:**
  - ``enrich_visual_chunks`` returns a list with text chunks + 2× picture_count + 2× table_count chunks.
  - Each visual chunk has correct ``embedding_type``, ``image_uri``, ``caption_text``, and ``chunk_text``.
  - LLM descriptions are generated concurrently (verify timing is faster than sequential).
  - Chunk IDs are deterministic and unique.
- **Cautionary Points (Risks & Edge Cases):**
  - The instructor API may be unavailable or rate-limited.  Handle failures gracefully — empty descriptions should not crash the pipeline.
  - If ``pic_table_lookup`` is empty, return the original list unchanged with an info log.
  - ``asyncio.run()`` cannot be called from within an already-running event loop.  The sync wrapper is designed to be the entry point; if integrating into an async app, call ``generate_all_image_descriptions`` directly.
- **Implementation Suggestions:**
  - Use ``asyncio.Semaphore(max_concurrent)`` to limit concurrent API calls (default 5).
  - The ``_describe_image_async`` coroutine should be thin — just the LLM call, leaving concurrency control to the caller.
- **Testing Suggestions:**
  - Mock ``get_llm_response_from_instructor`` to return a fixed ``_ImageDescription`` — verify the two chunks are created correctly.
  - Test with empty ``pic_table_lookup`` — verify no error, original list returned.
  - Test with a failing API call — verify warning logged, empty description, chunk still created.
- **Done When:**
  - ``pytest tests/test_visual_enricher.py`` passes with mocked LLM calls.

---

### Sub-Task 9: Implement Cross-Reference Resolution (R11)

- **Status:** Completed
- **Objective:** Scan chunk text for "see Figure 3", "Table IV", "as shown in Fig. 1" patterns and populate ``refers_to`` fields by matching references against a caption index built from the raw Docling picture/table items.
- **Related Requirements:** R11 (Cross-Reference Resolution)
- **Dependencies and Preconditions:** Sub-Task 8 (all chunks created, ``pic_table_lookup`` available).  Sub-Task 6 (ChunkMetadata model).  ``extract_caption_label()`` from ``src/utils/caption_extractor.py``.
- **In Scope for This Sub-Task:**
  - New function ``find_all_caption_refs(text) -> List[str]`` in ``src/utils/caption_extractor.py``.  Reuses the same ``_CAPTION_RE`` regex as ``extract_caption_label()`` but scans with ``finditer`` (not ``match``) to locate all references anywhere in a string.  Each returned string is a normalized label like ``"figure 1"`` or ``"table IV"``.
  - Create ``src/chunking/cross_reference_resolver.py`` with:
    1. ``_build_caption_index(pic_table_lookup, text_lookup, chunk_metadatas) -> Dict[str, str]``
       - Iterates every ``(ref, item)`` in ``pic_table_lookup``.
       - Extracts the caption text from ``text_lookup`` via the item's ``captions[0].cref``.
       - Calls ``extract_caption_label(caption_text)`` to get a normalized label (e.g. ``"figure 1"``).
       - Finds which chunk contains this visual element by checking ``element_self_refs`` in each chunk (build a ``{self_ref → chunk_id}`` reverse lookup once).
       - Inserts ``label → chunk_id`` into the index.
    2. ``resolve_cross_references(chunk_metadatas, pic_table_lookup, text_lookup) -> List[ChunkMetadata]``
       - Builds the caption index via ``_build_caption_index``.
       - For each chunk, calls ``find_all_caption_refs(chunk.chunk_text)``.
       - For each matched label, looks up the target ``chunk_id`` in the index.
       - Filters out self-references (target == current chunk's own ``chunk_id``).
       - Deduplicates and sorts ``refers_to``.
       - Returns a new list of ``ChunkMetadata`` via ``model_copy(update={"refers_to": [...]})`` (chunks are frozen — never mutated).
- **Out of Scope for This Sub-Task:**
  - No "relates_to" population (handled by Sub-Task 11 via top-k similarity).
  - No graph DB loading.
  - No changes to the ``ChunkMetadata`` model or its ``caption_number`` field.
- **Instructions:**
  1. **``find_all_caption_refs``**: Clone ``_CAPTION_RE`` from ``caption_extractor.py``, remove the ``^`` anchor, and use ``re.finditer``.  For each match, normalize the type through ``VALID_IMAGE_TYPES`` and return ``f"{normalized_type} {number}"`` — exactly the same string format as ``extract_caption_label()``.  This guarantees that a reference "Fig. 3" in chunk text and a caption "Figure 3" produce the same key ``"figure 3"``.
  2. **Caption index**: Built from raw Docling items (``pic_table_lookup``), not from ``ChunkMetadata.caption_number``.  This avoids depending on chunk-metadata representation and sidesteps any discrepancies between how the chunker populates ``caption_number`` vs. how references appear in text.
  3. **Reverse lookup**: Build ``Dict[str, str]`` mapping each ``element_self_ref`` to its containing chunk's ``chunk_id``.  This is O(n·m) but n ≤ ~100 chunks and m ≤ ~50 visual elements per document, so trivial.
  4. **No mutation**: All functions return new ``ChunkMetadata`` instances.  Use ``model_copy(update={...})`` on the original.
- **Acceptance Criteria:**
  - Chunk containing "see Figure 1" resolves to the chunk whose caption label is ``"figure 1"``.
  - Abbreviation matching works: "Fig. 3" → ``"figure 3"``, "Eqn 5" → ``"equation 5"``.
  - Multiple references in one chunk → multiple unique entries in ``refers_to``, sorted.
  - No match when a number is mentioned without a type keyword (e.g., "see Section 3" with no "Figure"/"Table" prefix — ``find_all_caption_refs`` returns empty and the chunk is not modified).
  - Self-references (a visual chunk's own chunk_text referencing its own caption) are filtered out.
  - If a reference's ``(type, number)`` is not in the caption index, it is silently skipped (no crash, no empty entry).
- **Cautionary Points (Risks & Edge Cases):**
  - **Regex divergence**: ``find_all_caption_refs`` must use the **exact same regex logic** as ``extract_caption_label`` — same ``VALID_IMAGE_TYPES`` normalization, same number patterns.  If they diverge, references won't match their caption index entries.
  - **Case sensitivity**: The regex is already ``re.IGNORECASE``.  "Figure", "figure", "FIGURE" all match.
  - **Performance**: ``finditer`` on every chunk's text is fast for regex; the bottleneck is the O(n·m) chunk→ref lookup during index construction, which is negligible at current scale.
- **Implementation Suggestions:**
  - Extract the shared regex logic from ``caption_extractor.py`` into a module-level helper that both ``extract_caption_label`` (via ``match``) and ``find_all_caption_refs`` (via ``finditer``) call.  This guarantees they never diverge.
  - The caption index key is a plain string (e.g. ``"figure 1"``) — no tuple needed since ``extract_caption_label`` returns strings.
- **Testing Suggestions:**
  - Create a mock ``pic_table_lookup`` with known captions and a mock chunk list — verify resolution.
  - Test abbreviation variants: "Fig. 1", "Fig 1", "Figure 1", "FIG 1" all resolve to the same target.
  - Test with no references, multiple references, ambiguous references (type word present but not followed by a number).
  - Test self-reference filtering.
- **Done When:**
  - ``pytest tests/test_cross_reference_resolver.py`` passes.
  - ``pytest tests/test_caption_extractor.py`` still passes (no regressions from the new ``find_all_caption_refs`` function).

---

### Sub-Task 10: Implement Batch Multimodal Embedding Pipeline (R12)

- **Status:** Completed
- **Objective:** Build a batch ``items_to_encode`` list from chunk metadata, dispatch by ``embedding_type``, and encode all items using a multimodal embedding model.
- **Related Requirements:** R12 (Multimodal Batch Embedding)
- **Dependencies and Preconditions:** Sub-Task 8 (all chunks created).  Multimodal embedding model available (e.g., ``Qwen/Qwen3-VL-Embedding-2B`` or similar via ``sentence-transformers``).
- **In Scope for This Sub-Task:**
  - Create ``src/retrieval/embedding_pipeline.py`` with:
    1. ``build_encode_items(chunks) -> List[Dict]`` — returns ``[{chunk_id, embedding_type, content (text or image URI), metadata}]``.
    2. ``encode_batch(items, model) -> List[np.ndarray]`` — dispatches to the correct encoder based on ``embedding_type``.
    3. ``attach_embeddings(chunks, embeddings) -> List[Dict]`` — combines chunk metadata with embedding vectors for Weaviate loading.
- **Out of Scope for This Sub-Task:**
  - No Weaviate loading yet (Sub-Task 11).
  - No retrieval / query logic.
- **Instructions:**
  1. For ``embedding_type="image"``, pass ``image_uri`` to the multimodal encoder's image path.
  2. For ``embedding_type in ("text", "textual_description")``, pass ``chunk_text`` to the text encoder.
  3. Use batched encoding where the model supports it.
  4. Normalize embeddings to unit vectors for cosine similarity.
- **Acceptance Criteria:**
  - ``build_encode_items`` correctly dispatches by embedding_type.
  - ``encode_batch`` returns embeddings for all items.
  - Embeddings are unit-normalized.
- **Cautionary Points (Risks & Edge Cases):**
  - Large batches may OOM.  Process in sub-batches (e.g., 32 items at a time).
  - Image URIs may be invalid or missing — skip with a warning, don't crash.
- **Implementation Suggestions:**
  - Use ``SentenceTransformer`` for text/image models that support both modalities.
  - Store embeddings as ``List[float]`` for JSON serialization to Weaviate.
- **Testing Suggestions:**
  - Mock the embedding model to return fixed vectors — verify dispatch logic.
  - Test with empty items list, items with invalid URIs.
- **Done When:**
  - ``pytest tests/test_embedding_pipeline.py`` passes.

---

### Sub-Task 11: Populate ``relates_to`` via Top-k Nearest Neighbors (R12b)

- **Status:** Completed
- **Objective:** For each chunk, compute its top-k most similar chunks by cosine similarity on the embeddings from Sub-Task 10 and populate ``relates_to``.  Sparse, deterministic, no clustering dependency.
- **Related Requirements:** R12b (Relates-to via Top-k Similarity)
- **Dependencies and Preconditions:** Sub-Task 10 (embeddings computed and attached to chunks).  Sub-Task 6 (ChunkMetadata model with ``relates_to`` and ``element_self_refs`` fields).
- **In Scope for This Sub-Task:**
  - Create ``src/retrieval/similarity.py`` with:
    1. ``compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray`` — builds the ``n × n`` cosine similarity matrix from unit-normalized embedding vectors.  Pure numpy.
    2. ``populate_relates_to(chunks: List[ChunkMetadata], embeddings: np.ndarray, *, top_k: int = 3, min_similarity: float = 0.75) -> List[ChunkMetadata]``
       - For each chunk at index ``i``:
         - Finds the ``top_k`` most similar chunks (excluding self, index ``i``).
         - Filters siblings: any candidate chunk whose ``element_self_refs`` **intersects** with the current chunk's ``element_self_refs`` is excluded (picture ↔ textual_description are the same element).
         - Applies ``min_similarity=0.75`` — if no candidate exceeds this threshold, ``relates_to`` stays empty.
         - Sorts by descending similarity.
       - Returns a new list of ``ChunkMetadata`` via ``model_copy(update={"relates_to": [...]})``.
- **Out of Scope for This Sub-Task:**
  - Cross-document similarity (that is Weaviate's job at query time).
  - HDBSCAN or any clustering-based approach.
  - Incremental / streaming updates (re-computing for a single document is trivial).
- **Instructions:**
  1. Embeddings are expected as an ``(n, d)`` numpy array where row ``i`` corresponds to ``chunks[i]``.
  2. Normalize rows to unit vectors before computing the dot-product matrix.
  3. For each row ``i``, zero out ``M[i, i]`` (self-similarity) before taking ``top_k``.
  4. Build a set of sibling indices for each chunk: two chunks are siblings if ``bool(set(A.element_self_refs) & set(B.element_self_refs))``.  Pre-compute this once per chunk pair or per chunk (since visual siblings are typically 2-3 chunks per element).
  5. Apply ``min_similarity`` as a hard cutoff — any candidate below this is discarded.  A chunk with no qualifying neighbors gets ``relates_to=[]``.
- **Acceptance Criteria:**
  - Each chunk has 0–3 entries in ``relates_to``, sorted by descending similarity.
  - No chunk relates to itself (index ``i`` is excluded).
  - No chunk relates to its own sibling (same element via image + textual_description).
  - Chunks below ``min_similarity=0.75`` get empty ``relates_to``.
  - Deterministic: same embeddings → same ``relates_to`` every run.
  - Pure numpy — no HDBSCAN, no sklearn clustering import.
- **Cautionary Points (Risks & Edge Cases):**
  - **Embedding quality dependent:** If the multimodal model produces poor embeddings (e.g., near-identical vectors for boilerplate headers), ``relates_to`` may contain false positives.  Validation on real document output is essential.
  - **Sibling filter completeness:** Two chunks sharing any ``element_self_ref`` are definitively siblings.  Adjacent-but-distinct visual elements (Figure 1 and Figure 2 on the same page) may also be near-identical in embedding space — that is acceptable; they ARE semantically related content.
  - **``min_similarity=0.75``** is a deliberate tradeoff: higher threshold → fewer edges → cleaner graph but quieter traversal.  At 0.75 cosine, only genuinely similar chunks survive.  This value can be exposed as a pipeline parameter if needed later.
- **Implementation Suggestions:**
  - Use ``numpy.einsum("ij,kj->ik", normalized, normalized)`` for the dot-product matrix — avoids explicit looping and is cache-friendly.
  - Use ``numpy.argpartition`` for O(n) partial sort to find top_k indices without full sort.
- **Testing Suggestions:**
  - Create 10 mock chunks with synthetic 2d embeddings — verify top-3 neighbors.
  - Test sibling exclusion with chunks sharing ``element_self_refs``.
  - Test threshold: with ``min_similarity=0.99``, verify most chunks get empty ``relates_to``.
  - Test empty list (n=0) and single-chunk list (n=1) edge cases.
- **Done When:**
  - ``pytest tests/test_similarity.py`` passes.

---

### Sub-Task 12: Weaviate Vector Database Integration (R13)

- **Status:** Pending
- **Objective:** Load chunks and their embeddings into Weaviate with all filterable properties exposed for hybrid retrieval.
- **Related Requirements:** R13 (Weaviate Integration)
- **Dependencies and Preconditions:** Sub-Task 10 (embeddings computed).  Weaviate instance running (local or cloud).
- **In Scope for This Sub-Task:**
  - Create ``src/retrieval/weaviate_loader.py`` with:
    1. ``create_schema(client)`` — defines the Weaviate collection with vectorizer=none (we bring our own vectors) and all filterable properties.
    2. ``ingest_chunks(client, chunks_with_embeddings)`` — batch-inserts chunks into Weaviate.
  - Filterable Weaviate properties must include: ``document_hash``, ``page_numbers`` (int array), ``section_path`` (text), ``caption_number`` (text), ``chunk_types`` (text array), ``token_count`` (int), ``embedding_type`` (text), ``image_type`` (text).
- **Out of Scope for This Sub-Task:**
  - No retrieval / query implementation.
  - No hybrid search configuration.
- **Instructions:**
  1. Use Weaviate v4 Python client.
  2. Collection name: ``"DocumentChunks"``.
  3. Vector index: HNSW with cosine distance.
  4. Batch import with error handling — log failed inserts, continue.
- **Acceptance Criteria:**
  - Collection is created with correct schema.
  - All chunks from a processed document are stored and queryable.
  - Filter queries like ``page_numbers contains 5`` return correct results.
- **Cautionary Points (Risks & Edge Cases):**
  - Weaviate may have per-batch limits (default 100 objects).  Split large imports.
  - UUID conflicts: use ``chunk_id`` as the Weaviate object UUID.  Re-import should be idempotent.
- **Implementation Suggestions:**
  - Use ``weaviate.classes.config.Configure.NamedVectors.none`` since we provide pre-computed vectors.
- **Testing Suggestions:**
  - Integration test with a local Weaviate instance (or mock).
  - Verify idempotent re-import.
- **Done When:**
  - ``pytest tests/test_weaviate_loader.py`` passes against a running Weaviate instance.

---

### Sub-Task 13: Graph Database Construction (R14)

- **Status:** Pending
- **Objective:** Build a property graph from chunk metadata: document node, chunk nodes, section nodes, and edges for sequential order, section membership, cross-references, and relatedness.
- **Related Requirements:** R14 (Graph DB Construction)
- **Dependencies and Preconditions:** Sub-Task 9 (cross-references resolved).  Sub-Task 11 (``relates_to`` populated).  Sub-Task 8 (all chunks created).  Neo4j or Apache AGE available.
- **In Scope for This Sub-Task:**
  - Create ``src/retrieval/graph_builder.py`` with:
    1. ``build_nodes(chunks)`` — creates document node, chunk nodes, section nodes (one per unique ``section_path``).
    2. ``build_edges(chunks)`` — creates:
       * ``(chunk_N)-[:FOLLOWS]->(chunk_N+1)`` from ``sequence_number`` order.
       * ``(chunk)-[:BELONGS_TO]->(section)`` from ``section_path``.
       * ``(chunk_A)-[:REFERS_TO]->(chunk_B)`` from ``refers_to`` lists.
       * ``(chunk_A)-[:RELATES_TO]->(chunk_B)`` from ``relates_to`` lists.
    3. ``load_graph(client, nodes, edges)`` — loads into the graph DB.
- **Out of Scope for This Sub-Task:**
  - No graph-based context expansion or traversal for retrieval — that is future RAG work.
- **Instructions:**
  1. Use Cypher for Neo4j or openCypher for AGE.
  2. Node properties: chunk nodes carry ``chunk_id``, ``document_hash``, ``chunk_types``, ``section_path``, ``page_numbers``, ``caption_number``, ``token_count``.
  3. Edge properties: ``weight`` (default 1.0), ``relationship_type``.
- **Acceptance Criteria:**
  - Graph contains correct node and edge counts matching the chunk metadata.
  - ``FOLLOWS`` edges form a valid linked list (chunk N → chunk N+1).
  - ``REFERS_TO`` edges match the resolved cross-references.
- **Cautionary Points (Risks & Edge Cases):**
  - Duplicate edges: deduplicate before loading.
  - Large graphs: batch the Cypher statements.
- **Testing Suggestions:**
  - Integration test with Neo4j community edition (or mock).
  - Verify graph queries return expected traversal results.
- **Done When:**
  - ``pytest tests/test_graph_builder.py`` passes.

---

### Sub-Task 14: Pipeline Orchestrator, Export, and Validation (R15)

- **Status:** Pending
- **Objective:** Chain all stages into a single entry point, add CLI, export artifacts, and validate outputs.
- **Related Requirements:** R15 (Pipeline Orchestrator)
- **Dependencies and Preconditions:** All sub-tasks 6-13 complete.
- **In Scope for This Sub-Task:**
  - Create ``src/pipeline.py`` with an ``ExtractionPipeline`` class that runs: ingest → extract → validate → chunk → enrich visuals → resolve cross-refs → embed → populate relates-to → load Weaviate → build graph.
  - CLI entry point via ``console_scripts`` in ``pyproject.toml``.
  - Export: chunk metadata JSONL, metadata summary JSON.
  - Validation: chunk coverage (every picture/table has a chunk), ID uniqueness, embedding completeness.
- **Acceptance Criteria:**
  - ``edr-pipeline --source doc.pdf --output ./outputs`` runs end-to-end.
  - All output files produced.
  - Idempotent re-run produces identical output.
- **Done When:**
  - ``pytest tests/test_pipeline.py`` passes, CLI works end-to-end.

## Final Integration & Verification

- **System-Wide Test:** Run the full pipeline on the sample document (``2502.04644v1.pdf``). Verify:
  - Chunk metadata JSON is produced with correct fields.
  - All three embedding types are represented (text, image, textual_description).
  - Chunk IDs are deterministic (re-run produces same IDs).
  - Pipeline completes without unhandled errors.
- **Completion Checklist:**
  - [ ] ``ChunkMetadata`` model defined and tested.
  - [ ] ``make_chunk_id()`` deterministic UUID generation works.
  - [ ] Docling HybridChunker wrapper extracts all metadata.
  - [ ] Visual enrichment produces image + description chunks concurrently.
  - [ ] Cross-reference resolver populates ``refers_to``.
  - [ ] Relates-to similarity pipeline populates ``relates_to`` (top-3 neighbors, min similarity 0.75).
  - [ ] Batch embedding pipeline dispatches by ``embedding_type``.
  - [ ] Weaviate collection created, chunks ingested with filterable properties.
  - [ ] Graph DB nodes and edges built from chunk metadata.
  - [ ] Pipeline orchestrator runs end-to-end from CLI.
  - [ ] All tests pass: ``pytest tests/``.
- **Performance Check:** The chunking + enrichment pipeline should process a 20-page document in under 3 minutes on CPU (excluding LLM description generation which depends on API latency).
- **Error Handling:** Every stage must handle failures gracefully, log the error, and continue (for non-critical stages) or abort with a clear message (for critical stages like ingestion/extraction).

## Open Questions

1. **Docling version pinning:** What specific version of ``docling`` is targeted?  Add ``docling>=2.0,<3.0`` to ``pyproject.toml`` dependencies.
2. **Multimodal embedding model:** ``Qwen/Qwen3-VL-Embedding-2B`` is referenced in the experimental script.  Is this the production model, or should ``sentence-transformers/all-MiniLM-L6-v2`` be used for text-only, with a separate model for images?  Decide before Sub-Task 10.
3. **Weaviate deployment:** Local Docker or Weaviate Cloud?  Determines connection config for Sub-Task 11.
4. **Graph DB choice:** Neo4j (Cypher) or Apache AGE (openCypher via PostgreSQL)?  Determines driver choice for Sub-Task 12.
