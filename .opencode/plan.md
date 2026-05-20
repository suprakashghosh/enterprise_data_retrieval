# Plan: Extraction and Structured Chunking Stage

## Objective

Build a production-grade extraction and structured chunking pipeline that ingests complex documents (PDFs with text, tables, charts, images, formulas, captions, footnotes, and rich layouts), extracts all elements via Docling, normalizes them into a well-defined internal schema, reconstructs document hierarchy, generates typed chunks with relationship metadata, and exports final artifacts (JSONL, Parquet, reports). This plan covers the full ingestion-to-chunks pipeline and **stops before** graph database loading, vector indexing, hybrid search, query decomposition, context expansion, and answer generation.

## Requirements Snapshot

- **R1 (Docling Extraction):** Use Docling as the primary extraction engine. Convert PDFs into Docling document format with full layout, table structure, OCR, and page-level detail.
- **R2 (Internal Schema):** Define and implement target schemas/data models for Document, Page, Section, Element (Paragraph/TextBlock, Table, Image, Chart, Graph, Formula), Chunk, Relationship, and Metadata.
- **R3 (ID Strategy):** Define a stable, deterministic ID strategy for documents, elements, and chunks enabling cross-referencing and idempotent reprocessing.
- **R4 (Project Structure):** Establish modular project structure with separate packages for ingestion, extraction, normalization, metadata, chunking, schemas, validation, and utilities.
- **R5 (Raw Document Storage):** Implement ingestion that stores raw PDFs immutably and associates them with a document record.
- **R6 (Docling Output Persistence):** Persist raw Docling outputs (JSON, markdown, page images, table images) alongside the document record for audit and re-processing.
- **R7 (Output Validation):** Validate Docling outputs for completeness, correct page counts, non-empty content, and structural consistency.
- **R8 (Normalization & Registry):** Normalize Docling's output into internal element objects and create an element registry (a lookup by element ID).
- **R9 (Positional & Structural Preservation):** Preserve page numbers, bounding boxes, reading order, section assignments, captions, and layout proximity for every element.
- **R10 (Hierarchy Reconstruction):** Reconstruct document hierarchy (Document → Page → Section → Element) and assign hierarchical section paths (e.g., "3.2.1") to all elements.
- **R11 (Table Processing):** Process tables into multiple representations (markdown, HTML, JSON, plain-text summary), extract table metadata (row count, column count, headers), and handle multi-page/spanned tables.
- **R12 (Image/Chart/Graph Processing):** Extract and save image/chart/graph assets, classify visual type (chart vs. diagram vs. photograph), prepare metadata, and set up extensibility for future vision-language descriptions.
- **R13 (Formula Processing):** Extract formulas (LaTeX or symbolic), classify inline vs. display formulas, and link to surrounding explanatory text.
- **R14 (Relationship Metadata):** Define and generate typed relationships (contains, belongs_to, relates_to, refers_to, describes, follows, precedes, explains, supports, has_caption, nearby, same_section_as) between elements.
- **R15 (Chunking):** Implement hierarchical chunking (by section boundary), semantic chunking (by semantic similarity), and cluster-semantic grouping (via embeddings + clustering). Integrate tables/images/charts/formulas into chunks via references and summaries rather than flattening to plain text alone.
- **R16 (Exports):** Export final artifacts: chunks JSONL/Parquet, elements JSONL, relationships JSONL, metadata JSON, and human-readable review reports (HTML or Markdown).
- **R17 (Validation & Benchmarks):** Add validation/evaluation checks for chunk quality, element coverage, relationship fidelity, and define sample benchmark documents for regression testing.
- **R18 (Out of Scope — Downstream):** Graph database creation/loading, vector database indexing, hybrid retrieval, query decomposition/expansion, graph-based context expansion, answer generation, and agentic reasoning productionization are explicitly out of scope for this plan.

## Scope

- Building the entire extraction-to-chunks pipeline: ingestion, Docling conversion, output persistence, validation, normalization, hierarchy reconstruction, element typing, relationship metadata generation, chunking (hierarchical, semantic, cluster-semantic), and export.
- Defining all internal data models (Pydantic) in a `schemas/` module.
- Modularizing into `ingestion/`, `extraction/`, `normalization/`, `metadata/`, `chunking/`, `schemas/`, `validation/`, and `utils/` packages under `src/`.
- Handling PDF documents with Docling; no other input formats in this phase.
- Supporting all document element types: text, tables, images, charts, graphs, formulas, captions, footnotes, headers/footers, lists.
- Generating human-readable review reports for debugging and quality inspection.
- Creating sample benchmark documents and automated validation tests.

## Assumptions and Constraints

- **Python 3.10+** is the target runtime; all implementation must be compatible.
- **Docling** (the `docling` PyPI package) is the extraction engine. Assume Docling version 2.x or later with `DocumentConverter`, `HybridChunker`, and the Docling document model available. The existing notebook `1_PDF_data_extraction.ipynb` shows a working Docling integration pattern.
- **GPU is not required** but acceleration (CUDA) should be leveraged when available for layout analysis and embedding-based chunking.
- The project currently has no test infrastructure; tests must be introduced alongside the implementation (pytest).
- All modules should be importable without a running GPU or external service; graceful fallbacks where sensible.
- Output directories (`data/`, `outputs/`) are gitignored already.
- The existing notebook code is prototype quality and should serve as reference, not as code to copy directly.
- Docling's built-in `HybridChunker` can be used as a foundation but must be wrapped and extended for the project's custom chunking needs.
- No data is sensitive/confidential in the sample documents; security controls (PII redaction, access control) are out of scope for this stage.

## Risks and Areas Requiring Care

- **Docling API volatility:** Docling is under active development. Pin the `docling` version in `requirements.txt` and isolate Docling interactions behind a thin adapter/abstraction layer so internal models do not leak.
- **Large documents:** Financial reports can be 300+ pages with hundreds of tables and figures. Ensure all processing is streaming or page-batched to avoid OOM. Use iterators for chunk output rather than loading everything in memory.
- **Multi-page table spanning:** A single logical table may span multiple pages. Docling may split them. The normalization step must attempt to re-merge or at least link split table parts.
- **Broken PDFs / OCR failures:** PDFs may be scanned images, have corrupted streams, or mixed content. Docling may produce incomplete output. Validation (R7) must catch these cases and report clearly rather than crashing.
- **Formula extraction quality:** Docling's formula extraction may be incomplete. The plan must accommodate fallback LaTeX or symbolic representation and not assume perfect extraction.
- **ID determinism and idempotency:** If a pipeline step is re-run on the same input, it should produce the same IDs. This requires content-hash-based or deterministic UUID strategies.
- **Chunk boundary quality:** Poorly chosen chunk boundaries can break relationships. The chunking step must not split a table away from its caption, or a paragraph from its footnotes.
- **Circular or duplicate relationships:** Relationship generation must avoid self-referencing edges and deduplicate structural links (e.g., same `follows` edge between identical element pairs).
- **No downstream consumers defined yet:** The schema and export formats must be designed for extensibility but not over-engineered. Keep it simple; downstream graph and vector stages can adapt later.

## Core Concepts

### Internal Data Model Overview

The pipeline transforms raw PDFs through several stages. The core data model evolves as follows:

```
PDF (raw bytes)
  │  [Docling conversion]
  ▼
DoclingDocument (Docling's native representation)
  │  [Normalization]
  ▼
InternalDocument (our Pydantic schema)
  ├── Document metadata (title, source path, hash, timestamps)
  ├── Pages[] (page number, dimensions, relationships)
  │     └── Elements[] (typed: TextBlock, Table, Image, Chart, Formula, etc.)
  │           ├── BoundingBox, reading_order, section_path
  │           ├── content (text, markdown, asset_ref)
  │           └── relationships[] (target_id, type, metadata)
  └── Relationships[] (element-to-element graph edges)
  │  [Chunking]
  ▼
Chunk[] (typed for content type)
  ├── content (text with element references like {{ELEM:UUID}})
  ├── element_refs[] (list of element UUIDs included)
  ├── chunk_type (hierarchical / semantic / cluster)
  ├── section_path, page_range, metadata
  └── relationships[]
```

### ID Strategy

Each document, element, and chunk gets a deterministic UUIDv5 generated from stable inputs:

| Entity | UUID Namespace | Input to Hash |
|--------|---------------|----------------|
| Document | `docling-project-doc` | SHA256 of raw file content |
| Element | `docling-project-elem` | `{doc_id}:{page_num}:{reading_order}:{element_type}` |
| Chunk | `docling-project-chunk` | `{doc_id}:{chunk_type}:{section_path}:{element_count}` |

This ensures re-running the pipeline on the same input yields the same IDs, enabling idempotent incremental processing.

### Relationship Types

| Relationship | Direction | Meaning |
|-------------|-----------|---------|
| `contains` | parent → child | Structural containment (Document contains Page, Section contains Element) |
| `belongs_to` | child → parent | Inverse of contains |
| `follows` | A → B | A appears immediately after B in reading order |
| `precedes` | A → B | A appears immediately before B |
| `refers_to` | A → B | Element A references element B (e.g., "see Table 3") |
| `relates_to` | A ↔ B | General semantic relatedness (undirected in data, stored as symmetric) |
| `describes` | A → B | Text element A describes visual element B (chart/image) |
| `explains` | A → B | Text element A explains formula B |
| `supports` | A → B | Element A provides supporting data for element B |
| `has_caption` | visual → text | The visual element has a caption text element |
| `nearby` | A ↔ B | Elements are spatially proximate (same page, within threshold distance) |
| `same_section_as` | A ↔ B | Elements share the same section path |

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

### Sub-Task 6: Normalize Docling Output into Internal Objects and Create Element Registry (R8, R9)

- **Status:** Pending
- **Objective:** Convert the raw Docling document into the project's internal `DocumentSchema` with all typed elements, preserving page numbers, bounding boxes, reading order, section assignments, captions, and layout proximity.
- **Related Requirements:** R8 (Normalization & Registry), R9 (Positional & Structural Preservation)
- **Dependencies and Preconditions:** Sub-Task 5 (validation), Sub-Task 1 (schemas), Sub-Task 4 (Docling output persisted).
- **In Scope for This Sub-Task:**
  - Create `src/normalization/__init__.py` and `src/normalization/docling_normalizer.py`.
  - Implement `normalize_document(doc: DocumentSchema, dl_doc: DoclingDocument) -> DocumentSchema` that:
    1. Iterates over every item in `dl_doc` (text items, tables, figures, formulas, etc.).
    2. Maps each Docling item type to the appropriate `ElementSchema` subclass.
    3. Extracts bounding boxes (converting coordinate systems to normalized 0-1).
    4. Assigns `reading_order` from Docling's provenance/ordering (falling back to spatial top-to-bottom, left-to-right).
    5. Assigns `element_id` using the deterministic UUID strategy.
    6. Links captions to their parent elements (Docling often provides `caption_of` references).
    7. Preserves the original content string and any available structured content (table cells, formula LaTeX).
    8. Creates `PageSchema` objects for each page with page-level metadata.
  - Implement `ElementRegistry` class:
    - `add(element: ElementSchema)`
    - `get(element_id: UUID) -> ElementSchema`
    - `get_by_page(page_num: int) -> List[ElementSchema]`
    - `get_by_type(element_type: str) -> List[ElementSchema]`
    - `iter_in_reading_order() -> Iterator[ElementSchema]`
    - Stores in-memory dict for O(1) lookup.
  - Implement `preserve_proximity(elements: List[ElementSchema]) -> None` that computes and stores:
    - Page-level left/right/top/bottom neighbors within a configurable distance threshold.
    - Same-page, same-column elements as being in proximity.
  - Handle text items that span multiple lines — merge into logical paragraphs based on font size, spacing, and style.
- **Out of Scope for This Sub-Task:**
  - No hierarchy reconstruction (sections) — that is Sub-Task 8.
  - No relationship generation — that is Sub-Task 9 (but proximity relationships are stored here).
  - No chunking.
- **Instructions:**
  1. Study Docling's output format by examining `dl_doc.export_to_dict()` on the sample PDF. Identify the item types and their fields.
  2. Create a mapping dict: `DOCLING_TYPE_TO_INTERNAL_TYPE`.
  3. For reading order: Docling may provide `provenance` with ordering. If not, implement a spatial sort (top-to-bottom, left-to-right) per page.
  4. For caption detection: Docling often labels captions as `caption` type. Link them by checking `caption_of` or spatial proximity + heuristics.
  5. The ElementRegistry should not be persisted separately — it can be reconstructed from `DocumentSchema.elements`.
- **Acceptance Criteria:**
  - `normalize_document` returns a fully populated `DocumentSchema` with all elements from the Docling output.
  - Every element has: `element_id`, `page_num`, `bbox`, `reading_order`, `element_type`, `content`.
  - The ElementRegistry provides O(1) lookup by ID and correct iteration in reading order.
  - Proximity information is stored as `nearby` relationships on elements.
  - Captions are correctly linked to their parent (table/figure) elements.
- **Cautionary Points (Risks & Edge Cases):**
  - Docling may not provide reading order explicitly. The spatial fallback must handle multi-column layouts correctly. Consider using centroid-based sorting with column detection.
  - Text items that are split across pages (e.g., a paragraph flowing to the next page) should remain as separate elements with a `follows` relationship rather than being forcibly merged.
  - Some Docling items may be unrecognized types — assign them a generic `ElementSchema` and log a warning.
- **Implementation Suggestions:**
  - Use a visitor pattern: `class DoclingItemVisitor` that dispatches to `visit_text()`, `visit_table()`, `visit_figure()`, etc.
  - Write the normalization as a series of composable transforms (pipeline pattern): `NormalizeItems() → AssignReadingOrder() → LinkCaptions() → ComputeProximity()`.
- **Testing Suggestions:**
  - Test with a minimal Docling output (synthetic dict) for each element type.
  - Test reading order with multi-column layout (two columns of 3 items each — verify left column items come first).
  - Test caption linking: figure with adjacent caption text, verify they are linked.
  - Test proximity: elements on same page, within threshold, get `nearby` relationship.
- **Done When:**
  - `pytest tests/test_normalization.py` passes, and the normalized output for `2502.04644v1.pdf` contains all expected elements with correct metadata.

---

### Sub-Task 7: Reconstruct Document Hierarchy and Assign Section Paths (R10)

- **Status:** Pending
- **Objective:** Analyze the normalized elements to reconstruct the document's hierarchical structure (Document → Page → Section → SubSection → Element), assign section paths (e.g., "1.2.3") to every element, and create `contains` relationships.
- **Related Requirements:** R10 (Hierarchy Reconstruction)
- **Dependencies and Preconditions:** Sub-Task 6 (normalization and element registry). Sub-Task 1 (schemas).
- **In Scope for This Sub-Task:**
  - Create `src/normalization/hierarchy_builder.py`.
  - Implement `build_hierarchy(doc: DocumentSchema, registry: ElementRegistry)` that:
    1. Identifies section headers by element type (`SectionHeader`) or by heuristic (large font, bold, numeric prefix pattern like "1.", "1.1.", "A.", "Appendix").
    2. Builds a section tree: each `Section` has a `section_path` string, `title`, `level` (depth), `parent_section`, and `children_sections`.
    3. Assigns every element to the most recent open section (the section it falls under based on reading order and spatial position).
    4. Generates `contains` relationships: section → child elements, section → subsection, document → section.
    5. Re-orders elements within a section by reading order.
    6. Handles special cases: front matter (title page, abstract, table of contents), back matter (references, appendices), and multi-level numbered sections.
  - Implement `assign_section_paths(doc: DocumentSchema) -> None` that writes the section path string (e.g., "3.2.1") into each element's `section_path` field.
  - Add a `SectionSchema` to `src/schemas/document.py` if not already present.
  - Handle un-sectioned elements (e.g., headers, footers, page numbers) — assign them to a virtual "page-level" section or leave section_path empty.
- **Out of Scope for This Sub-Task:**
  - No chunking (section paths are used during chunking, but chunking itself is later).
  - No relationship generation beyond `contains`.
- **Instructions:**
  1. Use the reading order established in Sub-Task 6.
  2. Section header detection: prefer explicit type from Docling (`heading`). When not available, use heuristics: font size jump, bold, regex for "Chapter 1", "1.", "1.1.", "Section", "Appendix".
  3. Section tree: maintain a stack of open sections. On encountering a section header at level L, pop all sections at depth >= L, then push the new section.
  4. Elements between a section header and the next section header (or end) belong to the current top-of-stack section.
 - **Acceptance Criteria:**
   - The hierarchy builder identifies all section headers in the sample document.
   - Every element (except headers/footers/page numbers) has a non-empty `section_path`.
   - The section paths form a valid tree (parent paths are prefixes of child paths).
   - `contains` relationships are generated for all containment levels.
   - The hierarchy is consistent with the document's table of contents (when available).
- **Cautionary Points (Risks & Edge Cases):**
   - Documents may have un-numbered sections (e.g., "Abstract", "Introduction" without numbers). Assign paths like "0.1", "0.2" or use slugified titles.
   - Floating elements (figures, tables) may appear after the section they belong to (e.g., figure at top of next page). Use anchoring heuristics: if a figure appears within 3 pages after the section where it is referenced, assign it to that section.
   - Page headers/footers and page numbers should not be treated as sections.
- **Implementation Suggestions:**
   - Represent the section tree as a list of `SectionNode` objects with parent pointers.
   - Use regex patterns for numbered section detection: `r'^(\d+(?:\.\d+)*)\s+.*'`
   - Store the section tree in `DocumentSchema.sections` as a flat list with parent references (easier to serialize than recursive tree).
- **Testing Suggestions:**
   - Create synthetic documents (as normalized element lists) with known section structures (flat, deeply nested, mixed numbered/un-numbered).
   - Test with the sample PDF and verify the output section hierarchy matches the actual PDF structure.
- **Done When:**
   - `pytest tests/test_hierarchy.py` passes, and the hierarchy for `2502.04644v1.pdf` accurately reflects its section structure (title, abstract, sections, references, appendix).

---

### Sub-Task 8: Process Tables into Structured Representations (R11)

- **Status:** Pending
- **Objective:** Extract all tables from the normalized document, convert them into multiple structured formats (markdown, HTML, JSON, plain-text summary), compute table metadata, and handle multi-page/spanned tables.
- **Related Requirements:** R11 (Table Processing)
- **Dependencies and Preconditions:** Sub-Task 6 (normalized elements with TableSchema entries). Sub-Task 7 (section paths available for table elements).
- **In Scope for This Sub-Task:**
  - Create `src/normalization/table_processor.py`.
  - Implement `process_table(element: TableSchema, dl_doc: DoclingDocument) -> TableSchema` (enriches the element in-place or returns an updated copy) that:
    1. Extracts table data from Docling's structured table output (cells, rows, columns, headers).
    2. Converts to markdown table format → `element.markdown`.
    3. Converts to HTML table format → `element.html`.
    4. Converts to JSON (list of dicts, each dict is a row) → `element.json`.
    5. Generates a plain-text summary: "Table showing [title/description]: [row_count] rows × [col_count] columns, headers: [col1, col2, ...]" → `element.summary`.
    6. Computes and stores metadata: `row_count`, `col_count`, `headers`, `has_header_row`, `is_spanning` (multi-page).
  - Implement `detect_spanning_tables(elements: List[TableSchema]) -> List[Tuple[int, int, TableSchema]]` that:
    1. Identifies tables that are continued across pages (same table label, adjacent page numbers, identical column structure).
    2. Marks them with `is_spanning = True` and assigns a `span_group_id`.
    3. Optionally, merge the cells of spanned parts into a single logical table.
  - Implement `generate_table_relationship(element: TableSchema, registry: ElementRegistry)` that links tables to their captions (`has_caption`), footnotes (`refers_to`), and nearby text elements (`nearby`, `describes`).
- **Out of Scope for This Sub-Task:**
  - No table data quality assessment (e.g., verifying numerical accuracy).
  - No integration into chunks yet (that is Sub-Task 14-15).
- **Instructions:**
  1. Docling's `TableItem` includes a `data` attribute (a `pandas.DataFrame` or similar). Use that as the source of truth.
  2. For markdown: join cells with `|`, join rows with newlines, add a header separator row.
  3. For HTML: use `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<td>`, `<th>`.
  4. For JSON: use `[{"col1": val, "col2": val, ...}, ...]`.
  5. The summary text should be brief (2-3 sentences max).
  6. Span detection: check if two tables on consecutive pages have identical column count and similar column headers. If yes, treat as span.
- **Acceptance Criteria:**
   - Every `TableSchema` element has populated `markdown`, `html`, `json`, `summary`, and metadata fields.
   - Multi-page spanned tables are detected and linked (if any exist in sample docs).
   - Tables are linked to their captions and nearby text.
   - The summary text is human-readable and informative.
- **Cautionary Points (Risks & Edge Cases):**
   - Tables with merged cells (colspan/rowspan) — markdown representation cannot fully represent these. HTML and JSON can. The markdown should use a best-effort approximation.
   - Empty tables: Docling might return 0-row tables. Handle without error.
   - Very wide tables (many columns): the markdown may be very wide but that's acceptable.
- **Implementation Suggestions:**
   - Use `tabulate` library for markdown conversion if available, otherwise manual.
   - For JSON, use pandas DataFrame's `.to_dict(orient='records')`.
- **Testing Suggestions:**
   - Create a synthetic table in a PDF using `reportlab`, process it, and verify all output formats are correct.
   - Test multi-page span detection by creating two tables on consecutive pages with identical structure.
   - Test edge cases: 1-row table, 1-column table, table with merged cells, empty table.
- **Done When:**
   - `pytest tests/test_table_processor.py` passes, and all tables in `2502.04644v1.pdf` are processed into all formats with correct metadata.

---

### Sub-Task 9: Process Images/Charts/Graphs — Asset Saving, Classification, Metadata (R12)

- **Status:** Pending
- **Objective:** Extract images, charts, and graphs from Docling output, save them as dedicated assets, classify their visual type, and prepare metadata (with extensibility for future vision-language descriptions).
- **Related Requirements:** R12 (Image/Chart/Graph Processing)
- **Dependencies and Preconditions:** Sub-Task 6 (normalized elements with ImageSchema/ChartSchema/GraphSchema). Sub-Task 4 (page and table images already saved by Docling).
- **In Scope for This Sub-Task:**
  - Create `src/metadata/image_processor.py`.
  - Implement `save_visual_asset(element: ImageSchema | ChartSchema | GraphSchema, dl_doc: DoclingDocument, doc: DocumentSchema)` that:
    1. Extracts the image bitmap from Docling (if embedded) or renders from the PDF page using the bounding box.
    2. Saves as PNG to `data/raw/{doc_id}/assets/{element_id}.png`.
    3. Stores the asset file path in `element.asset_path`.
    4. Saves a thumbnail (max 256px) to `data/raw/{doc_id}/assets/thumb_{element_id}.png`.
  - Implement `classify_visual_type(element, image_data) -> str` that:
    1. Uses simple heuristics or a lightweight classifier (optional, skip if no model) to categorize: `chart`, `graph`, `diagram`, `photograph`, `illustration`, `logo`, `screenshot`, `other`.
    2. Stores the classification in `element.visual_type`.
    3. If classification is not possible (no model loaded), label as `"unclassified"` and log a debug message.
  - Implement `prepare_visual_metadata(element, doc)` that:
    1. Computes basic image properties: width, height (pixels), aspect ratio, file size, color mode.
    2. Links to caption: if Docling provides a caption, set `has_caption` relationship.
    3. Links to nearby text elements (`nearby`, `describes`) for elements within the same page within a configurable distance.
    4. Prepares an extensibility `vision_description` field (initially `None`) for future Vision-Language Model (VLM) integration.
  - Update `ChartSchema` and `GraphSchema` with appropriate fields: `visual_type`, `asset_path`, `thumbnail_path`, `image_properties: dict`, `vision_description: Optional[str]`.
- **Out of Scope for This Sub-Task:**
  - No actual Vision-Language Model inference. The `vision_description` field is reserved but not populated.
  - No chart data extraction (e.g., reading values from chart plots — that would be a future enhancement).
- **Instructions:**
  1. Image extraction from PDF: use PyMuPDF (`fitz`) or `pdf2image` to render the page region given by the bounding box. Docling may already provide the image bytes — prefer that.
  2. Classification: if `doclayout-yolo` is installed and has a visual classifier, use it. Otherwise, use a simple rule based on aspect ratio, size, and position (charts are often full-width, logos are small and at corners).
  3. Thumbnail: use Pillow's `Image.thumbnail()`.
  4. Asset paths should use the `element_id` (UUID) as filename for uniqueness.
- **Acceptance Criteria:**
   - Every visual element has a saved asset file (PNG) and thumbnail.
   - `element.asset_path` and `element.thumbnail_path` are valid paths.
   - Visual type is classified (or set to `"unclassified"`).
   - Image properties are populated.
   - Caption and nearby relationships are established.
   - The `vision_description` field exists and is `None` (not populated yet).
- **Cautionary Points (Risks & Edge Cases):**
   - Embedded images in PDF may have different resolution than page rendering. Use the higher-quality source available.
   - Some elements may be marked as figures but are actually decorative (borders, lines). The classification step should attempt to detect these and exclude them or mark as `"decorative"`.
   - Memory: avoid loading all page images simultaneously; process elements page-by-page.
- **Implementation Suggestions:**
   - Use a `VisualAssetManager` class that handles saving and deduplication (if same image appears on multiple pages, save once).
   - For the classifier stub, use a simple function that can be replaced with a model in the future.
- **Testing Suggestions:**
   - Create a PDF with embedded images, charts (from matplotlib), and photographs. Run processing and verify assets are saved correctly.
   - Verify that classification heuristics label known chart images correctly.
   - Test with a PDF that has no images — no errors, empty results.
- **Done When:**
   - `pytest tests/test_image_processor.py` passes, and visual elements from `2502.04644v1.pdf` are correctly processed.

---

### Sub-Task 10: Process Formulas and Link to Explanatory Text (R13)

- **Status:** Pending
- **Objective:** Extract formulas from Docling output, classify them (inline vs. display), convert to LaTeX/symbolic representation, and link to surrounding explanatory text.
- **Related Requirements:** R13 (Formula Processing)
- **Dependencies and Preconditions:** Sub-Task 6 (normalized elements with FormulaSchema). Docling must extract formula items.
- **In Scope for This Sub-Task:**
  - Create `src/normalization/formula_processor.py`.
  - Implement `process_formula(element: FormulaSchema, dl_doc: DoclingDocument) -> FormulaSchema` that:
    1. Extracts the raw formula text from Docling (may be in LaTeX, MathML, or plain text).
    2. Converts to LaTeX if not already in that format (use `pylatexenc` or similar).
    3. Stores LaTeX in `element.latex`.
    4. Stores a plain-text approximation (e.g., removing LaTeX commands) in `element.text_approximation`.
    5. Classifies as `inline` or `display` based on Docling's classification or layout heuristics (whether the formula is in its own block or embedded in a text line).
    6. Extracts inline formulas embedded within text elements and splits them out (or links them to the parent text).
  - Implement `link_formula_to_text(element: FormulaSchema, registry: ElementRegistry)` that:
    1. Finds the nearest paragraph element (same page, reading order) that discusses the formula.
    2. Creates `explains` relationship from text → formula.
    3. If the formula is inline, creates `contains` from paragraph → formula.
  - Implement `link_formula_to_notation(element: FormulaSchema)` that:
    1. Extracts variable mentions from the formula (e.g., `x`, `\theta`, `\Sigma`).
    2. Stores a `variables: List[str]` list on the formula element.
    3. Links to other elements that define those variables.
- **Out of Scope for This Sub-Task:**
   - No symbolic computation or CAS integration.
   - No rendering of formulas as images (could be future enhancement).
- **Instructions:**
  1. Docling may not extract formulas in all documents. For documents without formulas, skip processing gracefully.
  2. LaTeX conversion: if Docling provides `latex` field, use it directly. If it provides `mathml`, convert using `latexcodec` or `pylatexenc`.
  3. Inline formula detection: check if the formula's bbox is within the bbox of a text block, or if Docling labels it as `inline`.
- **Acceptance Criteria:**
   - Every `FormulaSchema` has populated `latex` and `text_approximation` fields.
   - Inline vs. display classification is correct.
   - Formulas are linked to their surrounding text via `explains` relationships.
   - Variable mentions are extracted (when feasible).
- **Cautionary Points (Risks & Edge Cases):**
   - Formula extraction quality varies widely between PDFs. The processor must gracefully handle missing or malformed LaTeX.
   - Some formulas may be images (not extracted by Docling). These will be in the image pipeline instead.
   - Multi-line equations: treat as display formulas, keep them as a single formula element.
- **Implementation Suggestions:**
   - Use a regex-based extraction for variable names: `r'\\([a-zA-Z]+|\\[a-zA-Z]+'` to find LaTeX commands and single-letter variables.
   - For linking, use reading-order proximity: the text element immediately preceding a display formula usually discusses it.
- **Testing Suggestions:**
   - Create a PDF with known LaTeX formulas (use `reportlab` with Platypus or LaTeX-generated PDF).
   - Test inline formula detection, display formula detection, LaTeX storage, and text linking.
   - Test with a document without formulas — no errors.
- **Done When:**
   - `pytest tests/test_formula_processor.py` passes.

---

### Sub-Task 11: Define and Generate Relationship Metadata (R14)

- **Status:** Pending
- **Objective:** Implement the full relationship generation engine that creates all typed relationships between elements as defined in the Core Concepts table.
- **Related Requirements:** R14 (Relationship Metadata)
- **Dependencies and Preconditions:** Sub-Tasks 6-10 (all element types processed, captions linked, hierarchy built, spatial proximity computed). Sub-Task 1 (RelationshipSchema).
- **In Scope for This Sub-Task:**
  - Create `src/metadata/relationship_generator.py`.
  - Implement `generate_all_relationships(doc: DocumentSchema, registry: ElementRegistry) -> List[RelationshipSchema]` that composes:
    1. **Structural relationships** (from hierarchy builder): `contains`, `belongs_to`.
    2. **Sequential relationships** (from reading order): `follows`, `precedes` between consecutive elements on the same page and consecutive pages.
    3. **Caption relationships** (from table & image processors): `has_caption`.
    4. **Spatial relationships** (from proximity): `nearby`.
    5. **Section relationships** (from hierarchy builder): `same_section_as` (all elements sharing a section path get pairwise or group relationships — use a group edge rather than fully connected to avoid blowup).
    6. **Reference relationships** (`refers_to`): scan text elements for patterns like "see Table X", "as shown in Figure Y", "Equation Z", and link to the corresponding element using caption/heading text matching.
    7. **Descriptive relationships** (`describes`): text elements that precede or follow a figure/table/chart and share the same section are likely describing it.
    8. **Formula explanation relationships** (from formula processor): `explains`.
  - Implement `deduplicate_relationships(rels: List[RelationshipSchema]) -> List[RelationshipSchema]` that:
    1. Removes exact duplicates (same source, target, type).
    2. Removes self-references (source == target).
    3. Resolves symmetric relationships: `nearby` and `same_section_as` should be stored once but treated as undirected.
  - Implement `generate_relationship_summary(rels: List[RelationshipSchema]) -> dict` that counts relationships by type.
- **Out of Scope for This Sub-Task:**
  - No graph database loading (that is future work).
  - No external LLM calls for relationship inference (relationships are deterministic based on structure/spatial/text-pattern).
- **Instructions:**
  1. For `refers_to`: compile a registry of all tables, figures, and formulas with their captions/labels. Scan each text element for patterns like `r'(?:Table|Figure|Fig\.|Equation|Eq\.)\s*(\d+(?:\.\d+)*)'` and look up matching labeled elements.
  2. For `describes`: assign text elements that are within N positions of a visual element in reading order (N=2) and share the same section as describing it.
  3. For `same_section_as`: store one relationship per section path (a group identifier) rather than O(n^2) pairwise edges.
  4. Ensure all relationships have stable relationship IDs (UUIDv5).
- **Acceptance Criteria:**
   - All relationship types from the Core Concepts table are generated.
   - `refers_to` correctly matches "see Table 1" patterns to actual table elements.
   - `describes` links paragraphs to adjacent figures.
   - No duplicate or self-referencing relationships.
   - `same_section_as` is stored efficiently (group-based, not pairwise).
   - The relationship generator produces deterministic results (same input → same output).
- **Cautionary Points (Risks & Edge Cases):**
   - Pattern matching for `refers_to` may produce false positives (e.g., "see Table for settings" when "Table" is not a numbered reference). Mitigate by requiring a number after the keyword, or by checking if an element with matching label exists.
   - In very large documents, the number of `same_section_as` relationships could be large if stored pairwise. Use a single relationship with a group id and a list of members instead.
- **Implementation Suggestions:**
   - Use `re` for text pattern matching.
   - For `same_section_as`, use `RelationshipSchema(group_id=str, member_ids=List[UUID], relationship_type="same_section_as")`.
   - Write each relationship type as a separate generator function, composed by `generate_all_relationships`.
- **Testing Suggestions:**
   - Create a mock document with known elements (text, table, figure, formula) and known reading order. Verify all expected relationships are generated.
   - Test `refers_to` matching with various patterns: "see Table 1", "as shown in Figure 2.1", "Equation 5", "Fig. 3a".
   - Test deduplication: add duplicate relationships to input, verify output has no duplicates.
- **Done When:**
   - `pytest tests/test_relationships.py` passes, and the relationship output for `2502.04644v1.pdf` is plausible (visual inspection of counts and examples).

---

### Sub-Task 12: Implement Hierarchical Chunking (R15 — part 1)

- **Status:** Pending
- **Objective:** Implement hierarchical chunking that creates chunks based on document section boundaries. Each section produces one or more chunks, preserving section context and element references.
- **Related Requirements:** R15 (Chunking)
- **Dependencies and Preconditions:** Sub-Task 7 (hierarchy/section paths). Sub-Tasks 8-10 (tables, images, formulas processed). Sub-Task 11 (relationships available). Sub-Task 1 (ChunkSchema).
- **In Scope for This Sub-Task:**
  - Create `src/chunking/__init__.py`, `src/chunking/hierarchical_chunker.py`, and `src/chunking/base.py`.
  - Create `BaseChunker` abstract class in `base.py` with:
    - `chunk(doc: DocumentSchema, registry: ElementRegistry) -> List[ChunkSchema]`
    - Common utilities: `_build_chunk_content()`, `_assign_relationships()`.
  - Implement `HierarchicalChunker(BaseChunker)`:
    1. For each section in the document's hierarchy, collect all elements belonging to that section (in reading order).
    2. If the section's total text length / token count exceeds the configured chunk token limit (default 512), split the elements within the section by reading order into multiple sub-chunks, each keeping the same `section_path`.
    3. For each chunk, generate `content` by concatenating element contents with element references: `"{{ELEM:uuid}}"` markers embedded inline.
    4. Store `element_refs: List[UUID]` — all element UUIDs whose content is included in this chunk.
    5. Store `chunk_type = "hierarchical"`.
    6. Store `section_path` and `page_range` (min page to max page across elements).
    7. Attach relevant relationships: if all elements in the chunk share a `same_section_as` relationship, include it; otherwise carry over all relationships whose source or target is in `element_refs`.
    8. Ensure a table and its caption are always in the same chunk (never split across chunks).
  - Implement `_enforce_integrity_constraints(elements, chunk_candidates)` that prevents splitting:
    - A table from its caption.
    - A figure from its caption.
    - A formula from its immediately preceding explanatory text.
- **Out of Scope for This Sub-Task:**
  - Semantic chunking and cluster-semantic chunking (Sub-Tasks 13, 14).
  - No chunk-level embedding generation (that is future/vector stage).
- **Instructions:**
  1. Token counting: use `tiktoken` with `cl100k_base` encoding or Docling's tokenizer to count tokens.
  2. Content building: for text elements, use their text content. For tables, use the summary text with a marker `{{TABLE:uuid}}`. For images, use `{{IMAGE:uuid}}` with alt text or caption. For formulas, use `{{FORMULA:uuid}}` with LaTeX.
  3. The `{{ELEM:uuid}}` markers serve as references; downstream consumers can use them to look up full structured data.
- **Acceptance Criteria:**
   - Every section with content produces at least one chunk.
   - Chunks respect the token limit (configurable).
   - A table and its caption are always in the same chunk.
   - A figure and its caption are always in the same chunk.
   - Each chunk has valid `element_refs`, `section_path`, and `page_range`.
   - The content includes element reference markers.
- **Cautionary Points (Risks & Edge Cases):**
   - A very long section (e.g., 5000 tokens) is split into multiple chunks. Each chunk should still have a coherent reading order (no reordering).
   - A single element may be larger than the chunk limit (e.g., a huge table). In that case, the element becomes a chunk by itself (or the chunk limit is temporarily stretched).
   - Empty sections (only a heading, no body text) should produce no chunk, or a very minimal chunk.
- **Implementation Suggestions:**
   - Implement `_merge_elements_into_chunks(elements, token_limit)` that greedily groups elements until the token budget is reached, respecting integrity constraints.
   - Store the token limit in `PipelineSettings.chunk_token_limit`.
- **Testing Suggestions:**
   - Create a mock document with sections of varying lengths. Test that small sections fit in one chunk, large sections are split.
   - Test integrity: create a section with a table on page 1 and its caption on page 2; verify they end up in the same chunk.
   - Test content format: verify `{{ELEM:...}}` markers are present.
- **Done When:**
   - `pytest tests/test_hierarchical_chunker.py` passes.

---

### Sub-Task 13: Implement Semantic Chunking (R15 — part 2)

- **Status:** Pending
- **Objective:** Implement semantic chunking that uses embedding similarity to group elements into semantically coherent chunks, independent of section boundaries.
- **Related Requirements:** R15 (Chunking)
- **Dependencies and Preconditions:** Sub-Task 12 (base chunker class, chunk schema). `sentence-transformers` installed.
- **In Scope for This Sub-Task:**
  - Create `src/chunking/semantic_chunker.py`.
  - Implement `SemanticChunker(BaseChunker)`:
    1. For each page (or for the whole document), embed each text element using the configured sentence-transformer model (default: `BAAI/bge-m3`).
    2. Compute cosine similarity between adjacent elements in reading order.
    3. When the similarity between adjacent elements drops below a threshold (configurable, default 0.6), that point becomes a chunk boundary.
    4. Each chunk gets `chunk_type = "semantic"`.
    5. Enforce the same integrity constraints as hierarchical chunking (tables + captions, etc.).
    6. If a semantic chunk would exceed the token limit, further split by similarity within the chunk (recursive splitting).
  - Implement `compute_similarity_matrix(elements: List[ElementSchema], model) -> np.ndarray` that computes pairwise cosine similarities for all elements on a page (optional optimization).
  - Implement `find_semantic_boundaries(similarities, threshold) -> List[int]` that returns indices where a boundary should be inserted.
- **Out of Scope for This Sub-Task:**
  - No cluster-semantic grouping (Sub-Task 14).
- **Instructions:**
  1. Use `sentence-transformers` `SentenceTransformer` for embeddings.
  2. Cache embeddings per element to avoid recomputation (embed once, use for both semantic chunking and future cluster-semantic).
  3. For non-text elements (tables, images, formulas), generate a text representation (summary, caption, LaTeX) before embedding.
- **Acceptance Criteria:**
   - Chunks are created at points of low semantic similarity between adjacent elements.
   - Chunk boundaries align with topic shifts (e.g., end of one paragraph, beginning of a different topic).
   - Token limit is respected; integrity constraints are enforced.
   - The threshold is configurable and affects chunk granularity.
- **Cautionary Points (Risks & Edge Cases):**
   - Documents with uniform semantic density (e.g., continuous prose) may produce few boundaries. Handle this by falling back to token-count-based splitting when similarity variance is low.
   - Very short elements (single line, page number) may have noisy embeddings. Consider filtering out very short elements (< 5 characters) before similarity computation.
- **Implementation Suggestions:**
   - Run similarity-based chunking per-page first, then merge across pages if the last chunk of page N and first of page N+1 are similar.
   - Store embeddings temporarily in `EmbeddingCache` (dict keyed by element_id) to reuse in Sub-Task 14.
- **Testing Suggestions:**
   - Create a document with clear topic shifts (paragraph about Q1 earnings → paragraph about R&D investment). Verify chunk boundaries occur at topic shifts.
   - Test with a document where all text is on one topic — verify few or no splits.
   - Test threshold sensitivity: higher threshold → more chunks, lower → fewer chunks.
- **Done When:**
   - `pytest tests/test_semantic_chunker.py` passes.

---

### Sub-Task 14: Implement Cluster-Semantic Grouping (R15 — part 3)

- **Status:** Pending
- **Objective:** Implement cluster-semantic grouping that uses global embedding clustering to group related elements across section and page boundaries into thematically coherent chunks.
- **Related Requirements:** R15 (Chunking)
- **Dependencies and Preconditions:** Sub-Task 13 (embeddings, semantic chunker utilities). Sub-Task 12 (base chunker class).
- **In Scope for This Sub-Task:**
  - Create `src/chunking/cluster_chunker.py`.
  - Implement `ClusterSemanticChunker(BaseChunker)`:
    1. Embed all elements in the document using the configured model (reuse cache from Sub-Task 13).
    2. Apply clustering (e.g., HDBSCAN or KMeans with optimal k via silhouette score) to group elements into thematic clusters.
    3. For each cluster, organize elements by reading order within the cluster.
    4. Apply the token limit constraint: if a cluster exceeds the limit, split into sub-chunks by reading order within the cluster.
    5. Each chunk gets `chunk_type = "cluster"`.
    6. Enforce integrity constraints (tables + captions, etc.).
    7. Assign a descriptive cluster label (optional) based on most frequent n-grams or TF-IDF of the cluster's text content.
  - Implement `_determine_cluster_count(elements, model)` that uses silhouette analysis or the elbow method to pick an appropriate number of clusters.
  - Implement `_label_cluster(elements: List[ElementSchema]) -> str` that generates a short human-readable label (e.g., "Financial Performance", "Experimental Setup").
- **Out of Scope for This Sub-Task:**
   - No persistent storage of clusters (they are transient for chunk generation).
- **Instructions:**
  1. Use `sklearn.cluster` for KMeans or `hdbscan` for HDBSCAN. HDBSCAN is preferred as it handles noise points (elements that don't fit a cluster).
  2. Noise elements (cluster = -1 in HDBSCAN) should fall back to hierarchical chunking behavior (group with nearest cluster or form own chunk).
  3. For very small documents (< 20 elements), skip clustering and fall back to semantic or hierarchical chunking.
- **Acceptance Criteria:**
   - Elements from different sections but similar topics end up in the same cluster/chunk.
   - Element ordering within a cluster respects reading order.
   - Token limit and integrity constraints are enforced.
   - Noise elements are handled gracefully.
   - Descriptive cluster labels are generated (even if generic).
- **Cautionary Points (Risks & Edge Cases):**
   - Clustering on a single document may not produce meaningful clusters for short or homogeneous documents. Fall back gracefully.
   - HDBSCAN can be slow on very large numbers of elements (>5000). Consider subsampling or mini-batch KMeans.
   - Clusters may mix element types (text + table + figure). This is intentional and desirable.
- **Implementation Suggestions:**
   - Use `HDBSCAN(min_cluster_size=5, min_samples=1)` as a starting point.
   - For cluster labeling, extract top 5 TF-IDF terms from each cluster and combine (e.g., "budget, revenue, forecast" → "Budget & Revenue").
- **Testing Suggestions:**
   - Create a document with two distinct topics (e.g., first half about finance, second half about technology). Verify that cluster chunks cleanly separate the topics.
   - Test with a short document — verify fallback to hierarchical/semantic chunking.
   - Test with noise elements — verify they don't prevent clustering of main topics.
- **Done When:**
   - `pytest tests/test_cluster_chunker.py` passes.

---

### Sub-Task 15: Integrate Tables, Images, Charts, and Formulas into Chunks (R15 — part 4)

- **Status:** Pending
- **Objective:** Ensure all chunking strategies correctly integrate non-text elements (tables, images, charts, formulas) into chunks with references to structured data and summaries, without flattening everything to plain text.
- **Related Requirements:** R15 (Chunking — integration)
- **Dependencies and Preconditions:** Sub-Tasks 12-14 (all chunkers implemented). Sub-Tasks 8-10 (table, image, formula processing complete).
- **In Scope for This Sub-Task:**
  - Update all three chunkers to use a unified `_element_to_chunk_content(element: ElementSchema) -> str` method that produces the content string for any element type:
    - `TextBlock` / `SectionHeader` / `Footnote` / `Header` / `Footer` / `ListBlock`: plain text.
    - `TableSchema`: `"[Table: {summary}] {{TABLE:{element_id}}}` where `summary` is the plain-text summary from Sub-Task 8.
    - `ImageSchema` / `ChartSchema` / `GraphSchema`: `"[Image/Chart/Graph: {caption or 'Untitled'}] {{IMAGE:{element_id}}}"`.
    - `FormulaSchema`: `"[Formula: {text_approximation}] {{FORMULA:{element_id}}}"`.
    - `CaptionSchema`: plain text (but mark as `#caption` metadata on the referenced element's chunk entry).
  - Ensure that chunk content is human-readable even without resolving the structured references.
  - Ensure that `element_refs` includes the UUID of every element whose content appears in the chunk.
  - Add a `ChunkElementRef` metadata on each chunk: a list of `{"element_id": UUID, "role": "text"|"table"|"image"|"chart"|"formula"|"caption", "summary": str}` for richer downstream use.
- **Out of Scope for This Sub-Task:**
   - No chunk-level embedding generation.
   - No merging of chunking strategies (that would be a future ensembling step).
- **Instructions:**
  1. Modify the `BaseChunker` to use the unified element-to-content converter.
  2. The reference markers (`{{TABLE:uuid}}`, `{{IMAGE:uuid}}`, `{{FORMULA:uuid}}`) allow downstream stages to retrieve the full structured data from the elements registry or data store.
  3. The plain-text summary in brackets provides immediate readability without resolution.
- **Acceptance Criteria:**
   - All three chunkers produce chunks with the correct reference markers.
   - Every chunk's `element_refs` contains the UUID of every element it references.
   - Every chunk includes `ChunkElementRef` metadata.
   - A chunk's content is readable (e.g., "[Table: 10 rows × 3 columns showing quarterly revenue] {{TABLE:abc-123}}").
- **Cautionary Points (Risks & Edge Cases):**
   - An element with no content (e.g., empty image with no caption) should still produce a minimal reference.
   - Ensure no duplicated references if the same element appears in multiple chunks (each chunk independently lists its own refs).
- **Implementation Suggestions:**
   - Create a `content_renderer.py` module in `chunking/` with the `element_to_chunk_content` function.
   - Use a registry pattern: `ContentRenderer.register(ElementSchema, render_func)` for extensibility.
- **Testing Suggestions:**
   - Unit test `element_to_chunk_content` for each element type.
   - Integration test with a mock document containing all element types — verify each chunker produces correct content with all reference types.
- **Done When:**
   - All chunker tests pass with the integrated content rendering.

---

### Sub-Task 16: Export Final Artifacts (R16)

- **Status:** Pending
- **Objective:** Export all processed data (chunks, elements, relationships, metadata) into standardized file formats (JSONL, Parquet, JSON metadata, human-readable reports).
- **Related Requirements:** R16 (Exports)
- **Dependencies and Preconditions:** Sub-Tasks 12-15 (chunking complete). Sub-Task 11 (relationships complete). Sub-Task 6 (elements normalized).
- **In Scope for This Sub-Task:**
  - Create `src/utils/exporter.py`.
  - Implement `export_chunks(chunks: List[ChunkSchema], output_dir: Path)`:
    - Writes `chunks.jsonl` (one JSON object per line, each a serialized ChunkSchema).
    - Writes `chunks.parquet` (same data in Parquet format with schema inferred from ChunkSchema).
  - Implement `export_elements(elements: List[ElementSchema], output_dir: Path)`:
    - Writes `elements.jsonl`.
    - Writes `elements.parquet`.
  - Implement `export_relationships(relationships: List[RelationshipSchema], output_dir: Path)`:
    - Writes `relationships.jsonl` (JSONL).
    - Writes `relationships.parquet`.
  - Implement `export_metadata(doc: DocumentSchema, output_dir: Path)`:
    - Writes `metadata.json` (a single JSON object with document-level stats: page count, element count by type, chunk count by type, relationship count by type, processing timestamps, versions).
  - Implement `export_review_report(doc: DocumentSchema, output_dir: Path) -> str`:
    - Generates a human-readable HTML (or Markdown) report that includes:
      - Document summary (title, pages, elements, chunks).
      - Per-page element map (element type, reading order, section).
      - Sample chunks with content preview.
      - Relationship summary.
      - Processing metrics (time per stage).
    - Uses Jinja2 or simple f-string templates to render the report.
  - Organize output directory as:
    ```
    outputs/{doc_id}/
    ├── chunks.jsonl
    ├── chunks.parquet
    ├── elements.jsonl
    ├── elements.parquet
    ├── relationships.jsonl
    ├── relationships.parquet
    ├── metadata.json
    └── review_report.html
    ```
- **Out of Scope for This Sub-Task:**
   - No database loading.
   - No API endpoints.
- **Instructions:**
  1. Use `orjson` for fast JSON serialization (or `json` with `default=str` for UUID/datetime handling).
  2. For Parquet, use `pyarrow` or `pandas.DataFrame.to_parquet()`.
  3. The review report should be self-contained (no external CSS/JS) for easy sharing. Use inline styles or a Markdown file.
  4. All exports should be atomic: write to a temp file, then rename.
- **Acceptance Criteria:**
   - Running `export_all(doc, output_dir)` produces all expected files.
   - JSONL files are valid (one JSON object per line, each line parseable).
   - Parquet files can be read back with `pyarrow.parquet.read_table()`.
   - The review report opens in a browser and displays document structure clearly.
   - The metadata JSON contains accurate counts.
- **Cautionary Points (Risks & Edge Cases):**
   - UUID serialization: ensure `uuid.UUID` objects are serialized to strings.
   - `datetime` serialization: use ISO 8601 format.
   - Very large outputs: JSONL and Parquet handle large datasets well, but memory-mapped writing is preferred for parquet.
- **Implementation Suggestions:**
   - Create an `ExportManifest` class that tracks what was exported, when, and with which pipeline version.
   - Add a `--export-format` option in the future, but default to both JSONL and Parquet.
- **Testing Suggestions:**
   - Create a mock document with 5 elements, 3 chunks, 10 relationships. Export and verify:
     - File existence.
     - File content correctness (read back and compare).
     - Parquet round-trip: write then read, compare data.
   - Test review report generation: verify the HTML contains the document title, page count, etc.
- **Done When:**
   - `pytest tests/test_exporter.py` passes, and manual review of generated reports confirms correctness.

---

### Sub-Task 17: Add Validation / Evaluation Checks and Define Sample Benchmark Documents (R17)

- **Status:** Pending
- **Objective:** Create automated validation checks for pipeline output quality and define a set of sample benchmark documents for regression testing.
- **Related Requirements:** R17 (Validation & Benchmarks)
- **Dependencies and Preconditions:** All previous sub-tasks (pipeline is functional).
- **In Scope for This Sub-Task:**
  - Create `tests/benchmark/` directory.
  - Create `tests/benchmark/sample_docs/` with at least 3 PDFs:
    1. A simple document: 3-5 pages, plain text, one table, one image, no formulas.
    2. A moderately complex document: 10-15 pages, multiple sections, several tables and figures, numbered equations.
    3. A complex document: 20+ pages, multi-column layout, multi-page spanned tables, charts, footnotes, headers/footers, appendix.
    - (Note: The existing `2502.04644v1.pdf` can serve as one of these if suitable.)
  - Create `src/validation/pipeline_validator.py` with:
    - `PipelineValidationResult` model with per-stage success/failure flags.
    - `validate_full_pipeline(doc: DocumentSchema) -> PipelineValidationResult` that runs end-to-end validation:
      - **Coverage:** Every element in the normalized document appears in at least one chunk's `element_refs`.
      - **Completeness:** All elements have required fields populated.
      - **Relationship coverage:** Every visual element (table, image, chart, formula) has at least one relationship (caption, nearby, describes).
      - **Chunk quality checks:**
        - No chunk exceeds 1.5x the configured token limit (some tolerance allowed for single-element chunks).
        - Each chunk has at least one element_ref.
        - No two chunks are identical in content.
      - **ID uniqueness:** No duplicate element IDs or chunk IDs.
  - Create `src/validation/benchmark_runner.py` that:
    - Runs the full pipeline on all benchmark documents.
    - Collects timing and quality metrics.
    - Writes a benchmark report to `outputs/benchmark_results.json`.
  - Create `tests/test_pipeline_integration.py` with at least 3 integration tests that run the full pipeline (or partial, with mocking) and verify the outputs.
- **Out of Scope for This Sub-Task:**
   - No human evaluation of chunk quality beyond automated checks.
   - No retrieval evaluation (NDCG, recall, etc. — that is future work).
- **Instructions:**
  1. For benchmark documents: if the project does not have suitable PDFs, create them programmatically using `reportlab` (simple), or download open-access papers (complex).
  2. For `validate_full_pipeline`: each check should be independent and produce a pass/fail/warning.
  3. For `benchmark_runner`: use `time.perf_counter` for stage-level timing.
- **Acceptance Criteria:**
   - All benchmark documents pass the integrated pipeline (may have warnings, no crashes).
   - Validation checks catch known issues (e.g., element not in any chunk, duplicate IDs).
   - Benchmark report is generated with timing and quality metrics.
   - Integration tests run in CI (no external dependencies except Docling models).
- **Cautionary Points (Risks & Edge Cases):**
   - Creating real-looking benchmark PDFs with `reportlab` is time-consuming. Use open-access papers from arXiv (which are CC-licensed) as sample complex documents.
   - Benchmark documents must be small enough for CI (avoid >50MB PDFs).
- **Implementation Suggestions:**
   - For `validate_full_pipeline`, implement as a series of composable validators (similar to the Docling validator pattern).
   - Store benchmark results as JSON to track regressions over time.
- **Testing Suggestions:**
   - Run `validate_full_pipeline` on a known-good output → all checks pass.
   - Introduce an intentional error (e.g., remove an element from all chunks) → that specific check fails.
   - Run `benchmark_runner` on all 3 docs and verify output file is produced.
- **Done When:**
   - `pytest tests/test_pipeline_integration.py` passes, `benchmark_runner` produces outputs for all sample docs, and manual inspection of validation results confirms correctness.

---

### Sub-Task 18: Integrate Full Pipeline Orchestrator

- **Status:** Pending
- **Objective:** Create a pipeline orchestrator that chains all stages from ingestion through export into a single callable entry point.
- **Related Requirements:** R1-R17 (all requirements, integration)
- **Dependencies and Preconditions:** All sub-tasks 1-17 complete.
- **In Scope for This Sub-Task:**
  - Create `src/pipeline.py` with:
    ```python
    class ExtractionPipeline:
        def __init__(self, settings: PipelineSettings):
            ...
        def run(self, source: str | Path) -> DocumentSchema:
            # 1. Ingest
            # 2. Extract with Docling
            # 3. Validate Docling output
            # 4. Normalize
            # 5. Build hierarchy
            # 6. Process tables
            # 7. Process images
            # 8. Process formulas
            # 9. Generate relationships
            # 10. Chunk (hierarchical)
            # 11. Chunk (semantic)
            # 12. Chunk (cluster)
            # 13. Export all
            # 14. Validate pipeline output
            # 15. Return fully-populated DocumentSchema
    ```
  - Add command-line entry point via `console_scripts` in `pyproject.toml`:
    ```bash
    edr-pipeline --source path/to/doc.pdf --output ./outputs
    ```
  - Implement `run_multiple(directory: str) -> List[DocumentSchema]` for batch processing.
  - Add progress logging via the structured logger at each stage.
  - Add checkpoints: if a stage output exists and is valid, skip re-execution (idempotent).
- **Out of Scope for This Sub-Task:**
   - No FastAPI server (that is future).
   - No async processing (single-threaded is fine for this stage).
- **Instructions:**
  1. Use the `PipelineSettings` from `utils/config.py`.
  2. Each stage should be a method on `ExtractionPipeline` that takes the document and returns it updated.
  3. Log duration per stage.
  4. The orchestrator should handle partial failures: if chunking fails, the document with extraction/normalization outputs should still be exportable (with errors noted).
- **Acceptance Criteria:**
   - `edr-pipeline --source 2502.04644v1.pdf --output ./outputs` runs the full pipeline end-to-end.
   - All output files are produced in the specified output directory.
   - Running the pipeline twice on the same input is idempotent (skips completed stages).
   - A pipeline with a deliberate mid-stage error produces partial outputs and clear error messages.
- **Cautionary Points (Risks & Edge Cases):**
   - The pipeline may take several minutes for complex documents. Add a `--verbose` flag for detailed progress and `--dry-run` for validation.
   - Ensure temporary files are cleaned up on failure.
- **Implementation Suggestions:**
   - Use `click` or `argparse` for the CLI if needed, but keep it simple (argparse is sufficient).
   - Use a `PipelineContext` dataclass to carry state (current doc, settings, timers, errors) through stages.
- **Testing Suggestions:**
   - Integration test: run the pipeline on the simple benchmark document and verify the full output directory structure.
   - Test idempotency: run twice, compare output checksums (should be identical).
   - Test with non-existent file → clear error.
- **Done When:**
   - `pytest tests/test_pipeline.py` passes, and `edr-pipeline --help` works with the CLI.

---

## Final Integration & Verification

- **System-Wide Test:** Run `edr-pipeline` on all three benchmark documents (simple, moderate, complex). Verify:
  - Output directory contains all expected files (chunks, elements, relationships, metadata, review report).
  - All files are valid (JSONL lines parse, Parquet reads back).
  - The review report is visually meaningful.
  - The pipeline completes without errors on all three documents.
- **Completion Checklist:**
  - [ ] All Pydantic schemas defined and tested (`pytest tests/test_schemas.py`).
  - [ ] Project structure created, installable via `pip install -e .`.
  - [ ] Ingestion copies PDFs immutably with metadata manifest.
  - [ ] Docling extraction works and persists all raw outputs.
  - [ ] Docling outputs validated (page count, content, structure).
  - [ ] Normalization converts Docling output to internal schemas.
  - [ ] Hierarchy builder assigns section paths to all elements.
  - [ ] Table processor produces markdown/HTML/JSON/summary for all tables.
  - [ ] Image processor saves assets, classifies types, prepares metadata.
  - [ ] Formula processor extracts LaTeX and links to explanatory text.
  - [ ] Relationship generator creates all typed relationships, no duplicates.
  - [ ] Hierarchical chunker produces section-based chunks with element refs.
  - [ ] Semantic chunker produces similarity-based chunks.
  - [ ] Cluster-semantic chunker produces thematic clusters.
  - [ ] Non-text elements integrated into chunks via structured references.
  - [ ] Export module writes JSONL, Parquet, metadata, and review reports.
  - [ ] Validation checks and benchmark documents exist and pass.
  - [ ] Pipeline orchestrator runs end-to-end from CLI.
  - [ ] All tests pass: `pytest tests/`.
- **Performance Check:** The pipeline should process a 20-page document in under 5 minutes on CPU (or faster on GPU). Log timing per stage for benchmarking.
- **Error Handling:** The pipeline should never crash with an unhandled exception. Every stage must handle failures gracefully, log the error, and continue or abort with a clear message.

## Open Questions

1. **Docling version pinning:** What specific version of `docling` should be targeted? The current `requirements.txt` does not include `docling`. We should add `docling>=2.0,<3.0` (or whatever the current stable version is) to `requirements.txt` / `pyproject.toml`. **Decision needed before Sub-Task 4.**
2. **Embedding model for semantic chunking:** `BAAI/bge-m3` is specified as the default. Should we also support smaller/faster models (e.g., `all-MiniLM-L6-v2`) for CPU-only environments? Can be resolved during Sub-Task 13 implementation.
3. **Review report format:** Should the review report be HTML (interactive, with collapsible sections) or Markdown (simpler, version-control friendly)? HTML is specified in the plan but can be changed to Markdown if preferred.
4. **Benchmark documents:** The project includes `2502.04644v1.pdf` (an arXiv paper). Can we use this as the complex benchmark document, or do we need to identify additional documents? The team should confirm before Sub-Task 17.
