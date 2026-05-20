"""
Document ingestion and raw storage implementation.

Handles copying source PDFs into an immutable raw store, computing file
hashes, creating document records, and writing ingestion manifests.

Public API
----------
- :class:`IngestionSource` — flexible source specification.
- :func:`ingest_pdf` — ingest a single PDF file.
- :func:`batch_ingest` — ingest multiple documents.
- :func:`get_document` — load a previously ingested document record.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from uuid import UUID

from src.schemas import (
    DocumentMetadata,
    DocumentSchema,
    PageSchema,
    make_doc_id,
)
from src.utils.config import PipelineSettings
from src.utils.file_io import copy_with_hash, ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IngestionSource
# ---------------------------------------------------------------------------


@dataclass
class IngestionSource:
    """Flexible specification of one or more source documents to ingest.

    Attributes:
        paths: Local file or directory paths to ingest.
        url:   Remote URL (not yet supported; raises ``NotImplementedError``
               if ingestion is attempted).
    """

    paths: List[Path] = field(default_factory=list)
    url: Optional[str] = None

    @classmethod
    def from_path(cls, path: str | Path) -> IngestionSource:
        """Create an ``IngestionSource`` from a single file or directory."""
        return cls(paths=[Path(path)])

    @classmethod
    def from_paths(cls, paths: list[str | Path]) -> IngestionSource:
        """Create an ``IngestionSource`` from a list of paths."""
        return cls(paths=[Path(p) for p in paths])

    @classmethod
    def from_url(cls, url: str) -> IngestionSource:
        """Create an ``IngestionSource`` from a URL (not yet supported)."""
        return cls(url=url)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_page_count(path: Path) -> int:
    """Return the number of pages in a PDF using pdfminer.

    Returns 0 (with a logged warning) for empty or malformed PDFs so that
    the ingestion pipeline does not crash on broken inputs.
    """
    from pdfminer.pdfpage import PDFPage

    try:
        with path.open("rb") as fh:
            count = sum(1 for _ in PDFPage.get_pages(fh))
        return count
    except Exception as exc:
        logger.warning("Could not determine page count for %s: %s", path, exc)
        return 0


def _get_pdf_title(path: Path) -> str:
    """Extract the title from PDF document metadata, falling back to the
    filename stem if metadata is absent or unparseable."""
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser

    try:
        with path.open("rb") as fh:
            parser = PDFParser(fh)
            doc = PDFDocument(parser)
            if doc.info:
                raw = doc.info[0].get("Title")
                if raw:
                    title = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )  # type: ignore[arg-type]
                    if title.strip():
                        return title.strip()
    except Exception:
        logger.debug(
            "Could not extract PDF title from %s, falling back to filename", path
        )

    return path.stem


def _write_manifest(doc_dir: Path, doc: DocumentSchema) -> None:
    """Write the ingestion manifest for *doc* into *doc_dir*."""
    manifest = {
        "doc_id": str(doc.doc_id),
        "original_source": doc.source_path,
        "stored_source": str(doc_dir / "source.pdf"),
        "timestamp": doc.created_at.isoformat(),
        "file_hash": doc.file_hash,
        "page_count": doc.page_count,
        "title": doc.title,
        "processing_status": "ingested",
    }
    write_json(doc_dir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# Core ingestion — single PDF
# ---------------------------------------------------------------------------


def ingest_pdf(
    source: str | Path,
    settings: PipelineSettings | None = None,
) -> DocumentSchema:
    """Ingest a single PDF file and store it immutably.

    Steps
    -----
    1. Validate that *source* exists and has a ``.pdf`` extension.
    2. Stream-copy the file to a staging path under ``raw_dir``, computing
       the SHA-256 hash simultaneously.
    3. Derive the deterministic ``doc_id`` from the hash via
       :func:`~src.schemas.id_gen.make_doc_id`.
    4. Move the staged copy to ``{raw_dir}/{doc_id}/source.pdf`` (idempotent:
       if the destination already exists the staging file is discarded).
    5. Extract page count (pdfminer) and title (PDF metadata or filename).
    6. Return a minimal frozen :class:`DocumentSchema` and write
       ``manifest.json``.

    Parameters
    ----------
    source:
        Path to a local PDF file.
    settings:
        Pipeline configuration.  Uses defaults when ``None``.

    Returns
    -------
    DocumentSchema
        A frozen document record with basic metadata populated.

    Raises
    ------
    FileNotFoundError
        If *source* does not exist.
    ValueError
        If *source* is not a PDF file (based on lowercase suffix).
    NotImplementedError
        If URL-based ingestion is attempted (reserved for future use).
    """
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Source file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Source file is not a PDF (suffix={path.suffix!r}): {path}")

    settings = settings or PipelineSettings()
    raw_dir = ensure_dir(settings.raw_dir)

    # ------------------------------------------------------------------
    # Stage: copy to a temporary path under raw_dir while computing hash
    # ------------------------------------------------------------------
    staging = raw_dir / f".ingest_staging_{os.getpid()}.pdf"
    try:
        file_hash = copy_with_hash(path, staging)
    except BaseException:
        # Clean up staging on any failure during copy
        if staging.exists():
            staging.unlink(missing_ok=True)
        raise

    doc_id = make_doc_id(file_hash)
    doc_dir = ensure_dir(raw_dir / str(doc_id))
    source_pdf_dst = doc_dir / "source.pdf"

    if source_pdf_dst.exists():
        # Already ingested — discard the staging copy
        staging.unlink(missing_ok=True)
    else:
        staging.replace(source_pdf_dst)

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------
    page_count = _get_page_count(path)
    title = _get_pdf_title(path)
    now = datetime.now()

    # ------------------------------------------------------------------
    # Build DocumentSchema (frozen — must supply all fields)
    # ------------------------------------------------------------------
    doc = DocumentSchema(
        doc_id=doc_id,
        title=title,
        source_path=str(path.resolve()),
        file_hash=file_hash,
        page_count=page_count,
        created_at=now,
        pages=[PageSchema(page_num=i) for i in range(1, page_count + 1)],
        metadata=DocumentMetadata(
            source_format="pdf",
            processing_status="ingested",
            processing_started_at=now,
            processing_completed_at=now,
        ),
    )

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    _write_manifest(doc_dir, doc)

    return doc


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------


def batch_ingest(
    sources: str | Path | list[str | Path] | IngestionSource,
    settings: PipelineSettings | None = None,
) -> list[DocumentSchema]:
    """Ingest multiple PDF documents.

    *sources* can be:

    - A single file path (string or :class:`~pathlib.Path`).
    - A directory path — all ``.pdf`` files are ingested in
      deterministic (sorted) order.
    - A list of file paths.
    - An :class:`IngestionSource` instance.

    If any individual file fails, the exception is propagated immediately
    (no partial-error aggregation).

    Parameters
    ----------
    sources:
        One or more source documents to ingest.
    settings:
        Pipeline configuration.  Uses defaults when ``None``.

    Returns
    -------
    list[DocumentSchema]
        Document records for every successfully ingested file.
    """
    # Normalise *sources* into a flat list of Paths
    if isinstance(sources, IngestionSource):
        if sources.url:
            raise NotImplementedError("URL ingestion is not yet supported")
        paths = list(sources.paths)
    elif isinstance(sources, (str, Path)):
        p = Path(sources)
        if p.is_dir():
            paths = sorted(p.iterdir())
        else:
            paths = [p]
    else:
        # Assume iterable of str / Path
        paths = [Path(s) for s in sources]

    # Keep only PDF files
    pdf_paths = sorted(p for p in paths if p.is_file() and p.suffix.lower() == ".pdf")

    return [ingest_pdf(p, settings=settings) for p in pdf_paths]


# ---------------------------------------------------------------------------
# Document lookup
# ---------------------------------------------------------------------------


def get_document(
    doc_id: str | UUID,
    settings: PipelineSettings | None = None,
) -> DocumentSchema:
    """Load a previously ingested document from its manifest.

    Parameters
    ----------
    doc_id:
        The document UUID (as a string or :class:`~uuid.UUID`).
    settings:
        Pipeline configuration.  Uses defaults when ``None``.

    Returns
    -------
    DocumentSchema
        The reconstructed document record.

    Raises
    ------
    FileNotFoundError
        If no manifest exists for *doc_id*.
    """
    settings = settings or PipelineSettings()

    doc_id_str = str(doc_id)
    manifest_path = settings.raw_dir / doc_id_str / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for document {doc_id_str}")

    manifest = read_json(manifest_path)

    # Use the UUID from the manifest (authoritative)
    resolved_id = UUID(manifest["doc_id"])
    page_count = manifest["page_count"]
    created_at = datetime.fromisoformat(manifest["timestamp"])

    return DocumentSchema(
        doc_id=resolved_id,
        title=manifest["title"],
        source_path=manifest["original_source"],
        file_hash=manifest["file_hash"],
        page_count=page_count,
        created_at=created_at,
        pages=[PageSchema(page_num=i) for i in range(1, page_count + 1)],
        metadata=DocumentMetadata(
            source_format="pdf",
            processing_status=manifest.get("processing_status", "ingested"),
            processing_started_at=created_at,
            processing_completed_at=created_at,
        ),
    )
