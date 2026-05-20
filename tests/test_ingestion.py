"""
Tests for ``src.ingestion`` (Sub-Task 3 — Document Ingestion and Raw Storage).

Covers:
- Basic ``ingest_pdf`` creates ``source.pdf`` and ``manifest.json`` and
  returns a valid ``DocumentSchema``.
- Same file ingested twice yields the same ``doc_id`` and file hash.
- ``batch_ingest`` handles a list and/or directory of PDFs.
- ``get_document`` round-trips an ingested document.
- Missing path raises ``FileNotFoundError``.
- Non-PDF local file raises a clear exception.
- Empty / malformed ``.pdf`` is handled gracefully (``page_count=0``).
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from uuid import UUID

import pytest

from src.ingestion import IngestionSource, batch_ingest, get_document, ingest_pdf
from src.utils.config import PipelineSettings


# ===================================================================
#  Helpers — minimal PDF generation
# ===================================================================


def _make_pdf_bytes(num_pages: int = 1) -> bytes:
    """Generate bytes for a minimal valid PDF with *num_pages* pages.

    The PDF is structurally valid enough for pdfminer to parse its page
    tree and count pages.  No content streams are included.
    """
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")

    offsets: dict[int, int] = {}

    def _write_obj(num: int, content: bytes) -> None:
        offsets[num] = buf.tell()
        buf.write(f"{num} 0 obj\n".encode())
        buf.write(content)
        buf.write(b"\nendobj\n")

    # Object 1: Catalog
    _write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")

    # Object 2: Pages tree root
    kids = " ".join(f"{i + 3} 0 R" for i in range(num_pages))
    _write_obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode())

    # Page objects
    for i in range(num_pages):
        _write_obj(i + 3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")

    # ---- xref table ----
    xref_offset = buf.tell()
    buf.write(b"xref\n")
    entry_count = num_pages + 3
    buf.write(f"0 {entry_count}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for num in range(1, entry_count):
        buf.write(f"{offsets[num]:010d} 00000 n \n".encode())

    # ---- trailer ----
    buf.write(f"trailer\n<< /Size {entry_count} /Root 1 0 R >>\n".encode())
    buf.write(b"startxref\n")
    buf.write(f"{xref_offset}\n".encode())
    buf.write(b"%%EOF")

    return buf.getvalue()


def _create_pdf(
    tmp_path: Path,
    name: str = "test.pdf",
    num_pages: int = 1,
) -> Path:
    """Write a minimal valid PDF to *tmp_path / name* and return its path."""
    path = tmp_path / name
    path.write_bytes(_make_pdf_bytes(num_pages))
    return path


def _settings_with(tmp_path: Path) -> PipelineSettings:
    """Return ``PipelineSettings`` with ``raw_dir`` scoped to *tmp_path*."""
    return PipelineSettings(raw_dir=tmp_path / "raw")


# ===================================================================
#  Tests
# ===================================================================


class TestIngestPdf:
    """Tests for :func:`ingest_pdf`."""

    def test_basic_ingest_creates_files(self, tmp_path: Path) -> None:
        """Ingesting a PDF creates ``source.pdf`` and ``manifest.json`` and
        returns a valid ``DocumentSchema``."""
        pdf = _create_pdf(tmp_path, "hello.pdf", num_pages=2)
        settings = _settings_with(tmp_path)

        doc = ingest_pdf(pdf, settings=settings)

        # Check returned schema
        assert isinstance(doc.doc_id, UUID)
        assert doc.title == "hello"  # filename stem
        assert doc.page_count == 2
        assert doc.file_hash == hashlib.sha256(pdf.read_bytes()).hexdigest()
        assert doc.metadata.source_format == "pdf"
        assert doc.metadata.processing_status == "ingested"
        assert doc.source_path == str(pdf.resolve())
        assert len(doc.pages) == 2
        assert doc.pages[0].page_num == 1
        assert doc.pages[1].page_num == 2

        # Check files on disk
        doc_dir = settings.raw_dir / str(doc.doc_id)
        assert (doc_dir / "source.pdf").is_file()
        assert (doc_dir / "manifest.json").is_file()

        # Verify manifest contents
        import json

        manifest = json.loads((doc_dir / "manifest.json").read_text())
        assert manifest["doc_id"] == str(doc.doc_id)
        assert manifest["file_hash"] == doc.file_hash
        assert manifest["page_count"] == 2
        assert manifest["processing_status"] == "ingested"
        assert manifest["title"] == "hello"

        # Staging file must be cleaned up
        staging_files = list(settings.raw_dir.glob(".ingest_staging_*"))
        assert len(staging_files) == 0

    def test_idempotent_same_file_same_doc_id(self, tmp_path: Path) -> None:
        """Ingesting the same file twice produces the same ``doc_id`` and
        does not duplicate the stored copy."""
        pdf = _create_pdf(tmp_path, "same.pdf")
        settings = _settings_with(tmp_path)

        doc1 = ingest_pdf(pdf, settings=settings)
        doc2 = ingest_pdf(pdf, settings=settings)

        assert doc1.doc_id == doc2.doc_id
        assert doc1.file_hash == doc2.file_hash

        # Only one copy on disk
        doc_dir = settings.raw_dir / str(doc1.doc_id)
        source_path = doc_dir / "source.pdf"
        assert source_path.is_file()
        # Check that source.pdf wasn't overwritten with different content
        assert source_path.read_bytes() == pdf.read_bytes()

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """A non-existent file path raises ``FileNotFoundError``."""
        missing = tmp_path / "does_not_exist.pdf"
        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            ingest_pdf(missing, settings=_settings_with(tmp_path))

    def test_non_pdf_extension_raises(self, tmp_path: Path) -> None:
        """A file without a ``.pdf`` extension raises ``ValueError``."""
        not_pdf = tmp_path / "data.txt"
        not_pdf.write_text("This is not a PDF")
        with pytest.raises(ValueError, match="not a PDF"):
            ingest_pdf(not_pdf, settings=_settings_with(tmp_path))

    def test_uppercase_extension_accepted(self, tmp_path: Path) -> None:
        """A file with ``.PDF`` (uppercase) is accepted."""
        pdf = _create_pdf(tmp_path, "UPPER.PDF")
        doc = ingest_pdf(pdf, settings=_settings_with(tmp_path))
        assert doc.page_count == 1
        assert doc.title == "UPPER"

    def test_title_from_pdf_metadata(self, tmp_path: Path) -> None:
        """When PDF metadata contains a Title, it is used instead of the
        filename stem."""
        # Create a PDF with a Title metadata entry
        pdf_bytes = _make_pdf_bytes(1)
        # Inject a metadata object (Object 4) and reference it from the
        # Catalog's /Metadata entry.  We use a minimal info dict instead
        # because pdfminer reads /Info from the trailer.
        #
        # Simpler approach: write a trailer with an /Info reference and
        # add an info object.
        import io as _io

        buf = _io.BytesIO()
        buf.write(b"%PDF-1.4\n")
        buf.write(
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /Metadata 4 0 R >>\nendobj\n"
        )
        offsets = {1: buf.tell() - 100}  # placeholder, fixed below
        # We'll track offsets properly
        buf.truncate(0)
        buf.seek(0)

        # Rebuild with proper offset tracking
        offsets2: dict[int, int] = {}

        def w(num: int, content: bytes) -> None:
            offsets2[num] = buf.tell()
            buf.write(f"{num} 0 obj\n".encode())
            buf.write(content)
            buf.write(b"\nendobj\n")

        buf.write(b"%PDF-1.4\n")
        w(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        w(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        w(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
        # Object 4: Info dict with Title
        w(4, b"<< /Title (My Custom Title) >>")

        xref_off = buf.tell()
        buf.write(b"xref\n")
        entry_cnt = 5  # 0..4
        buf.write(f"0 {entry_cnt}\n".encode())
        buf.write(b"0000000000 65535 f \n")
        for n in range(1, entry_cnt):
            buf.write(f"{offsets2[n]:010d} 00000 n \n".encode())
        buf.write(
            f"trailer\n<< /Size {entry_cnt} /Root 1 0 R /Info 4 0 R >>\n".encode()
        )
        buf.write(b"startxref\n")
        buf.write(f"{xref_off}\n".encode())
        buf.write(b"%%EOF")

        meta_pdf = tmp_path / "meta.pdf"
        meta_pdf.write_bytes(buf.getvalue())

        doc = ingest_pdf(meta_pdf, settings=_settings_with(tmp_path))
        assert doc.title == "My Custom Title"

    def test_settings_default_when_none(self, tmp_path: Path) -> None:
        """When ``settings`` is ``None``, default ``PipelineSettings`` are
        used and ``raw_dir`` defaults to ``data/raw`` relative to the project
        root."""
        pdf = _create_pdf(tmp_path, "default_settings.pdf")
        # Use default settings (not overridden) — raw_dir will be project-root/data/raw
        # To avoid polluting real data, we rely on the fact that we only
        # verify structural properties of the returned document.
        doc = ingest_pdf(pdf)  # noqa: F841
        # Note: this test uses the real data/raw directory.  In CI we might
        # want to skip it, but for now the default raw_dir should exist.
        # We just verify the returned document structure.
        assert isinstance(doc.doc_id, UUID)
        assert doc.page_count == 1


# ===================================================================
#  Batch ingestion
# ===================================================================


class TestBatchIngest:
    """Tests for :func:`batch_ingest`."""

    def test_list_of_paths(self, tmp_path: Path) -> None:
        """``batch_ingest`` accepts a list of file paths."""
        pdf1 = _create_pdf(tmp_path, "a.pdf", num_pages=1)
        pdf2 = _create_pdf(tmp_path, "b.pdf", num_pages=3)
        settings = _settings_with(tmp_path)

        docs = batch_ingest([pdf1, pdf2], settings=settings)
        assert len(docs) == 2
        assert docs[0].page_count == 1
        assert docs[1].page_count == 3
        # Order should match input order
        assert docs[0].title == "a"
        assert docs[1].title == "b"

    def test_directory_path(self, tmp_path: Path) -> None:
        """``batch_ingest`` accepts a directory and ingests PDFs in sorted
        order."""
        (tmp_path / "sub").mkdir()
        # Create PDFs with deliberately non-alphabetical names
        _create_pdf(tmp_path / "sub", "z.pdf", num_pages=2)
        _create_pdf(tmp_path / "sub", "a.pdf", num_pages=1)
        _create_pdf(tmp_path / "sub", "m.pdf", num_pages=3)
        settings = _settings_with(tmp_path)

        docs = batch_ingest(tmp_path / "sub", settings=settings)
        assert len(docs) == 3
        # Sorted order: a.pdf, m.pdf, z.pdf
        titles = [d.title for d in docs]
        assert titles == ["a", "m", "z"]

    def test_ingestion_source_from_path(self, tmp_path: Path) -> None:
        """``batch_ingest`` accepts an ``IngestionSource``."""
        pdf = _create_pdf(tmp_path, "from_source.pdf")
        settings = _settings_with(tmp_path)
        source = IngestionSource.from_path(pdf)

        docs = batch_ingest(source, settings=settings)
        assert len(docs) == 1
        assert docs[0].title == "from_source"

    def test_ingestion_source_url_raises(self) -> None:
        """``batch_ingest`` with a URL-based ``IngestionSource`` raises
        ``NotImplementedError``."""
        source = IngestionSource.from_url("https://example.com/doc.pdf")
        with pytest.raises(NotImplementedError, match="URL"):
            batch_ingest(source)

    def test_non_pdf_files_in_directory_skipped(self, tmp_path: Path) -> None:
        """Non-PDF files in a directory are silently skipped."""
        sub = tmp_path / "sub2"
        sub.mkdir()
        _create_pdf(sub, "doc.pdf")
        (sub / "notes.txt").write_text("not a pdf")
        (sub / "data.csv").write_text("a,b,c")
        settings = _settings_with(tmp_path)

        docs = batch_ingest(sub, settings=settings)
        assert len(docs) == 1


# ===================================================================
#  Document lookup
# ===================================================================


class TestGetDocument:
    """Tests for :func:`get_document`."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """``get_document`` reconstructs a ``DocumentSchema`` from the
        persisted manifest."""
        pdf = _create_pdf(tmp_path, "roundtrip.pdf", num_pages=2)
        settings = _settings_with(tmp_path)

        ingested = ingest_pdf(pdf, settings=settings)
        loaded = get_document(ingested.doc_id, settings=settings)

        assert loaded.doc_id == ingested.doc_id
        assert loaded.title == ingested.title
        assert loaded.source_path == ingested.source_path
        assert loaded.file_hash == ingested.file_hash
        assert loaded.page_count == ingested.page_count
        assert loaded.created_at == ingested.created_at
        assert len(loaded.pages) == len(ingested.pages)
        assert loaded.metadata.source_format == "pdf"
        assert loaded.metadata.processing_status == "ingested"

    def test_string_uuid_accepted(self, tmp_path: Path) -> None:
        """``get_document`` accepts ``doc_id`` as a string."""
        pdf = _create_pdf(tmp_path, "string_id.pdf")
        settings = _settings_with(tmp_path)
        ingested = ingest_pdf(pdf, settings=settings)

        loaded = get_document(str(ingested.doc_id), settings=settings)
        assert loaded.doc_id == ingested.doc_id

    def test_missing_doc_id_raises(self, tmp_path: Path) -> None:
        """A non-existent ``doc_id`` raises ``FileNotFoundError``."""
        settings = _settings_with(tmp_path)
        fake_id = "00000000-0000-0000-0000-000000000000"
        with pytest.raises(FileNotFoundError, match=fake_id):
            get_document(fake_id, settings=settings)


# ===================================================================
#  Edge cases
# ===================================================================


class TestEdgeCases:
    """Graceful handling of empty, malformed, or unusual inputs."""

    def test_empty_pdf(self, tmp_path: Path) -> None:
        """A PDF with zero pages (or an empty PDF) results in
        ``page_count=0`` and does not crash."""
        # Create a PDF that has no pages (Count=0 in the Pages tree)
        buf = io.BytesIO()
        buf.write(b"%PDF-1.4\n")
        offsets: dict[int, int] = {}

        def w(num: int, content: bytes) -> None:
            offsets[num] = buf.tell()
            buf.write(f"{num} 0 obj\n".encode())
            buf.write(content)
            buf.write(b"\nendobj\n")

        w(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        w(2, b"<< /Type /Pages /Kids [] /Count 0 >>")

        xref_off = buf.tell()
        buf.write(b"xref\n0 3\n")
        buf.write(b"0000000000 65535 f \n")
        buf.write(f"{offsets[1]:010d} 00000 n \n".encode())
        buf.write(f"{offsets[2]:010d} 00000 n \n".encode())
        buf.write(b"trailer\n<< /Size 3 /Root 1 0 R >>\n")
        buf.write(b"startxref\n")
        buf.write(f"{xref_off}\n".encode())
        buf.write(b"%%EOF")

        zero_page_pdf = tmp_path / "empty.pdf"
        zero_page_pdf.write_bytes(buf.getvalue())

        settings = _settings_with(tmp_path)
        doc = ingest_pdf(zero_page_pdf, settings=settings)
        assert doc.page_count == 0
        assert doc.pages == []

    def test_malformed_pdf(self, tmp_path: Path) -> None:
        """A malformed PDF (random bytes) is handled gracefully with
        ``page_count=0``."""
        malformed = tmp_path / "broken.pdf"
        malformed.write_bytes(b"\xff\xfe\x00\x01corrupted data")

        settings = _settings_with(tmp_path)
        doc = ingest_pdf(malformed, settings=settings)
        assert doc.page_count == 0
        # The file should still be stored
        doc_dir = settings.raw_dir / str(doc.doc_id)
        assert (doc_dir / "source.pdf").is_file()
        # The hash should match the corrupted bytes
        assert doc.file_hash == hashlib.sha256(malformed.read_bytes()).hexdigest()

    def test_file_hash_correctness(self, tmp_path: Path) -> None:
        """The returned ``file_hash`` matches the SHA-256 of the original
        source file."""
        pdf = _create_pdf(tmp_path, "hash_check.pdf")
        original_bytes = pdf.read_bytes()
        expected_hash = hashlib.sha256(original_bytes).hexdigest()

        settings = _settings_with(tmp_path)
        doc = ingest_pdf(pdf, settings=settings)

        assert doc.file_hash == expected_hash

        # The stored copy should have the same hash
        stored = settings.raw_dir / str(doc.doc_id) / "source.pdf"
        assert stored.read_bytes() == original_bytes
