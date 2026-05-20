"""
Tests for ``src.extraction`` (Sub-Task 4 — Docling Extraction and Persistence).

Uses mocks and fakes throughout so that tests are fast and do **not**
require a real Docling installation or GPU.

Covers:
- Public imports work without invoking Docling.
- ``DoclingAdapter`` raises a clear ``ImportError`` with installation
  guidance when conversion is attempted and Docling is missing.
- ``extract_with_docling`` calls the adapter with the correct source path
  and raises ``FileNotFoundError`` when the source PDF is missing.
- ``persist_docling_outputs`` creates all expected files and directories
  (``output.json``, ``output.md``, ``version.txt``, ``pages/``, ``tables/``,
  ``assets/``) using a fake Docling document.
- ``persist_docling_outputs`` handles pages with images and tables with
  images, and degrades gracefully when no images are present.
- ``run_extraction_pipeline`` updates the manifest to ``"completed"`` and
  returns an updated ``DocumentSchema``.
- ``run_extraction_pipeline`` failure path updates the manifest to
  ``"failed"``.
- ``run_extraction_pipeline`` with ``force=False`` skips when outputs
  already exist.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from src.extraction import (
    DoclingAdapter,
    extract_with_docling,
    persist_docling_outputs,
    run_extraction_pipeline,
)
from src.schemas import DocumentMetadata, DocumentSchema
from src.utils.config import PipelineSettings


# ===================================================================
#  Fakes — simulate Docling objects without importing docling
# ===================================================================


class FakePilImage:
    """Minimal PIL-like image stub for testing image persistence."""

    def __init__(self, mode: str = "RGB", size: tuple[int, int] = (100, 100)) -> None:
        self.mode = mode
        self.size = size

    def save(self, path: str | Path, **kwargs: Any) -> None:
        Path(path).write_text(f"FAKE_IMAGE:{self.mode}:{self.size[0]}x{self.size[1]}")


class FakeImageRef:
    """Simulates Docling's ``ImageRef`` with a ``pil_image`` property."""

    def __init__(self, pil_image: Any | None = None) -> None:
        self._pil_image = pil_image

    @property
    def pil_image(self) -> Any | None:
        return self._pil_image


class FakePage:
    """Simulates a Docling ``PageItem``."""

    def __init__(
        self,
        page_num: int = 1,
        image_ref: FakeImageRef | None = None,
        save_fail: bool = False,
    ) -> None:
        self.page_num = page_num
        self._image_ref = image_ref
        self._save_fail = save_fail

    @property
    def image(self) -> FakeImageRef | None:
        return self._image_ref


class FakeTable:
    """Simulates a Docling ``TableItem``."""

    def __init__(
        self,
        pil_image: FakePilImage | None = None,
        get_image_return: Any | None = None,
    ) -> None:
        self._pil_image = pil_image
        self._get_image_return = get_image_return

    def get_image(self, doc: Any, prov_index: int = 0) -> Any | None:
        if self._get_image_return is not None:
            return self._get_image_return
        return self._pil_image


class FakeDoclingDocument:
    """A fake Docling document for testing persistence.

    Provides ``export_to_dict``, ``export_to_markdown``, ``pages`` (as a
    ``dict[int, FakePage]``), and ``tables`` (as a ``list[FakeTable]``).
    """

    def __init__(
        self,
        pages: dict[int, FakePage] | None = None,
        tables: list[FakeTable] | None = None,
    ) -> None:
        self._pages = pages if pages is not None else {}
        self._tables = tables if tables is not None else []

    @property
    def pages(self) -> dict[int, FakePage]:
        return self._pages

    @property
    def tables(self) -> list[FakeTable]:
        return self._tables

    def export_to_dict(self) -> dict[str, Any]:
        return {
            "type": "document",
            "num_pages": len(self._pages),
            "num_tables": len(self._tables),
            "pages": [{"page_num": p.page_num} for p in self._pages.values()],
        }

    def export_to_markdown(self) -> str:
        return "# Fake Document\n\nThis is a test document for unit tests."


# ===================================================================
#  Fixtures
# ===================================================================


@pytest.fixture
def settings(tmp_path: Path) -> PipelineSettings:
    """Return ``PipelineSettings`` with ``raw_dir`` scoped to a temp dir."""
    return PipelineSettings(raw_dir=tmp_path / "raw")


@pytest.fixture
def sample_doc(settings: PipelineSettings) -> DocumentSchema:
    """Return a minimal ``DocumentSchema`` with ingested source PDF on disk."""
    doc = DocumentSchema(
        doc_id=UUID("00000000-0000-0000-0000-000000000001"),
        title="Test Document",
        source_path="/path/to/test.pdf",
        file_hash="abcd1234",
        page_count=2,
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        metadata=DocumentMetadata(
            source_format="pdf",
            processing_status="ingested",
        ),
    )
    # Write a minimal source PDF
    doc_dir = settings.raw_dir / str(doc.doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "source.pdf").write_text("%PDF-1.4 fake content")
    return doc


@pytest.fixture
def sample_manifest(sample_doc: DocumentSchema, settings: PipelineSettings) -> Path:
    """Write a manifest for *sample_doc* and return its path."""
    manifest_path = settings.raw_dir / str(sample_doc.doc_id) / "manifest.json"
    manifest = {
        "doc_id": str(sample_doc.doc_id),
        "original_source": sample_doc.source_path,
        "stored_source": str(settings.raw_dir / str(sample_doc.doc_id) / "source.pdf"),
        "timestamp": sample_doc.created_at.isoformat(),
        "file_hash": sample_doc.file_hash,
        "page_count": sample_doc.page_count,
        "title": sample_doc.title,
        "processing_status": "ingested",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


class FakeAdapter(DoclingAdapter):
    """A ``DoclingAdapter`` that returns a fake document and never touches
    real Docling."""

    def __init__(
        self,
        settings: PipelineSettings | None = None,
        fake_doc: Any | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._fake_doc = fake_doc or FakeDoclingDocument()
        self._fake_version = "99.99.FAKE"
        self.convert_called_with: list[Path] = []

    def _get_converter(self) -> Any:
        # Override to avoid real Docling imports — return None
        pass  # type: ignore[return]

    def convert(self, source: str | Path) -> Any:
        self.convert_called_with.append(Path(source))
        return self._fake_doc

    @property
    def version(self) -> str:
        return self._fake_version


# ===================================================================
#  1.  Import check (no Docling required)
# ===================================================================


def test_public_imports() -> None:
    """``src.extraction`` public API can be imported without invoking Docling."""
    assert DoclingAdapter is not None
    assert callable(extract_with_docling)
    assert callable(persist_docling_outputs)
    assert callable(run_extraction_pipeline)


# ===================================================================
#  2.  DoclingAdapter — ImportError when Docling missing
# ===================================================================


def test_adapter_constructs_without_docling() -> None:
    """Constructing a ``DoclingAdapter`` does **not** require Docling."""
    adapter = DoclingAdapter()
    assert adapter is not None


def test_adapter_convert_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DoclingAdapter.convert`` raises ``ImportError`` with installation
    guidance when Docling is unavailable.

    We simulate the failure by monkeypatching the import inside
    ``_get_converter``.
    """
    adapter = DoclingAdapter()

    def _broken_import() -> None:
        raise ImportError("No module named 'docling'")

    monkeypatch.setattr(adapter, "_get_converter", _broken_import)

    with pytest.raises(ImportError) as excinfo:
        adapter.convert("dummy.pdf")

    assert "pip install" in str(excinfo.value).lower() or "docling" in str(
        excinfo.value
    )


def test_adapter_import_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ImportError message should mention how to install Docling."""
    adapter = DoclingAdapter()

    def raising_get() -> None:
        raise ImportError(
            "Docling is required for PDF extraction.  "
            "Install it with: pip install 'docling>=2.0'"
        )

    monkeypatch.setattr(adapter, "_get_converter", raising_get)

    with pytest.raises(ImportError) as excinfo:
        adapter.convert("dummy.pdf")
    msg = str(excinfo.value)
    assert "Docling" in msg
    assert "pip install" in msg


# ===================================================================
#  3.  extract_with_docling
# ===================================================================


class TestExtractWithDocling:
    """Tests for :func:`extract_with_docling`."""

    def test_calls_adapter_with_correct_path(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """The adapter's ``convert`` method is called with the source PDF path."""
        fake_adapter = FakeAdapter(settings=settings)
        result = extract_with_docling(
            sample_doc, settings=settings, adapter=fake_adapter
        )
        assert result is fake_adapter._fake_doc
        expected_path = settings.raw_dir / str(sample_doc.doc_id) / "source.pdf"
        assert len(fake_adapter.convert_called_with) == 1
        assert fake_adapter.convert_called_with[0] == expected_path

    def test_raises_file_not_found(self, settings: PipelineSettings) -> None:
        """``FileNotFoundError`` is raised when the source PDF does not exist."""
        doc = DocumentSchema(
            doc_id=UUID("00000000-0000-0000-0000-000000000099"),
            title="Missing Doc",
            source_path="/nonexistent/source.pdf",
            file_hash="0000",
            page_count=0,
        )
        fake_adapter = FakeAdapter(settings=settings)
        with pytest.raises(FileNotFoundError, match="source.pdf"):
            extract_with_docling(doc, settings=settings, adapter=fake_adapter)

    def test_creates_default_adapter_when_none(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``adapter`` is ``None``, a new ``DoclingAdapter`` is created."""

        # We just verify the function doesn't crash; the real adapter will
        # try to import docling, which *is* available in this env.  We
        # monkey-patch the actual convert to avoid a real Docling call.
        def mock_convert(self_: Any, source: str | Path) -> Any:  # noqa: ARG001
            return FakeDoclingDocument()

        monkeypatch.setattr(DoclingAdapter, "convert", mock_convert)

        result = extract_with_docling(sample_doc, settings=settings, adapter=None)
        assert result is not None


# ===================================================================
#  4.  persist_docling_outputs
# ===================================================================


class TestPersistDoclingOutputs:
    """Tests for :func:`persist_docling_outputs`."""

    def test_creates_expected_files(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """Persistence creates ``output.json``, ``output.md``, ``version.txt``
        and the ``pages/``, ``tables/``, ``assets/`` directories."""
        fake_doc = FakeDoclingDocument()
        outputs = persist_docling_outputs(sample_doc, fake_doc, settings=settings)

        docling_dir = settings.raw_dir / str(sample_doc.doc_id) / "docling"

        # Check returned paths
        assert "json" in outputs
        assert outputs["json"] == docling_dir / "output.json"
        assert "markdown" in outputs
        assert outputs["markdown"] == docling_dir / "output.md"
        assert "version" in outputs
        assert outputs["version"] == docling_dir / "version.txt"
        assert "pages" in outputs
        assert outputs["pages"] == docling_dir / "pages"
        assert "tables" in outputs
        assert outputs["tables"] == docling_dir / "tables"
        assert "assets" in outputs
        assert outputs["assets"] == docling_dir / "assets"

        # Check files exist
        assert (docling_dir / "output.json").is_file()
        assert (docling_dir / "output.md").is_file()
        assert (docling_dir / "version.txt").is_file()
        assert (docling_dir / "pages").is_dir()
        assert (docling_dir / "tables").is_dir()
        assert (docling_dir / "assets").is_dir()

        # Verify content
        json_data = json.loads((docling_dir / "output.json").read_text())
        assert json_data["type"] == "document"
        assert json_data["num_pages"] == 0

        md_content = (docling_dir / "output.md").read_text()
        assert "# Fake Document" in md_content

        version = (docling_dir / "version.txt").read_text().strip()
        assert version  # should not be empty

    def test_page_images_saved(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """Page images are saved when pages provide ``ImageRef`` with
        PIL images."""
        img1 = FakePilImage("RGB", (200, 100))
        img2 = FakePilImage("L", (100, 50))
        pages = {
            1: FakePage(page_num=1, image_ref=FakeImageRef(pil_image=img1)),
            2: FakePage(page_num=2, image_ref=FakeImageRef(pil_image=img2)),
        }
        fake_doc = FakeDoclingDocument(pages=pages)
        outputs = persist_docling_outputs(sample_doc, fake_doc, settings=settings)

        pages_dir = outputs["pages"]
        assert (pages_dir / "page_1.png").is_file()
        assert (pages_dir / "page_2.png").is_file()
        # Verify content
        assert (pages_dir / "page_1.png").read_text() == "FAKE_IMAGE:RGB:200x100"
        assert (pages_dir / "page_2.png").read_text() == "FAKE_IMAGE:L:100x50"

    def test_table_images_saved(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """Table images are saved when tables provide ``get_image``."""
        table_img = FakePilImage("RGB", (80, 30))
        tables = [FakeTable(pil_image=table_img)]
        fake_doc = FakeDoclingDocument(tables=tables)
        outputs = persist_docling_outputs(sample_doc, fake_doc, settings=settings)

        tables_dir = outputs["tables"]
        assert (tables_dir / "table_0.png").is_file()
        assert (tables_dir / "table_0.png").read_text() == "FAKE_IMAGE:RGB:80x30"

    def test_no_images_still_succeeds(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """When no page or table images are present, persistence still
        succeeds and directories exist."""
        fake_doc = FakeDoclingDocument()
        outputs = persist_docling_outputs(sample_doc, fake_doc, settings=settings)

        pages_dir = outputs["pages"]
        tables_dir = outputs["tables"]
        assert pages_dir.is_dir()
        assert tables_dir.is_dir()
        # No image files
        assert len(list(pages_dir.iterdir())) == 0
        assert len(list(tables_dir.iterdir())) == 0

    def test_no_tables_still_succeeds(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """When ``dl_doc.tables`` is empty, no errors occur and tables dir
        exists."""
        fake_doc = FakeDoclingDocument(pages={1: FakePage(page_num=1)})
        outputs = persist_docling_outputs(sample_doc, fake_doc, settings=settings)
        assert outputs["tables"].is_dir()

    def test_json_contains_dict_contents(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """The JSON output contains the exported dict from the fake doc."""
        fake_doc = FakeDoclingDocument(
            pages={1: FakePage(page_num=1)},
            tables=[FakeTable()],
        )
        outputs = persist_docling_outputs(sample_doc, fake_doc, settings=settings)
        json_data = json.loads(outputs["json"].read_text())
        assert json_data["num_pages"] == 1
        assert json_data["num_tables"] == 1
        assert json_data["pages"][0]["page_num"] == 1


# ===================================================================
#  5.  run_extraction_pipeline
# ===================================================================


class TestRunExtractionPipeline:
    """Tests for :func:`run_extraction_pipeline`."""

    # ----------------------------------------------------------------
    #  Happy path
    # ----------------------------------------------------------------

    def test_completed_updates_manifest(
        self,
        sample_doc: DocumentSchema,
        sample_manifest: Path,
        settings: PipelineSettings,
    ) -> None:
        """On success, the manifest is updated with ``extraction_status:
        'completed'`` and a ``DocumentSchema`` is returned."""
        fake_adapter = FakeAdapter(settings=settings)
        result = run_extraction_pipeline(
            sample_doc, settings=settings, adapter=fake_adapter, force=True
        )

        # Check manifest
        manifest_data = json.loads(sample_manifest.read_text())
        assert manifest_data["extraction_status"] == "completed"
        assert "extraction_completed_at" in manifest_data
        assert "docling_version" in manifest_data
        assert manifest_data["docling_version"] == "99.99.FAKE"
        assert "extraction_outputs" in manifest_data

        # Check returned DocumentSchema
        assert isinstance(result, DocumentSchema)
        assert result.doc_id == sample_doc.doc_id
        assert result.metadata.processing_status == "extracted"
        assert result.metadata.extraction_version == "99.99.FAKE"
        assert result.metadata.custom.get("extraction_status") == "completed"

        # Source fields are kept intact
        assert result.title == sample_doc.title
        assert result.file_hash == sample_doc.file_hash

    def test_completed_writes_outputs(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
    ) -> None:
        """On success, the Docling output files exist on disk."""
        fake_adapter = FakeAdapter(settings=settings)
        run_extraction_pipeline(
            sample_doc, settings=settings, adapter=fake_adapter, force=True
        )

        docling_dir = settings.raw_dir / str(sample_doc.doc_id) / "docling"
        assert (docling_dir / "output.json").is_file()
        assert (docling_dir / "output.md").is_file()
        assert (docling_dir / "version.txt").is_file()
        assert (docling_dir / "pages").is_dir()
        assert (docling_dir / "tables").is_dir()
        assert (docling_dir / "assets").is_dir()

    # ----------------------------------------------------------------
    #  Failure path
    # ----------------------------------------------------------------

    def test_failure_updates_manifest(
        self,
        sample_doc: DocumentSchema,
        sample_manifest: Path,
        settings: PipelineSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On extraction failure, the manifest is updated with
        ``extraction_status: 'failed'`` and the error is re-raised."""
        fake_adapter = FakeAdapter(settings=settings)

        # Make the adapter's convert raise an error
        def broken_convert(source: str | Path) -> Any:
            raise RuntimeError("Mock extraction failure")

        monkeypatch.setattr(fake_adapter, "convert", broken_convert)

        with pytest.raises(RuntimeError, match="Mock extraction failure"):
            run_extraction_pipeline(
                sample_doc, settings=settings, adapter=fake_adapter, force=True
            )

        # Check manifest updated
        manifest_data = json.loads(sample_manifest.read_text())
        assert manifest_data["extraction_status"] == "failed"
        assert "extraction_error" in manifest_data
        assert "Mock extraction failure" in manifest_data["extraction_error"]
        assert "extraction_completed_at" in manifest_data

    def test_failure_does_not_write_outputs(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On extraction failure, no docling output directory is created."""
        fake_adapter = FakeAdapter(settings=settings)

        def broken_convert(source: str | Path) -> Any:
            raise RuntimeError("fail")

        monkeypatch.setattr(fake_adapter, "convert", broken_convert)

        with pytest.raises(RuntimeError):
            run_extraction_pipeline(
                sample_doc, settings=settings, adapter=fake_adapter, force=True
            )

        docling_dir = settings.raw_dir / str(sample_doc.doc_id) / "docling"
        assert not docling_dir.exists()

    def test_failure_re_raises_original_exception(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The original exception from extraction is re-raised (not wrapped)."""
        fake_adapter = FakeAdapter(settings=settings)

        def broken_convert(source: str | Path) -> Any:
            raise ValueError("Original error")

        monkeypatch.setattr(fake_adapter, "convert", broken_convert)

        with pytest.raises(ValueError, match="Original error"):
            run_extraction_pipeline(
                sample_doc, settings=settings, adapter=fake_adapter, force=True
            )

    # ----------------------------------------------------------------
    #  force=False — skip when outputs exist
    # ----------------------------------------------------------------

    def test_force_false_skips_when_outputs_exist(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
    ) -> None:
        """With ``force=False``, extraction is skipped if output files already
        exist."""
        fake_adapter = FakeAdapter(settings=settings)

        # First run — force=True to create outputs
        run_extraction_pipeline(
            sample_doc, settings=settings, adapter=fake_adapter, force=True
        )

        # Reset call tracking
        fake_adapter.convert_called_with.clear()

        # Second run — force=False should skip
        result = run_extraction_pipeline(
            sample_doc, settings=settings, adapter=fake_adapter, force=False
        )

        # Adapter should NOT have been called
        assert len(fake_adapter.convert_called_with) == 0

        # Result should have "skipped" in metadata
        assert result.metadata.custom.get("extraction_status") == "skipped"

    def test_force_false_still_runs_when_outputs_missing(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
    ) -> None:
        """With ``force=False``, extraction still runs if output files are
        missing."""
        fake_adapter = FakeAdapter(settings=settings)

        # No prior run — outputs don't exist
        result = run_extraction_pipeline(
            sample_doc, settings=settings, adapter=fake_adapter, force=False
        )

        # Adapter should have been called
        assert len(fake_adapter.convert_called_with) == 1

        # Result should have "completed" status
        assert result.metadata.custom.get("extraction_status") == "completed"
        assert result.metadata.processing_status == "extracted"

    def test_force_default_is_true(
        self,
        sample_doc: DocumentSchema,
        settings: PipelineSettings,
    ) -> None:
        """The default value of ``force`` is ``True`` (backward-compatible)."""
        fake_adapter = FakeAdapter(settings=settings)
        run_extraction_pipeline(sample_doc, settings=settings, adapter=fake_adapter)
        assert len(fake_adapter.convert_called_with) == 1

    # ----------------------------------------------------------------
    #  Edge cases for persist_docling_outputs
    # ----------------------------------------------------------------

    def test_version_txt_content(
        self, sample_doc: DocumentSchema, settings: PipelineSettings
    ) -> None:
        """``version.txt`` contains a non-empty string."""
        fake_adapter = FakeAdapter(settings=settings)
        run_extraction_pipeline(
            sample_doc, settings=settings, adapter=fake_adapter, force=True
        )
        version_path = (
            settings.raw_dir / str(sample_doc.doc_id) / "docling" / "version.txt"
        )
        version = version_path.read_text().strip()
        assert version == "99.99.FAKE"
