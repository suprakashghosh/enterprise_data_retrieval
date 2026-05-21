"""
Docling extraction adapter and persistence layer.

Wraps Docling's ``DocumentConverter`` behind a mockable
:class:`DoclingAdapter`, provides functions to convert a PDF into a Docling
document and to persist all raw outputs (JSON, markdown, page images, table
images) alongside the document record.

.. note::

    All Docling imports are isolated inside :class:`DoclingAdapter` so that
    importing ``src.extraction`` does **not** fail when Docling is
    unavailable.  An ``ImportError`` with installation guidance is raised
    only when a conversion is actually attempted.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from src.schemas import DocumentSchema
from src.utils.config import PipelineSettings
from src.utils.file_io import atomic_write_text, ensure_dir, read_json, write_json
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)

# ===================================================================
#  DoclingAdapter — thin, mockable wrapper
# ===================================================================


class DoclingAdapter:
    """Thin wrapper around Docling's ``DocumentConverter``.

    All Docling imports happen lazily inside :meth:`_get_converter` so that
    constructing an adapter does **not** require Docling to be installed.
    Conversion will fail with a clear ``ImportError`` only when
    :meth:`convert` is called.

    Parameters
    ----------
    settings:
        Pipeline configuration (used for defaults; may be ``None``).
    """

    def __init__(self, settings: PipelineSettings | None = None) -> None:
        self._settings = settings or PipelineSettings()
        self._converter: Any = None  # lazy — set by _get_converter()

    # ------------------------------------------------------------------
    #  Lazy converter initialisation
    # ------------------------------------------------------------------

    def _get_converter(self) -> Any:
        """Return (and cache) a configured ``DocumentConverter``.

        Raises
        ------
        ImportError
            If Docling is not installed.
        """
        if self._converter is not None:
            return self._converter

        try:
            from docling.datamodel.pipeline_options import (
                CodeFormulaVlmOptions,
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter
            from docling_core.types.doc import CodeItem, FormulaItem
        except ImportError as exc:
            raise ImportError(
                "Docling is required for PDF extraction.  "
                "Install it with: pip install 'docling>=2.0'"
            ) from exc

        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_table_structure = True

        # Defensive — API may differ across Docling versions
        tso = getattr(pipeline_opts, "table_structure_options", None)
        if tso is not None:
            if hasattr(tso, "do_cell_matching"):
                tso.do_cell_matching = False
            if hasattr(tso, "mode"):
                tso.mode = TableFormerMode.ACCURATE

        layout_options= getattr(pipeline_opts, "layout_options", None)
        if layout_options is not None:
            if hasattr(layout_options, "model_spec"):
                from docling.datamodel.layout_model_specs import (
                    DOCLING_LAYOUT_EGRET_LARGE,
                    DOCLING_LAYOUT_EGRET_MEDIUM,
                    DOCLING_LAYOUT_EGRET_XLARGE,
                    DOCLING_LAYOUT_HERON,
                    DOCLING_LAYOUT_HERON_101,
                    DOCLING_LAYOUT_V2,
                    LayoutModelConfig,
                )
                layout_options.model_spec = DOCLING_LAYOUT_EGRET_XLARGE


        # Enable image generation for pages and tables if the API supports it
        if hasattr(pipeline_opts, "generate_page_images"):
            pipeline_opts.generate_page_images = True
        if hasattr(pipeline_opts, "generate_table_images"):
            pipeline_opts.generate_table_images = True
        if hasattr(pipeline_opts, "generate_picture_images"):
            pipeline_opts.generate_picture_images = True
        if hasattr(pipeline_opts, "do_formula_enrichment"):
            pipeline_opts.do_formula_enrichment = True
        if hasattr(pipeline_opts, "do_code_enrichment"):
            pipeline_opts.do_code_enrichment = True
        if hasattr(pipeline_opts, "images_scale"):
            pipeline_opts.images_scale = 2.0
        if hasattr(pipeline_opts, "code_formula_options"):
            preset_name= 'codeformulav2' #preset_name: Name of the preset to use ('codeformulav2' or 'granite_docling')
            code_formula_options = CodeFormulaVlmOptions.from_preset(preset_name)
            pipeline_opts.code_formula_options = code_formula_options

        # Construct the converter with custom pipeline options.
        # Docling v2.x APIs differ across minor versions; we try the
        # modern format-options API first, then fall back gracefully.
        try:
            from docling.backend.docling_parse_backend import (
                DoclingParseDocumentBackend,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import FormatOption
            from docling.pipeline.standard_pdf_pipeline import (
                StandardPdfPipeline,
            )

            fmt_options = {
                InputFormat.PDF: FormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseDocumentBackend,
                    pipeline_options=pipeline_opts,
                )
            }
            self._converter = DocumentConverter(format_options=fmt_options)
        except (ImportError, Exception):
            # Fallback: try the older constructor signature or default
            try:
                self._converter = DocumentConverter(pipeline_options=pipeline_opts)
            except TypeError:
                self._converter = DocumentConverter()

        return self._converter

    # ------------------------------------------------------------------
    #  Conversion
    # ------------------------------------------------------------------

    def convert(self, source: str | Path) -> Any:
        """Convert a PDF *source* into a Docling document object.

        Parameters
        ----------
        source:
            Path to a PDF file.

        Returns
        -------
        DoclingDocument
            The Docling document object (type varies with Docling version).

        Raises
        ------
        ImportError
            If Docling is not installed.
        """
        converter = self._get_converter()
        result = converter.convert(str(source))
        return result.document

    # ------------------------------------------------------------------
    #  Version
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Docling package version, or ``"unknown"`` if not detectable."""
        try:
            from importlib.metadata import version as _pkg_version

            return _pkg_version("docling")
        except (ImportError, Exception):
            return "unknown"


# ===================================================================
#  JSON-safe serialisation helper
# ===================================================================


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-serialisable objects for JSON output.

    Handles ``UUID``, ``Path``, ``datetime``, and falls back to ``str()``
    for unknown types.
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    # Try standard JSON serialisation; fall back to string
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ===================================================================
#  Core extraction
# ===================================================================


def extract_with_docling(
    doc: DocumentSchema,
    settings: PipelineSettings | None = None,
    adapter: DoclingAdapter | None = None,
) -> Any:
    """Load the source PDF for *doc* and convert it with Docling.

    The source PDF is expected at ``{raw_dir}/{doc_id}/source.pdf``.

    Parameters
    ----------
    doc:
        An ingested document record.
    settings:
        Pipeline configuration.  Defaults when ``None``.
    adapter:
        Docling adapter to use.  A new default adapter is created when
        ``None``.

    Returns
    -------
    DoclingDocument
        The raw Docling document object.

    Raises
    ------
    FileNotFoundError
        If the source PDF file does not exist on disk.
    """
    settings = settings or PipelineSettings()
    adapter = adapter or DoclingAdapter(settings=settings)

    source_pdf = settings.raw_dir / str(doc.doc_id) / "source.pdf"
    if not source_pdf.is_file():
        raise FileNotFoundError(
            f"Source PDF not found for document {doc.doc_id}: {source_pdf}"
        )

    logger.info("Extracting document %s with Docling …", doc.doc_id)
    dl_doc = adapter.convert(source_pdf)
    logger.info("Extraction complete for document %s", doc.doc_id)
    return dl_doc


# ===================================================================
#  Persistence
# ===================================================================


def _extract_dict(dl_doc: Any) -> dict[str, Any]:
    """Return a dictionary representation of ``dl_doc``.

    Tries ``export_to_dict()`` first, then falls back to plain ``dict()``,
    and finally to a stub dict.
    """
    export = getattr(dl_doc, "export_to_dict", None)
    if callable(export):
        return export()
    if isinstance(dl_doc, dict):
        return dl_doc
    as_dict = getattr(dl_doc, "dict", None)
    if callable(as_dict):
        return as_dict()
    return {"_fallback_note": f"Unrecognised document type: {type(dl_doc).__name__}"}


def _extract_markdown(dl_doc: Any) -> str:
    """Return a markdown representation of ``dl_doc``.

    Tries ``export_to_markdown()`` first, with a fallback note.
    """
    export = getattr(dl_doc, "export_to_markdown", None)
    if callable(export):
        return export()
    return (
        "# NOTE: Markdown export not available\n\n"
        "The Docling document object in use does not provide an "
        "``export_to_markdown()`` method."
    )


def _save_page_images(dl_doc: Any, pages_dir: Path) -> int:
    """Save page images from *dl_doc* into *pages_dir*.

    Best-effort: if image extraction APIs are absent the function returns 0
    without raising.

    Returns the number of successfully saved page images.
    """
    saved = 0
    pages = getattr(dl_doc, "pages", {})
    if isinstance(pages, dict):
        page_iter = pages.values()
    elif isinstance(pages, (list, tuple)):
        page_iter = pages
    else:
        return 0

    for page in page_iter:
        page_num = getattr(page, "page_num", saved + 1)
        try:
            # --- Path A: ImageRef via .image attribute ---
            image_ref = getattr(page, "image", None)
            if image_ref is not None:
                pil_image = getattr(image_ref, "pil_image", None)
                if pil_image is not None:
                    if callable(pil_image):
                        img = pil_image()
                    else:
                        img = pil_image
                    if img is not None:
                        img.save(str(pages_dir / f"page_{page_num}.png"))
                        saved += 1
                        continue

            # --- Path B: page.save(path) method ---
            save_method = getattr(page, "save", None)
            if callable(save_method):
                save_method(str(pages_dir / f"page_{page_num}.png"))
                saved += 1
                continue

            # --- Path C: direct PIL .image attribute ---
            img = getattr(page, "image", None)
            if img is not None and hasattr(img, "save"):
                img.save(str(pages_dir / f"page_{page_num}.png"))
                saved += 1
                continue

            # --- Path D: TableItem-like get_image method ---
            get_image_method = getattr(page, "get_image", None)
            if callable(get_image_method):
                img = get_image_method(dl_doc)
                if img is not None:
                    img.save(str(pages_dir / f"page_{page_num}.png"))
                    saved += 1
                    continue
        except (AttributeError, OSError, TypeError, ImportError) as exc:
            logger.debug("Could not save page image for page %s: %s", page_num, exc)

    return saved


def _save_picture_images(dl_doc: Any, pictures_dir: Path) -> int:
    """Save picture images from *dl_doc* into *pictures_dir*.

    Best-effort: if picture image APIs are absent the function returns 0
    without raising.

    Returns the number of successfully saved picture images.
    """
    saved = 0
    pictures = getattr(dl_doc, "pictures", [])
    if isinstance(pictures, dict):
        pic_iter = pictures.values()
    elif isinstance(pictures, (list, tuple)):
        pic_iter = pictures
    else:
        return 0

    for idx, pic in enumerate(pic_iter):
        try:
            # --- Path A: get_image method (similar to TableItem) ---
            get_image_method = getattr(pic, "get_image", None)
            if callable(get_image_method):
                img = get_image_method(dl_doc)
                if img is not None:
                    img.save(str(pictures_dir / f"picture_{idx}.png"))
                    saved += 1
                    continue

            # --- Path B: pic.save(path) ---
            save_method = getattr(pic, "save", None)
            if callable(save_method):
                save_method(str(pictures_dir / f"picture_{idx}.png"))
                saved += 1
                continue

            # --- Path C: direct PIL .image attribute ---
            img = getattr(pic, "image", None)
            if img is not None and hasattr(img, "save"):
                img.save(str(pictures_dir / f"picture_{idx}.png"))
                saved += 1
                continue
        except (AttributeError, OSError, TypeError, ImportError) as exc:
            logger.debug("Could not save picture image for picture %s: %s", idx, exc)

    return saved



def _save_table_images(dl_doc: Any, tables_dir: Path) -> int:
    """Save table images from *dl_doc* into *tables_dir*.

    Best-effort: returns 0 if table image APIs are absent.
    """
    saved = 0
    tables = getattr(dl_doc, "tables", [])
    if isinstance(tables, dict):
        table_iter = tables.values()
    elif isinstance(tables, (list, tuple)):
        table_iter = tables
    else:
        return 0

    for idx, table in enumerate(table_iter):
        try:
            # --- Path A: TableItem.get_image(doc) ---
            get_image_method = getattr(table, "get_image", None)
            if callable(get_image_method):
                img = get_image_method(dl_doc)
                if img is not None:
                    img.save(str(tables_dir / f"table_{idx}.png"))
                    saved += 1
                    continue

            # --- Path B: table.save(path) ---
            save_method = getattr(table, "save", None)
            if callable(save_method):
                save_method(str(tables_dir / f"table_{idx}.png"))
                saved += 1
                continue

            # --- Path C: direct PIL .image attribute ---
            img = getattr(table, "image", None)
            if img is not None and hasattr(img, "save"):
                img.save(str(tables_dir / f"table_{idx}.png"))
                saved += 1
                continue
        except (AttributeError, OSError, TypeError, ImportError) as exc:
            logger.debug("Could not save table image for table %s: %s", idx, exc)

    return saved


def persist_docling_outputs(
    doc: DocumentSchema,
    dl_doc: Any,
    settings: PipelineSettings | None = None,
    *,
    docling_version: str | None = None,
) -> dict[str, Path]:
    """Persist all raw Docling outputs for *doc* to disk.

    Creates the following structure under
    ``{raw_dir}/{doc_id}/docling/``:

    * ``output.json`` — Docling dict export (JSON-safe).
    * ``output.md`` — Markdown export.
    * ``pages/`` — Page images (best-effort).
    * ``tables/`` — Table images (best-effort).
    * ``pictures/`` — Picture images (best-effort).
    * ``assets/`` — Asset storage directory (empty for now).
    * ``version.txt`` — Installed Docling version.

    Parameters
    ----------
    doc:
        The document record.
    dl_doc:
        The Docling document object returned by
        :func:`extract_with_docling`.
    settings:
        Pipeline configuration.  Defaults when ``None``.
    docling_version:
        Version string to write into ``version.txt``.  Auto-detected when
        ``None``.

    Returns
    -------
    dict[str, Path]
        Mapping of logical output names to their paths on disk.
    """
    settings = settings or PipelineSettings()
    doc_dir = ensure_dir(settings.raw_dir / str(doc.doc_id))
    docling_dir = ensure_dir(doc_dir / "docling")
    pages_dir = ensure_dir(docling_dir / "pages")
    tables_dir = ensure_dir(docling_dir / "tables")
    pictures_dir = ensure_dir(docling_dir / "pictures")
    assets_dir = ensure_dir(docling_dir / "assets")

    outputs: dict[str, Path] = {}

    # ------------------------------------------------------------------
    #  JSON export
    # ------------------------------------------------------------------
    raw_dict = _extract_dict(dl_doc)
    safe_dict = _json_safe(raw_dict)
    json_path = docling_dir / "output.json"
    write_json(json_path, safe_dict)
    outputs["json"] = json_path

    # ------------------------------------------------------------------
    #  Markdown export
    # ------------------------------------------------------------------
    md_content = _extract_markdown(dl_doc)
    md_path = docling_dir / "output.md"
    atomic_write_text(md_path, md_content)
    outputs["markdown"] = md_path

    # ------------------------------------------------------------------
    #  Page images (best-effort)
    # ------------------------------------------------------------------
    _save_page_images(dl_doc, pages_dir)
    outputs["pages"] = pages_dir

    # ------------------------------------------------------------------
    #  Table images (best-effort)
    # ------------------------------------------------------------------
    _save_table_images(dl_doc, tables_dir)
    outputs["tables"] = tables_dir

    # ------------------------------------------------------------------
    #  Picture images (best-effort)
    # ------------------------------------------------------------------
    _save_picture_images(dl_doc, pictures_dir)
    outputs["pictures"] = pictures_dir

    # ------------------------------------------------------------------
    #  Assets directory
    # ------------------------------------------------------------------
    outputs["assets"] = assets_dir

    # ------------------------------------------------------------------
    #  Docling version
    # ------------------------------------------------------------------
    version_path = docling_dir / "version.txt"
    if docling_version is None:
        from importlib.metadata import version as _pkg_version

        try:
            docling_version = _pkg_version("docling")
        except (ImportError, Exception):
            docling_version = "unknown"
    atomic_write_text(version_path, docling_version + "\n")
    outputs["version"] = version_path

    logger.info(
        "Persisted Docling outputs for %s (JSON=%s, MD=%s, pages_dir=%s, "
        "tables_dir=%s, pictures_dir=%s, assets_dir=%s, version=%s)",
        doc.doc_id,
        json_path,
        md_path,
        pages_dir,
        tables_dir,
        pictures_dir,
        assets_dir,
        docling_version,
    )

    return outputs


# ===================================================================
#  Pipeline orchestration
# ===================================================================


def _update_manifest(
    doc_id: UUID, updates: dict[str, Any], settings: PipelineSettings
) -> None:
    """Merge *updates* into the manifest JSON for *doc_id*."""
    manifest_path = settings.raw_dir / str(doc_id) / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest: dict[str, Any] = {}

    manifest.update(updates)
    write_json(manifest_path, manifest)


def run_extraction_pipeline(
    doc: DocumentSchema,
    settings: PipelineSettings | None = None,
    adapter: DoclingAdapter | None = None,
    force: bool = True,
) -> tuple[Any, DocumentSchema]:
    """Orchestrate Docling extraction and output persistence for *doc*.

    Steps
    -----
    1. If ``force=False`` and all output files already exist, skip extraction
       and return the document with a ``"skipped"`` status.
    2. Otherwise run :func:`extract_with_docling`.
    3. Run :func:`persist_docling_outputs`.
    4. Update ``manifest.json`` with extraction status, timestamps, Docling
       version, and persisted output paths.
    5. Return an updated copy of *doc* with metadata reflecting the new
       status.

    Parameters
    ----------
    doc:
        An ingested document record.
    settings:
        Pipeline configuration.  Defaults when ``None``.
    adapter:
        Docling adapter.  Created automatically when ``None``.
    force:
        When ``False`` and outputs already exist, skip extraction.

    Returns
    -------
    DocumentSchema
        A frozen document copy with extraction metadata updated.

    Raises
    ------
    FileNotFoundError
        If the source PDF is missing.
    ImportError
        If Docling is not installed.
    Exception
        Any extraction failure is **also** recorded in the manifest (status
        ``"failed"``) before re-raising.
    """
    settings = settings or PipelineSettings()
    adapter = adapter or DoclingAdapter(settings=settings)
    doc_id = doc.doc_id

    # ------------------------------------------------------------------
    #  Skip-if-already-completed (force=False)
    # ------------------------------------------------------------------
    if not force:
        docling_dir = settings.raw_dir / str(doc_id) / "docling"
        expected = [
            docling_dir / "output.json",
            docling_dir / "output.md",
            docling_dir / "version.txt",
        ]
        if all(p.is_file() for p in expected):
            logger.info("Outputs already exist for %s — skipping extraction", doc_id)
            _update_manifest(
                doc_id,
                {
                    "processing_status": "ingested",
                    "extraction_status": "skipped",
                    "extraction_completed_at": None,
                },
                settings,
            )
            # Return doc with a note in metadata
            new_meta = doc.metadata.model_copy(
                update={
                    "processing_status": "ingested",
                    "custom": {
                        **doc.metadata.custom,
                        "extraction_status": "skipped",
                    },
                }
            )
            return "skipped", doc.model_copy(update={"metadata": new_meta})

    # ------------------------------------------------------------------
    #  Extract
    # ------------------------------------------------------------------
    started_at = datetime.now()
    try:
        dl_doc = extract_with_docling(doc, settings=settings, adapter=adapter)
        # print(dl_doc)
        persisted = persist_docling_outputs(
            doc, dl_doc, settings=settings, docling_version=adapter.version
        )

        docling_version = adapter.version
        completed_at = datetime.now()

        # Build output paths relative to doc directory (for manifest)
        output_paths = {
            name: str(
                p.relative_to(settings.raw_dir / str(doc_id))
                if p.is_relative_to(settings.raw_dir / str(doc_id))
                else p
            )
            for name, p in persisted.items()
        }

        manifest_updates = {
            "processing_status": "extracted",
            "extraction_status": "completed",
            "extraction_completed_at": completed_at.isoformat(),
            "docling_version": docling_version,
            "extraction_outputs": output_paths,
        }
        _update_manifest(doc_id, manifest_updates, settings)

        # Build updated DocumentSchema
        new_meta = doc.metadata.model_copy(
            update={
                "processing_status": "extracted",
                "processing_completed_at": completed_at,
                "extraction_version": docling_version,
                "custom": {
                    **doc.metadata.custom,
                    "extraction_status": "completed",
                    "docling_version": docling_version,
                    "output_paths": output_paths,
                },
            }
        )
        return dl_doc, doc.model_copy(update={"metadata": new_meta})
        # return dl_doc

    except Exception:
        failed_at = datetime.now()
        import traceback

        error_msg = traceback.format_exc()
        logger.error("Extraction failed for document %s:\n%s", doc_id, error_msg)

        _update_manifest(
            doc_id,
            {
                "processing_status": "ingested",
                "extraction_status": "failed",
                "extraction_error": error_msg,
                "extraction_completed_at": failed_at.isoformat(),
            },
            settings,
        )
        raise
