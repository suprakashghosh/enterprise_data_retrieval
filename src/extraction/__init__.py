"""
``src.extraction`` — Docling extraction and output persistence.

Wraps Docling's ``DocumentConverter``, runs extraction on ingested PDFs,
and persists all raw outputs (JSON, markdown, page images, table images)
for audit and re-processing.

Public API
----------
::

    from src.extraction import (
        DoclingAdapter,
        extract_with_docling,
        persist_docling_outputs,
        run_extraction_pipeline,
    )
"""

from src.extraction.docling_extractor import (
    DoclingAdapter,
    extract_with_docling,
    persist_docling_outputs,
    run_extraction_pipeline,
)

__all__ = [
    "DoclingAdapter",
    "extract_with_docling",
    "persist_docling_outputs",
    "run_extraction_pipeline",
]
