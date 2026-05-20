"""
``src.ingestion`` — Document ingestion and raw storage.

Handles copying source PDFs into an immutable raw store, computing file
hashes, creating document records, and writing ingestion manifests.

Public API
----------
::

    from src.ingestion import IngestionSource, ingest_pdf, batch_ingest, get_document
"""

from src.ingestion.ingestor import (
    IngestionSource,
    batch_ingest,
    get_document,
    ingest_pdf,
)

__all__ = [
    "IngestionSource",
    "ingest_pdf",
    "batch_ingest",
    "get_document",
]
