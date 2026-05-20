"""
``src`` — Enterprise Data Retrieval pipeline.

This package contains all pipeline stages:

- ``src.schemas`` — Pydantic data models (Sub-Task 1)
- ``src.ingestion`` — Document ingestion and raw storage
- ``src.extraction`` — Docling extraction and output persistence
- ``src.normalization`` — Normalisation and hierarchy reconstruction
- ``src.metadata`` — Metadata generation and image/table processing
- ``src.chunking`` — Hierarchical, semantic, and cluster chunking
- ``src.validation`` — Output quality validation
- ``src.utils`` — Shared logging, configuration, and file I/O utilities
"""

__all__: list[str] = []
