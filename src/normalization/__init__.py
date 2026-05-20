"""
``src.normalization`` — Normalisation, element registry, and hierarchy
reconstruction.

Converts raw Docling output into the project's internal schema,
preserves page numbers, bounding boxes, reading order, section
assignments, captions, and layout proximity.  Reconstructs the
document hierarchy and assigns section paths to every element.

Public API
----------
::

    from src.normalization import (
        DOCLING_TYPE_TO_INTERNAL_TYPE,
        ElementRegistry,
        normalize_document,
        preserve_proximity,
    )
"""

from src.normalization.docling_normalizer import (
    DOCLING_TYPE_TO_INTERNAL_TYPE,
    ElementRegistry,
    normalize_document,
    preserve_proximity,
)

__all__: list[str] = [
    "DOCLING_TYPE_TO_INTERNAL_TYPE",
    "ElementRegistry",
    "normalize_document",
    "preserve_proximity",
]
