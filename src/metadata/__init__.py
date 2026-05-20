"""
``src.metadata`` — Metadata generation and image/table processing.

Processes visual elements (images, charts, graphs), saves asset files,
classifies visual types, and enriches element metadata for downstream
chunking and retrieval.

Public API
----------
::

    from src.metadata import (
        save_visual_asset,
        classify_visual_type,
        prepare_visual_metadata,
        process_images,
    )
"""

from src.metadata.image_processor import (
    classify_visual_type,
    prepare_visual_metadata,
    process_images,
    save_visual_asset,
)

__all__: list[str] = [
    "classify_visual_type",
    "prepare_visual_metadata",
    "process_images",
    "save_visual_asset",
]
