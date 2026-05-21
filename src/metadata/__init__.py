"""
``src.metadata`` — Metadata generation, relationship generation, and
image/table processing.

Processes visual elements (images, charts, graphs), saves asset files,
classifies visual types, and enriches element metadata for downstream
chunking and retrieval.  Also generates all typed relationships between
document elements.

Public API
----------
::

    from src.metadata import (
        # Image processing
        save_visual_asset,
        classify_visual_type,
        prepare_visual_metadata,
        process_images,
        # Relationship generation
        generate_all_relationships,
        generate_structural_relationships,
        generate_sequential_relationships,
        generate_caption_relationships,
        generate_spatial_relationships,
        generate_section_relationships,
        generate_reference_relationships,
        generate_descriptive_relationships,
        deduplicate_relationships,
        generate_relationship_summary,
    )
"""

from src.metadata.image_processor import (
    classify_visual_type,
    prepare_visual_metadata,
    process_images,
    save_visual_asset,
)
from src.metadata.relationship_generator import (
    deduplicate_relationships,
    generate_all_relationships,
    generate_caption_relationships,
    generate_descriptive_relationships,
    generate_reference_relationships,
    generate_relationship_summary,
    generate_section_relationships,
    generate_sequential_relationships,
    generate_spatial_relationships,
    generate_structural_relationships,
)

__all__: list[str] = [
    "classify_visual_type",
    "prepare_visual_metadata",
    "process_images",
    "save_visual_asset",
    "generate_all_relationships",
    "generate_structural_relationships",
    "generate_sequential_relationships",
    "generate_caption_relationships",
    "generate_spatial_relationships",
    "generate_section_relationships",
    "generate_reference_relationships",
    "generate_descriptive_relationships",
    "deduplicate_relationships",
    "generate_relationship_summary",
]
