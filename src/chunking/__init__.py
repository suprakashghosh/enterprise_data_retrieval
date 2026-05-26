"""
``src.chunking`` — Docling-based chunking pipeline.

Wraps Docling's ``HybridChunker`` to produce typed ``ChunkMetadata``
objects.  Each chunk carries enough information for:
* vector database storage with filterable properties (Weaviate),
* graph database node / edge construction (Neo4j / Apache AGE), and
* multimodal embedding dispatch (text, image, LLM-generated description).

The pipeline is three steps:

1. ``build_chunk_metadata_list()`` — chunk the document and extract metadata.
2. ``enrich_visual_chunks()`` — add image and textual-description chunks for
   every picture / table.
3. (future) cross-reference resolver — populate ``refers_to`` between chunks.
"""

from src.chunking.models import ChunkMetadata, make_chunk_id
from src.chunking.docling_chunker import (
    build_chunk_metadata_list,
    chunk_document,
    create_hybrid_chunker,
    extract_chunk_metadata,
)
from src.chunking.visual_enricher import (
    enrich_visual_chunks,
    generate_all_image_descriptions,
)

__all__ = [
    "ChunkMetadata",
    "make_chunk_id",
    "build_chunk_metadata_list",
    "chunk_document",
    "create_hybrid_chunker",
    "extract_chunk_metadata",
    "enrich_visual_chunks",
    "generate_all_image_descriptions",
]
