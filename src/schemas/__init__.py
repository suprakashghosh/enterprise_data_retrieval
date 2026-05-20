"""
``src.schemas`` — Pydantic data models for the extraction and chunking
pipeline.

All public model classes and ID helpers are re-exported here so consumers
can import directly from ``src.schemas``:

    from src.schemas import DocumentSchema, ElementSchema, ChunkSchema
"""

from src.schemas.chunks import ChunkSchema, ChunkTypeLiteral
from src.schemas.document import DocumentSchema, PageSchema, SectionSchema
from src.schemas.elements import (
    ELEMENT_TYPE_LITERAL,
    CaptionSchema,
    ChartSchema,
    ElementSchema,
    FooterSchema,
    FootnoteSchema,
    FormulaSchema,
    GraphSchema,
    HeaderSchema,
    ImageSchema,
    ListBlockSchema,
    SectionHeaderSchema,
    TableSchema,
    TextBlockSchema,
)
from src.schemas.geometry import BoundingBox, CoordSystem, Point, Size
from src.schemas.id_gen import (
    CHUNK_NAMESPACE,
    DOC_NAMESPACE,
    ELEM_NAMESPACE,
    REL_NAMESPACE,
    make_chunk_id,
    make_doc_id,
    make_element_id,
    make_relationship_id,
)
from src.schemas.metadata import ChunkMetadata, DocumentMetadata, ElementMetadata
from src.schemas.relationships import (
    ALL_RELATIONSHIP_TYPES,
    RELATIONSHIP_TYPE_LITERAL,
    RelationshipSchema,
)

__all__ = [
    # --- geometry ---
    "BoundingBox",
    "CoordSystem",
    "Point",
    "Size",
    # --- metadata ---
    "DocumentMetadata",
    "ElementMetadata",
    "ChunkMetadata",
    # --- id generation ---
    "DOC_NAMESPACE",
    "ELEM_NAMESPACE",
    "CHUNK_NAMESPACE",
    "REL_NAMESPACE",
    "make_doc_id",
    "make_element_id",
    "make_chunk_id",
    "make_relationship_id",
    # --- elements ---
    "ELEMENT_TYPE_LITERAL",
    "ElementSchema",
    "TextBlockSchema",
    "TableSchema",
    "ImageSchema",
    "ChartSchema",
    "GraphSchema",
    "FormulaSchema",
    "CaptionSchema",
    "FootnoteSchema",
    "HeaderSchema",
    "FooterSchema",
    "ListBlockSchema",
    "SectionHeaderSchema",
    # --- relationships ---
    "RELATIONSHIP_TYPE_LITERAL",
    "ALL_RELATIONSHIP_TYPES",
    "RelationshipSchema",
    # --- chunks ---
    "ChunkTypeLiteral",
    "ChunkSchema",
    # --- document ---
    "PageSchema",
    "SectionSchema",
    "DocumentSchema",
]
