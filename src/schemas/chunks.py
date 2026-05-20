"""
Chunk model representing a segment of document content produced by the
chunking pipeline.

Chunks aggregate one or more document elements, optionally decorate them
with relationship metadata, and are the unit of retrieval in downstream
graph / vector stores.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.metadata import ChunkMetadata
from src.schemas.relationships import RelationshipSchema

# ---------------------------------------------------------------------------
# Chunk type literal
# ---------------------------------------------------------------------------

ChunkTypeLiteral = Literal["hierarchical", "semantic", "cluster", "mixed"]


class ChunkSchema(BaseModel):
    """A content chunk produced by the chunking pipeline.

    Attributes:
        chunk_id: Deterministic UUID for this chunk.
        doc_id: UUID of the parent document.
        chunk_type: Strategy used to produce the chunk (hierarchical,
            semantic, cluster, or mixed).
        content: Text content of the chunk, potentially with embedded
            element references.
        element_refs: Ordered list of element UUIDs included in this chunk.
        section_path: Hierarchical section path covering the chunk content.
        page_range: ``(first_page, last_page)`` tuple.
        relationships: Relationships that originate from or point to
            this chunk.
        metadata: Chunk-level metadata (token count, embedding model, etc.).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID = Field(..., description="Deterministic chunk UUID")
    doc_id: UUID = Field(..., description="Parent document UUID")
    chunk_type: ChunkTypeLiteral = Field(
        ..., description="Strategy used to produce this chunk"
    )
    content: str = Field(
        default="",
        description="Text content with optional embedded element references",
    )
    element_refs: List[UUID] = Field(
        default_factory=list,
        description="Ordered list of contained element UUIDs",
    )
    section_path: str = Field(
        default="",
        description="Hierarchical section path covering the chunk",
    )
    page_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="(first_page, last_page) inclusive page range",
    )
    relationships: List[RelationshipSchema] = Field(
        default_factory=list,
        description="Relationships linking this chunk to other entities",
    )
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata,
        description="Chunk-level metadata",
    )
