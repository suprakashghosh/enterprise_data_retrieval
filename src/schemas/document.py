"""
Top-level document model.

``DocumentSchema`` is the root object produced by the extraction and
normalisation pipeline.  It holds all pages, elements, chunks, and
relationships for a single source document.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.chunks import ChunkSchema
from src.schemas.elements import ElementSchema
from src.schemas.geometry import Size
from src.schemas.metadata import DocumentMetadata
from src.schemas.relationships import RelationshipSchema


class PageSchema(BaseModel):
    """A single page of a document.

    Attributes:
        page_num: 1-based page number.
        size: Physical dimensions of the page.
        element_ids: UUIDs of elements that appear on this page, in reading
            order.
    """

    model_config = ConfigDict(frozen=True)

    page_num: int = Field(..., ge=1, description="1-based page number")
    size: Optional[Size] = Field(default=None, description="Page dimensions")
    element_ids: List[UUID] = Field(
        default_factory=list,
        description="Element UUIDs on this page, in reading order",
    )


class SectionSchema(BaseModel):
    """A hierarchical section within a document.

    Attributes:
        section_id: Deterministic UUID for this section.
        section_path: Hierarchical path string (e.g. ``"3.2.1"``).
        title: Section title text.
        level: Depth level (1 = top-level section).
        parent_section_id: UUID of the parent section, or ``None`` for
            top-level sections.
        element_ids: UUIDs of elements belonging to this section, in
            reading order.
    """

    model_config = ConfigDict(frozen=True)

    section_id: UUID = Field(..., description="Deterministic section UUID")
    section_path: str = Field(
        ..., description="Hierarchical section path (e.g. '3.2.1')"
    )
    title: str = Field(default="", description="Section title text")
    level: int = Field(default=1, ge=1, description="Section depth level")
    parent_section_id: Optional[UUID] = Field(
        default=None, description="UUID of the parent section, if any"
    )
    element_ids: List[UUID] = Field(
        default_factory=list,
        description="Element UUIDs belonging to this section",
    )


class DocumentSchema(BaseModel):
    """Root model for a processed document.

    This is the primary output of the extraction and normalisation pipeline
    and the primary input to the chunking pipeline.

    Attributes:
        doc_id: Deterministic UUID derived from the file content hash.
        title: Document title (from filename, PDF metadata, or extracted).
        source_path: Original file path of the source document.
        file_hash: SHA-256 hex digest of the raw source file.
        page_count: Total number of pages.
        created_at: Timestamp when this document record was created.
        pages: Ordered list of page descriptors.
        elements: All extracted elements keyed by their UUID string.
        chunks: Generated chunks (empty until the chunking stage).
        relationships: All relationships between elements and chunks.
        metadata: Document-level metadata.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: UUID = Field(..., description="Deterministic document UUID")
    title: str = Field(default="", description="Document title")
    source_path: str = Field(..., description="Original source file path")
    file_hash: str = Field(..., description="SHA-256 hex digest of the source file")
    page_count: int = Field(..., ge=0, description="Total number of pages")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Record creation timestamp"
    )
    pages: List[PageSchema] = Field(
        default_factory=list, description="Page descriptors"
    )
    elements: Dict[str, ElementSchema] = Field(
        default_factory=dict,
        description="All elements keyed by their UUID string",
    )
    chunks: List[ChunkSchema] = Field(
        default_factory=list, description="Generated chunks"
    )
    relationships: List[RelationshipSchema] = Field(
        default_factory=list, description="All relationships"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Document-level metadata"
    )
