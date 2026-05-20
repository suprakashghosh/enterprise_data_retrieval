"""
Metadata models for documents, elements, and chunks.

Each metadata model carries optional fields for confidence scores, processing
timestamps, extraction versions, and an extensibility dictionary (``custom``)
for future expansion.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with a processed document.

    Attributes:
        source_format: Original format of the source file (e.g. ``"pdf"``).
        source_details: Arbitrary details about the source (e.g. PDF metadata).
        extraction_version: Version identifier of the extraction pipeline.
        processing_status: Current processing status (e.g. ``"ingested"``,
            ``"extracted"``, ``"chunked"``).
        confidence_score: Overall confidence score for the document extraction
            (0.0 – 1.0).
        processing_started_at: Timestamp when processing began.
        processing_completed_at: Timestamp when processing finished.
        custom: Extensible dictionary for non-standard metadata.
    """

    source_format: Optional[str] = None
    source_details: Optional[Dict[str, Any]] = None
    extraction_version: Optional[str] = None
    processing_status: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class ElementMetadata(BaseModel):
    """Metadata associated with a single document element.

    Attributes:
        confidence_score: Extraction confidence for this element (0.0 – 1.0).
        extraction_version: Version identifier of the extraction pipeline.
        is_inferred: Whether the element was inferred / reconstructed rather
            than directly extracted.
        font_info: Font-level details (family, size, weight, etc.) if available.
        custom: Extensible dictionary for non-standard metadata.
    """

    confidence_score: Optional[float] = None
    extraction_version: Optional[str] = None
    is_inferred: bool = False
    font_info: Optional[Dict[str, Any]] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Metadata associated with a generated chunk.

    Attributes:
        token_count: Number of tokens in the chunk (according to the
            configured tokenizer).
        embedding_model: Name of the embedding model used for semantic
            chunking, if applicable.
        chunking_version: Version identifier of the chunking pipeline.
        confidence_score: Quality / relevance confidence for this chunk
            (0.0 – 1.0).
        custom: Extensible dictionary for non-standard metadata.
    """

    token_count: Optional[int] = None
    embedding_model: Optional[str] = None
    chunking_version: Optional[str] = None
    confidence_score: Optional[float] = None
    custom: Dict[str, Any] = Field(default_factory=dict)
