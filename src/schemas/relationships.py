"""
Typed relationship model for linking document elements and chunks.

Relationships form a directed graph over the element / chunk space and capture
structural containment, reading-order adjacency, semantic relatedness, and
domain-specific links (captioning, describing, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Relationship type literal – every link kind recognised by the pipeline.
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPE_LITERAL = Literal[
    "contains",
    "belongs_to",
    "relates_to",
    "refers_to",
    "describes",
    "follows",
    "precedes",
    "summarizes",
    "supports",
    "explains",
    "has_caption",
    "has_table",
    "has_image",
    "has_formula",
    "same_section_as",
    "nearby",
]

# All valid relationship types in a list so callers can iterate / validate.
ALL_RELATIONSHIP_TYPES: List[str] = list(
    RELATIONSHIP_TYPE_LITERAL.__args__  # type: ignore[attr-defined]
)


class RelationshipSchema(BaseModel):
    """A typed, directed relationship between two entities (elements or chunks).

    Attributes:
        relationship_id: Deterministic UUID for this relationship.
        source_id: UUID of the source entity.
        target_id: UUID of the target entity.
        relationship_type: Semantic type of the relationship.
        metadata: Optional key-value payload for the relationship.
        weight: Numeric weight / strength (0.0 – 1.0).  Defaults to 1.0
            for hard structural links.
    """

    model_config = ConfigDict(frozen=True)

    relationship_id: UUID = Field(..., description="Deterministic relationship UUID")
    source_id: UUID = Field(..., description="UUID of the source entity")
    target_id: UUID = Field(..., description="UUID of the target entity")
    relationship_type: RELATIONSHIP_TYPE_LITERAL = Field(
        ..., description="Semantic type of the relationship"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key-value metadata for this relationship",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strength / confidence weight (0.0 – 1.0)",
    )
