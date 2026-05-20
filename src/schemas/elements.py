"""
Typed element models for document content.

Every content item extracted from a document is represented as an
``ElementSchema`` subclass.  The ``element_type`` field uses a ``Literal``
string that uniquely identifies the kind of content, making it safe to
use in discriminated unions and serialisation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.geometry import BoundingBox
from src.schemas.metadata import ElementMetadata

# ---------------------------------------------------------------------------
# Element type literal – the complete set of recognised content types.
# ---------------------------------------------------------------------------

ELEMENT_TYPE_LITERAL = Literal[
    "text_block",
    "table",
    "image",
    "chart",
    "graph",
    "formula",
    "caption",
    "footnote",
    "header",
    "footer",
    "list_block",
    "section_header",
]

# ---------------------------------------------------------------------------
# Base element
# ---------------------------------------------------------------------------


class ElementSchema(BaseModel):
    """Base model for every document element.

    All content items share these common fields.  Subclasses add specialised
    fields for their content type.

    Attributes:
        element_id: Deterministic UUID for this element.
        doc_id: UUID of the parent document.
        page_num: 1-based page number where this element appears.
        bbox: Bounding box on the page.
        reading_order: Ordinal position in reading order within the page.
        section_path: Hierarchical section path (e.g. ``"3.2.1"``).
        element_type: Discriminator literal identifying the concrete type.
        content: Primary text content (markdown, plain text, etc.).
        metadata: Type-specific metadata.
    """

    model_config = ConfigDict(frozen=True)

    element_id: UUID = Field(..., description="Deterministic element UUID")
    doc_id: UUID = Field(..., description="Parent document UUID")
    page_num: int = Field(..., ge=1, description="1-based page number")
    bbox: BoundingBox = Field(..., description="Bounding box on the page")
    reading_order: int = Field(..., ge=0, description="Reading order index within page")
    section_path: str = Field(default="", description="Hierarchical section path")
    element_type: ELEMENT_TYPE_LITERAL = Field(
        ..., description="Concrete element type discriminator"
    )
    content: str = Field(default="", description="Text / markdown content")
    metadata: ElementMetadata = Field(
        default_factory=ElementMetadata, description="Element-level metadata"
    )


# ---------------------------------------------------------------------------
# Concrete element subclasses
# ---------------------------------------------------------------------------


class TextBlockSchema(ElementSchema):
    """A paragraph or running-text block."""

    element_type: Literal["text_block"] = "text_block"
    language: Optional[str] = Field(default=None, description="Detected language code")


class TableSchema(ElementSchema):
    """A structured table with multiple output representations."""

    element_type: Literal["table"] = "table"
    markdown: str = Field(default="", description="Markdown table representation")
    html: str = Field(default="", description="HTML table representation")
    json_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON (list-of-rows) representation of the table",
    )
    row_count: int = Field(default=0, ge=0, description="Number of data rows")
    col_count: int = Field(default=0, ge=0, description="Number of columns")
    headers: List[str] = Field(default_factory=list, description="Column header labels")
    summary: str = Field(default="", description="Human-readable plain-text summary")
    is_spanning: bool = Field(
        default=False, description="Whether this table spans multiple pages"
    )
    span_group_id: Optional[str] = Field(
        default=None, description="Group ID for multi-page spanned tables"
    )


class ImageSchema(ElementSchema):
    """An embedded image / figure (photograph, diagram, etc.)."""

    element_type: Literal["image"] = "image"
    asset_path: Optional[str] = Field(
        default=None, description="Path to the saved image asset"
    )
    thumbnail_path: Optional[str] = Field(
        default=None, description="Path to the thumbnail image"
    )
    caption: Optional[str] = Field(default=None, description="Extracted caption text")
    description: Optional[str] = Field(
        default=None, description="Human-written or machine-generated description"
    )
    visual_type: Optional[str] = Field(
        default=None,
        description="Visual classification (chart, diagram, photograph, ...)",
    )
    image_properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Image properties (width, height, aspect ratio, ...)",
    )
    vision_description: Optional[str] = Field(
        default=None,
        description="Vision-language model description (extensible placeholder)",
    )


class ChartSchema(ElementSchema):
    """A chart / plot extracted from the document."""

    element_type: Literal["chart"] = "chart"
    asset_path: Optional[str] = Field(
        default=None, description="Path to the saved chart asset"
    )
    thumbnail_path: Optional[str] = Field(
        default=None, description="Path to the thumbnail image"
    )
    caption: Optional[str] = Field(default=None, description="Extracted caption text")
    description: Optional[str] = Field(
        default=None, description="Human-written or machine-generated description"
    )
    visual_type: Optional[str] = Field(
        default=None, description="Visual classification (bar, line, pie, ...)"
    )
    image_properties: Optional[Dict[str, Any]] = Field(
        default=None, description="Image properties (width, height, ...)"
    )
    vision_description: Optional[str] = Field(
        default=None,
        description="Vision-language model description (extensible placeholder)",
    )


class GraphSchema(ElementSchema):
    """A graph / network diagram extracted from the document."""

    element_type: Literal["graph"] = "graph"
    asset_path: Optional[str] = Field(
        default=None, description="Path to the saved graph asset"
    )
    thumbnail_path: Optional[str] = Field(
        default=None, description="Path to the thumbnail image"
    )
    caption: Optional[str] = Field(default=None, description="Extracted caption text")
    description: Optional[str] = Field(
        default=None, description="Human-written or machine-generated description"
    )
    visual_type: Optional[str] = Field(
        default=None,
        description="Visual classification (network, tree, flowchart, ...)",
    )
    image_properties: Optional[Dict[str, Any]] = Field(
        default=None, description="Image properties (width, height, ...)"
    )
    vision_description: Optional[str] = Field(
        default=None,
        description="Vision-language model description (extensible placeholder)",
    )


class FormulaSchema(ElementSchema):
    """A mathematical formula, either inline or displayed."""

    element_type: Literal["formula"] = "formula"
    latex: str = Field(default="", description="LaTeX representation")
    text_approximation: str = Field(
        default="", description="Plain-text approximation of the formula"
    )
    formula_type: Literal["inline", "display"] = Field(
        default="display",
        description="Whether the formula is inline or displayed on its own line",
    )
    variables: List[str] = Field(
        default_factory=list,
        description="Variable/symbol names mentioned in the formula",
    )


class CaptionSchema(ElementSchema):
    """A caption that describes another element (table, figure, etc.)."""

    element_type: Literal["caption"] = "caption"
    parent_element_id: Optional[UUID] = Field(
        default=None, description="UUID of the element this caption describes"
    )


class FootnoteSchema(ElementSchema):
    """A footnote at the bottom of a page."""

    element_type: Literal["footnote"] = "footnote"
    footnote_id: Optional[str] = Field(
        default=None,
        description="Footnote marker / identifier (e.g. superscript number)",
    )


class HeaderSchema(ElementSchema):
    """A page header (repeated at the top of pages)."""

    element_type: Literal["header"] = "header"


class FooterSchema(ElementSchema):
    """A page footer (repeated at the bottom of pages)."""

    element_type: Literal["footer"] = "footer"


class ListBlockSchema(ElementSchema):
    """An ordered or unordered list."""

    element_type: Literal["list_block"] = "list_block"
    items: List[str] = Field(
        default_factory=list, description="Individual list item texts"
    )
    ordered: bool = Field(default=False, description="Whether the list is ordered")


class SectionHeaderSchema(ElementSchema):
    """A section heading with an optional number."""

    element_type: Literal["section_header"] = "section_header"
    level: int = Field(default=1, ge=1, description="Heading level (1 = top-level)")
    section_number: Optional[str] = Field(
        default=None, description="Numbered section path (e.g. '3.2.1')"
    )
