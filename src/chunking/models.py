"""
Chunk metadata model and deterministic chunk ID generation.

``ChunkMetadata`` is the single source of truth for every chunk produced
by the pipeline.  It carries enough information for:
* vector database storage with filterable properties (Weaviate),
* graph database node/edge construction (Neo4j / Apache AGE), and
* multimodal embedding dispatch (text, image, LLM-generated description).
"""

from __future__ import annotations

import uuid
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Determistic chunk ID generation
# ---------------------------------------------------------------------------

CHUNK_ID_NAMESPACE = uuid.UUID("d7e8f9a0-1234-5678-9abc-def012345678")


def make_chunk_id(
    document_hash: str,
    sequence_number: int,
    chunk_types: list[str],
) -> str:
    """Generate a deterministic UUIDv5 for a chunk.

    Stable inputs:
    * ``document_hash`` — Docling binary hash of the source PDF (content-derived).
    * ``sequence_number`` — stable given same chunker parameters.
    * ``chunk_types`` — sorted label strings of all doc_items in this chunk,
      to disambiguate structurally different chunks at the same position.

    Returns a string representation of the UUID suitable for Weaviate.
    """
    name = (
        f"doc:{document_hash}"
        f"|seq:{sequence_number:06d}"
        f"|types:{','.join(sorted(chunk_types))}"
    )
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, name))


# ---------------------------------------------------------------------------
# Embedding type literal
# ---------------------------------------------------------------------------

EmbeddingTypeLiteral = Literal["text", "image", "textual_description"]


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk produced by the pipeline.

    Three embedding strategies are supported (see ``embedding_type``):
    * ``"text"`` — ``chunk_text`` is embedded directly (raw document content).
    * ``"image"`` — the image at ``image_uri`` is embedded by the multimodal
      encoder; ``chunk_text`` stores the caption for reference / display only.
    * ``"textual_description"`` — ``chunk_text`` contains an LLM-generated
      description of a visual element; this text is embedded.

    The model is **frozen** — once created, a ``ChunkMetadata`` instance
    should never be mutated.  Downstream stages (cross-reference resolution,
    embedding) produce *new* instances or augmented copies.
    """

    model_config = ConfigDict(frozen=True)

    # ── Identity ────────────────────────────────────────────────────

    chunk_id: str = Field(
        ...,
        description="Deterministic UUIDv5 — stable across pipeline re-runs for the same inputs.",
    )
    document_name: str = Field(
        ...,
        description="Original filename (e.g. '2502.04644v1.pdf').",
    )
    document_type: str = Field(
        ...,
        description="MIME type (e.g. 'application/pdf').",
    )
    document_hash: int = Field(
        ...,
        description="Docling binary hash of the source document.  Enables document-scoped search in Weaviate.",
    )

    # ── Embedding ───────────────────────────────────────────────────

    embedding_type: EmbeddingTypeLiteral = Field(
        ...,
        description=(
            "Controls which payload is passed to the multimodal embedding model. "
            "'text' -> chunk_text; 'image' -> image_uri; "
            "'textual_description' -> chunk_text (the LLM description)."
        ),
    )
    chunk_text: str = Field(
        ...,
        description=(
            "Text stored alongside the chunk.  For 'text' and 'textual_description' "
            "embeddings this is what gets embedded.  For 'image' embeddings it holds "
            "the caption for reference / display, but is NOT passed to the encoder."
        ),
    )

    # ── Structure ───────────────────────────────────────────────────

    chunk_types: List[str] = Field(
        default_factory=list,
        description="DocItemLabel strings for every doc_item in this chunk (e.g. ['text', 'picture', 'section_header']).",
    )
    section_path: str = Field(
        default="",
        description='Normalized section hierarchy, e.g. "Introduction > Background > Prior Work".  Empty for un-sectioned chunks.',
    )
    section_headings: List[str] = Field(
        default_factory=list,
        description="Raw heading strings from Docling's meta.headings (outermost first).  Preserved for provenance.",
    )
    page_numbers: List[int] = Field(
        default_factory=list,
        description="Deduplicated, sorted page numbers from all doc_items in this chunk.  Enables page-range filters in Weaviate.",
    )
    sequence_number: int = Field(
        ...,
        description="Global chunk order in reading sequence (0-indexed).  Used to derive prev/next edges at graph-construction time.",
    )

    # ── Visual elements ─────────────────────────────────────────────

    image_type: List[Literal["picture", "table"]] = Field(
        default_factory=list,
        description=(
            "Parallel list with image_uri / caption_text / caption_number. "
            "Each entry is 'picture' or 'table' for the visual element at the same "
            "index.  Empty list when the chunk contains no visual elements."
        ),
    )
    image_uri: List[str] = Field(
        default_factory=list,
        description=(
            "File paths or URIs of image assets in this chunk (parallel with image_type). "
            "For 'image' embedding chunks this is what the multimodal encoder reads. "
            "At retrieval time, URIs are passed to the LLM as visual context."
        ),
    )
    caption_text: List[str] = Field(
        default_factory=list,
        description=(
            "Raw caption strings for each visual element (parallel with image_type). "
            "May be empty strings when Docling cannot resolve a caption ref."
        ),
    )
    caption_number: List[str] = Field(
        default_factory=list,
        description=(
            "Extracted figure/table numbers from captions (parallel with image_type). "
            "E.g. ['3.2', '1a', 'IV'].  Populated via ``extract_caption_label()``. "
            "Enables exact-match retrieval ('show me Figure 3.2')."
        ),
    )

    # ── Graph linking ───────────────────────────────────────────────

    element_self_refs: List[str] = Field(
        default_factory=list,
        description="self_ref strings for all doc_items in this chunk (e.g. ['#/texts/0', '#/pictures/2']).  Used as native Docling identifiers for graph node creation.",
    )
    refers_to: List[str] = Field(
        default_factory=list,
        description="chunk_ids that this chunk explicitly references (populated by the cross-reference resolver).  Creates directed 'refers_to' edges in the graph DB.",
    )
    relates_to: List[str] = Field(
        default_factory=list,
        description="chunk_ids with semantic or structural relatedness (populated by section/similarity grouping).  Creates undirected 'relates_to' edges in the graph DB.",
    )

    # ── Quality signals ─────────────────────────────────────────────

    token_count: int = Field(
        default=0,
        description=(
            "Actual token count of chunk_text (via the configured tokenizer). "
            "Most text chunks cluster near MAX_TOKENS; visual chunks may be much "
            "shorter.  Used to filter near-empty chunks during retrieval."
        ),
    )
