"""
Deterministic UUIDv5 generation for documents, elements, chunks, and
relationships.

Re-running the pipeline on the same input always produces the same
identifiers, enabling idempotent incremental processing.
"""

from __future__ import annotations

import uuid

# ---------------------------------------------------------------------------
# UUIDv5 Namespace Constants
# ---------------------------------------------------------------------------
# Each namespace is itself a deterministic UUIDv5 derived from a stable
# human-readable string under the DNS namespace.  This keeps every ID in
# the project traceable back to a common root.

DOC_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "docling-project-doc")
"""Namespace for document IDs."""

ELEM_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "docling-project-elem")
"""Namespace for element IDs."""

CHUNK_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "docling-project-chunk")
"""Namespace for chunk IDs."""

REL_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "docling-project-rel")
"""Namespace for relationship IDs."""


# ---------------------------------------------------------------------------
# ID Helpers
# ---------------------------------------------------------------------------


def make_doc_id(file_hash: str) -> uuid.UUID:
    """Generate a deterministic document ID from the SHA-256 file hash.

    Args:
        file_hash: Hex-encoded SHA-256 digest of the raw document file.

    Returns:
        A UUIDv5 unique to this document's content.
    """
    return uuid.uuid5(DOC_NAMESPACE, file_hash)


def make_element_id(
    doc_id: uuid.UUID,
    page_num: int,
    reading_order: int,
    element_type: str,
) -> uuid.UUID:
    """Generate a deterministic element ID.

    The input string is ``{doc_id}:{page_num}:{reading_order}:{element_type}``,
    ensuring that an element extracted from the same document, page, position,
    and type always receives the same ID.

    Args:
        doc_id: UUID of the parent document.
        page_num: 1-based page number.
        reading_order: Ordinal position within the page.
        element_type: Type label (e.g. ``"text_block"``, ``"table"``).

    Returns:
        A UUIDv5 unique to this element's position and type.
    """
    name = f"{doc_id}:{page_num}:{reading_order}:{element_type}"
    return uuid.uuid5(ELEM_NAMESPACE, name)


def make_chunk_id(
    doc_id: uuid.UUID,
    chunk_type: str,
    section_path: str,
    element_count: int,
) -> uuid.UUID:
    """Generate a deterministic chunk ID.

    The input string is
    ``{doc_id}:{chunk_type}:{section_path}:{element_count}``.

    Args:
        doc_id: UUID of the parent document.
        chunk_type: Type of chunk (e.g. ``"hierarchical"``, ``"semantic"``).
        section_path: Hierarchical section path (e.g. ``"3.2.1"``).
        element_count: Number of element references in the chunk.

    Returns:
        A UUIDv5 unique to this chunk.
    """
    name = f"{doc_id}:{chunk_type}:{section_path}:{element_count}"
    return uuid.uuid5(CHUNK_NAMESPACE, name)


def make_relationship_id(
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    relationship_type: str,
) -> uuid.UUID:
    """Generate a deterministic relationship ID.

    The input string is ``{source_id}:{target_id}:{relationship_type}``.

    Args:
        source_id: UUID of the source element / chunk.
        target_id: UUID of the target element / chunk.
        relationship_type: Type label (e.g. ``"contains"``, ``"follows"``).

    Returns:
        A UUIDv5 unique to this directed relationship.
    """
    name = f"{source_id}:{target_id}:{relationship_type}"
    return uuid.uuid5(REL_NAMESPACE, name)
