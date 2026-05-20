"""
Reconstruct document hierarchy and assign hierarchical section paths.

Identifies section headers (by explicit type or heuristic patterns),
builds a section tree using a stack-based algorithm, assigns every
element a ``section_path`` string, and creates ``contains`` relationships
for document → section, section → subsection, and section → element.

Public API
----------
::

    from src.normalization import build_hierarchy, assign_section_paths
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.normalization.docling_normalizer import ElementRegistry
from src.schemas import (
    BoundingBox,
    DocumentSchema,
    ElementSchema,
    HeaderSchema,
    FooterSchema,
    PageSchema,
    RelationshipSchema,
    SectionHeaderSchema,
    SectionSchema,
    make_element_id,
    make_relationship_id,
)

logger = logging.getLogger(__name__)

# ===================================================================
#  Constants
# ===================================================================

# UUIDv5 namespace for section IDs (stable deterministic).
_SECTION_NAMESPACE: uuid.UUID = uuid.uuid5(
    uuid.NAMESPACE_DNS, "docling-project-section"
)

# Keywords that indicate a section header when no number is present.
_FRONT_MATTER_KEYWORDS: frozenset = frozenset(
    {
        "abstract",
        "keywords",
        "executive summary",
        "table of contents",
        "list of figures",
        "list of tables",
        "nomenclature",
        "notation",
        "preface",
        "foreword",
        "acknowledgments",
        "acknowledgements",
    }
)

_BACK_MATTER_KEYWORDS: frozenset = frozenset(
    {
        "references",
        "bibliography",
        "appendix",
        "appendices",
        "index",
    }
)

_SECTION_KEYWORDS: frozenset = (
    _FRONT_MATTER_KEYWORDS
    | _BACK_MATTER_KEYWORDS
    | frozenset(
        {
            "introduction",
            "conclusion",
            "discussion",
            "results",
            "methodology",
            "method",
            "related work",
            "background",
            "summary",
            "future work",
            "limitations",
            "experiments",
            "evaluation",
            "implementation",
            "overview",
            "motivation",
            "approach",
        }
    )
)

# Element types that should never be treated as sections.
_NON_SECTION_TYPES: frozenset = frozenset(
    {
        "header",
        "footer",
        "footnote",
        "caption",
    }
)


# ===================================================================
#  Section header detection helpers
# ===================================================================


def _extract_numbering(text: str) -> Optional[Tuple[int, str, str]]:
    """Try to extract a numbered section prefix from *text*.

    Returns ``(level, number_str, title)`` on success, or ``None``.

    Examples that match::

        1. Introduction       -> (1, "1", "Introduction")
        1.1 Background        -> (2, "1.1", "Background")
        1.2.3 Details         -> (3, "1.2.3", "Details")
        Section 2: Method     -> (1, "2", "Method")
        Chapter 3 Results     -> (1, "3", "Results")
        Appendix A            -> (1, "A", "Appendix A")
        A.1 Additional        -> (2, "A.1", "Additional")
    """
    stripped = text.strip()
    if not stripped:
        return None

    # "Section X: Title" or "Section X Title"
    m = re.match(r"^Section\s+(\d+(?:\.\d+)*)\s*[:.]?\s*(.*)$", stripped, re.IGNORECASE)
    if m:
        num = m.group(1)
        title = m.group(2).strip() or stripped
        return (num.count(".") + 1, num, title)

    # "Chapter X: Title"
    m = re.match(r"^Chapter\s+(\d+)\s*[:.]?\s*(.*)$", stripped, re.IGNORECASE)
    if m:
        num = m.group(1)
        title = m.group(2).strip() or stripped
        return (1, num, title)

    # "Appendix A: Title" or "Appendix A Title"
    m = re.match(r"^Appendix\s+([A-Z])\s*[:.]?\s*(.*)$", stripped, re.IGNORECASE)
    if m:
        letter = m.group(1)
        title = m.group(2).strip()
        if not title:
            title = f"Appendix {letter}"
        return (1, letter, title)

    # "A.1 Title"  (appendix subsection)
    m = re.match(r"^([A-Z])\.(\d+)\s+(.+)$", stripped)
    if m:
        num = f"{m.group(1)}.{m.group(2)}"
        title = m.group(3).strip()
        return (2, num, title)

    # Standard numeric "1.", "1.1", "1.2.3" etc.
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", stripped)
    if m:
        num = m.group(1)
        title = m.group(2).strip()
        level = num.count(".") + 1
        return (level, num, title)

    # Bare number "1" or "1." with no following text (edge case).
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s*$", stripped)
    if m:
        num = m.group(1)
        level = num.count(".") + 1
        return (level, num, stripped)

    return None


def _match_known_keyword(text: str) -> bool:
    """Return ``True`` if *text* is a known section keyword.

    The text must match the keyword exactly (case-insensitive), or be
    the keyword followed only by punctuation (e.g. ``"Results:"``,
    ``"Abstract."``).  This avoids matching regular prose like
    ``"Results text."`` or ``"Introduction to the topic"``.
    """
    stripped = text.strip().lower()
    # Exact match
    if stripped in _SECTION_KEYWORDS:
        return True
    # Keyword followed by nothing but punctuation/whitespace
    for kw in _SECTION_KEYWORDS:
        if stripped.startswith(kw):
            suffix = stripped[len(kw) :].strip()
            if not suffix or all(c in ".:;!?" for c in suffix):
                return True
    return False


def _is_likely_page_number(text: str) -> bool:
    """Heuristic: short text that is purely digits or a page-range."""
    stripped = text.strip()
    if not stripped:
        return True
    # e.g. "42" or "42-43" or "Page 1"
    if re.match(r"^\d+\s*$", stripped):
        return True
    if re.match(r"^\d+\s*[-–]\s*\d+\s*$", stripped):
        return True
    if re.match(r"^Page\s+\d+", stripped, re.IGNORECASE):
        return True
    return False


def _detect_section_header(
    elem: ElementSchema,
) -> Optional[Tuple[int, Optional[str], str]]:
    """Detect whether an element is a section header.

    Returns ``(level, section_number_or_None, title)`` if *elem* should
    be treated as a section header, or ``None`` otherwise.

    Three detection strategies are tried in order:

    1. **Explicit type**: ``section_header`` with a ``SectionHeaderSchema``.
    2. **Heuristic numbering**: ``text_block`` whose content matches a
       numbered pattern.
    3. **Known keywords**: ``text_block`` whose content matches a known
       front/back-matter or common section keyword.
    """
    # Skip non-section types
    if elem.element_type in _NON_SECTION_TYPES:
        return None

    # Skip page-number-like text blocks
    if elem.element_type == "text_block" and _is_likely_page_number(elem.content):
        return None

    # Strategy 1: explicit section_header type
    if elem.element_type == "section_header":
        if isinstance(elem, SectionHeaderSchema):
            level = max(1, elem.level)
            number = elem.section_number
            title = elem.content or ""
            return (level, number, title)
        return (1, None, elem.content or "")

    # Strategy 2: heuristic numbering
    if elem.element_type == "text_block":
        result = _extract_numbering(elem.content)
        if result is not None:
            return result

        # Strategy 3: known keywords
        if _match_known_keyword(elem.content):
            title = elem.content.strip()
            return (1, None, title)

    return None


# ===================================================================
#  Section ID generation
# ===================================================================


def _make_section_id(doc_id: uuid.UUID, section_path: str, title: str) -> uuid.UUID:
    """Deterministic UUIDv5 for a section.

    Input: ``{doc_id}:{section_path}:{title_lower}``
    """
    name = f"{doc_id}:{section_path}:{title.strip().lower()}"
    return uuid.uuid5(_SECTION_NAMESPACE, name)


# ===================================================================
#  Section path determination
# ===================================================================


class _SectionFrame:
    """Tracking information for an open section on the stack."""

    __slots__ = ("section_path", "section_id", "level", "title")

    def __init__(
        self,
        section_path: str,
        section_id: uuid.UUID,
        level: int,
        title: str,
    ) -> None:
        self.section_path = section_path
        self.section_id = section_id
        self.level = level
        self.title = title


def _determine_section_path(
    stack: List[_SectionFrame],
    next_child: Dict[str, int],
    level: int,
    number: Optional[str],
    title: str,
) -> str:
    """Determine the section path for a new section.

    - Numbered sections always use their *number* as the path.
    - Unnumbered top-level sections get ``0.{counter}``.
    - Unnumbered child sections get ``{parent}.{next_child}``.
    """
    if number is not None:
        # Numbered path — use as-is.
        path = number
        # Update the parent's child counter so unnumbered siblings
        # don't collide.
        _update_next_child_for_numbered(next_child, stack, path, level)
        return path

    # Unnumbered: find parent context.
    # Pop stack frames whose level >= current level.
    adjusted_stack = _adjusted_stack(stack, level)
    parent_path = adjusted_stack[-1].section_path if adjusted_stack else ""

    child_num = next_child.get(parent_path, 1)
    child_path = f"{parent_path}.{child_num}" if parent_path else f"0.{child_num}"
    next_child[parent_path] = child_num + 1
    return child_path


def _adjusted_stack(stack: List[_SectionFrame], level: int) -> List[_SectionFrame]:
    """Return the stack after popping frames with ``level >= *level*``."""
    result = list(stack)
    while result and result[-1].level >= level:
        result.pop()
    return result


def _update_next_child_for_numbered(
    next_child: Dict[str, int],
    stack: List[_SectionFrame],
    path: str,
    level: int,
) -> None:
    """Ensure that the parent's next-child counter is past this number.

    For nested numbered paths (e.g. ``"1.2"`` under parent ``"1"``),
    the parent's child counter is bumped so unnumbered siblings get
    the next logical number.  Top-level numbered sections (e.g.
    ``"1"``) do **not** consume the root unnumbered counter because
    unnumbered top-level sections use synthetic ``0.x`` paths.
    """
    parts = path.split(".")
    if len(parts) >= 2:
        parent_path = ".".join(parts[:-1])
        child_num = int(parts[-1])
        current = next_child.get(parent_path, 1)
        if child_num >= current:
            next_child[parent_path] = child_num + 1
    # Top-level numbered sections intentionally do **not** advance
    # next_child[""] — unnumbered sections use independent 0.x paths.


# ===================================================================
#  Main builder
# ===================================================================


def build_hierarchy(
    doc: DocumentSchema,
    registry: Optional[ElementRegistry] = None,
) -> DocumentSchema:
    """Reconstruct the document hierarchy and assign section paths.

    Iterates all elements in reading order, detects section headers by
    explicit type or heuristic, builds a section tree using a stack of
    open sections, assigns a ``section_path`` (e.g. ``"3.2.1"``) to
    every element, and creates ``contains`` relationships.

    Args:
        doc: The document to process (frozen model — an updated copy is
            returned).
        registry: An existing ``ElementRegistry``, or ``None`` to build
            one from ``doc.elements``.

    Returns:
        A new ``DocumentSchema`` with updated elements, relationships,
        and section data stored in ``metadata.custom["sections"]``.
    """
    # ------------------------------------------------------------------
    #  1. Build / use registry
    # ------------------------------------------------------------------
    if registry is None:
        registry = ElementRegistry()
        for elem in doc.elements.values():
            registry.add(elem)

    # ------------------------------------------------------------------
    #  2. State
    # ------------------------------------------------------------------
    stack: List[_SectionFrame] = []  # open sections (ancestor chain)
    next_child: Dict[str, int] = defaultdict(
        lambda: 1
    )  # parent_path -> next child number
    all_sections: List[SectionSchema] = []  # completed sections
    element_section_map: Dict[uuid.UUID, str] = {}  # elem_id -> section_path
    section_element_ids: Dict[uuid.UUID, List[uuid.UUID]] = defaultdict(list)
    seen_relationship_ids: set[uuid.UUID] = set()
    new_relationships: List[RelationshipSchema] = []

    # Helper to add a relationship if not already present.
    def _add_rel(source: uuid.UUID, target: uuid.UUID, rtype: str) -> None:
        rid = make_relationship_id(source, target, rtype)
        if rid not in seen_relationship_ids:
            seen_relationship_ids.add(rid)
            new_relationships.append(
                RelationshipSchema(
                    relationship_id=rid,
                    source_id=source,
                    target_id=target,
                    relationship_type=rtype,  # type: ignore[arg-type]
                    weight=1.0,
                )
            )

    # Also carry forward existing relationships (deduplicated).
    for rel in doc.relationships:
        seen_relationship_ids.add(rel.relationship_id)
    new_relationships.extend(doc.relationships)

    # ------------------------------------------------------------------
    #  3. Iterate elements in reading order
    # ------------------------------------------------------------------
    for elem in registry.iter_in_reading_order():
        elem_id = elem.element_id
        elem_type = elem.element_type

        # Skip non-section decorative elements entirely for section detection
        # but still assign them empty section path.
        if elem_type in _NON_SECTION_TYPES:
            element_section_map[elem_id] = ""
            continue

        # Detect if this element is a section header.
        detected = _detect_section_header(elem)

        if detected is not None:
            level, number, title = detected

            # Pop stack to correct depth (same-level sections close their
            # predecessors).
            while stack and stack[-1].level >= level:
                stack.pop()

            # Determine section path.
            path = _determine_section_path(stack, next_child, level, number, title)

            # Compute parent section ID (top of adjusted stack).
            adjusted = _adjusted_stack(stack, level)
            parent_id: Optional[uuid.UUID] = (
                adjusted[-1].section_id if adjusted else None
            )

            # Create section object.
            section_id = _make_section_id(doc.doc_id, path, title)
            section = SectionSchema(
                section_id=section_id,
                section_path=path,
                title=title,
                level=level,
                parent_section_id=parent_id,
                element_ids=[],
            )
            all_sections.append(section)

            # Push onto stack.
            stack.append(_SectionFrame(path, section_id, level, title))

            # Section header element belongs to itself.
            element_section_map[elem_id] = path
            section_element_ids[section_id].append(elem_id)

            # Create containment relationship with parent section.
            if parent_id is not None:
                _add_rel(parent_id, section_id, "contains")

        else:
            # Regular (non-header) element.
            if stack:
                active = stack[-1]
                element_section_map[elem_id] = active.section_path
                section_element_ids[active.section_id].append(elem_id)
            else:
                element_section_map[elem_id] = ""

    # ------------------------------------------------------------------
    #  4. Update element_ids in SectionSchema objects & store in metadata
    # ------------------------------------------------------------------
    updated_section_dicts: List[Dict[str, Any]] = []
    for sec in all_sections:
        sec_elem_ids = section_element_ids.get(sec.section_id, [])
        # Rematerialize as dict since SectionSchema is frozen and we
        # need to update element_ids.
        sec_dict = sec.model_dump()
        sec_dict["element_ids"] = sec_elem_ids
        updated_section_dicts.append(sec_dict)

    # ------------------------------------------------------------------
    #  5. Update element section_paths (frozen model copies)
    # ------------------------------------------------------------------
    updated_elements: Dict[str, ElementSchema] = {}
    for elem_key, elem in doc.elements.items():
        new_path = element_section_map.get(elem.element_id, elem.section_path)
        if new_path != elem.section_path:
            elem = elem.model_copy(update={"section_path": new_path})
        updated_elements[elem_key] = elem

    # ------------------------------------------------------------------
    #  6. Create containment relationships for section -> element
    #     Skip the first element_id in each section (the section header
    #     element that opened it) — its relationship to the section is
    #     already captured as a label, not as a separate contained item.
    # ------------------------------------------------------------------
    for sec_dict in updated_section_dicts:
        sec_id: uuid.UUID = sec_dict["section_id"]  # already a UUID from model_dump
        elem_ids: list = sec_dict.get("element_ids", [])
        # First element is the section header — skip it
        for elem_id in elem_ids[1:]:
            _add_rel(sec_id, elem_id, "contains")

    # ------------------------------------------------------------------
    #  7. Create document -> top-level section relationships
    # ------------------------------------------------------------------
    doc_id = doc.doc_id
    for sec_dict in updated_section_dicts:
        parent_id_raw = sec_dict.get("parent_section_id")
        if parent_id_raw is None:
            sec_id: uuid.UUID = sec_dict["section_id"]  # already a UUID
            _add_rel(doc_id, sec_id, "contains")

    # ------------------------------------------------------------------
    #  8. Store sections in metadata.custom
    # ------------------------------------------------------------------
    new_custom = dict(doc.metadata.custom)
    new_custom["sections"] = updated_section_dicts
    new_metadata = doc.metadata.model_copy(update={"custom": new_custom})

    # ------------------------------------------------------------------
    #  9. Return updated document
    # ------------------------------------------------------------------
    return doc.model_copy(
        update={
            "elements": updated_elements,
            "relationships": new_relationships,
            "metadata": new_metadata,
        }
    )


# ===================================================================
#  Convenience wrapper
# ===================================================================


def assign_section_paths(
    doc: DocumentSchema,
    registry: Optional[ElementRegistry] = None,
) -> DocumentSchema:
    """Convenience wrapper around :func:`build_hierarchy`.

    Ensures all elements receive ``section_path`` values based on the
    reconstructed document hierarchy.

    Args:
        doc: The document to process.
        registry: Optional pre-built ``ElementRegistry``.

    Returns:
        An updated ``DocumentSchema``.
    """
    return build_hierarchy(doc, registry=registry)
