"""
Relationship generation engine.

Produces all typed relationships between document elements as defined
in the pipeline's relationship taxonomy.  Each relationship type has a
dedicated generator function; ``generate_all_relationships`` composes
them and deduplicates the results.

Public API
----------
::

    from src.metadata import (
        generate_all_relationships,
        generate_structural_relationships,
        generate_sequential_relationships,
        generate_caption_relationships,
        generate_spatial_relationships,
        generate_section_relationships,
        generate_reference_relationships,
        generate_descriptive_relationships,
        deduplicate_relationships,
        generate_relationship_summary,
    )
"""

from __future__ import annotations

import copy
import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.normalization.docling_normalizer import ElementRegistry
from src.schemas import (
    BoundingBox,
    CaptionSchema,
    ChartSchema,
    DocumentSchema,
    ElementSchema,
    FormulaSchema,
    GraphSchema,
    ImageSchema,
    RelationshipSchema,
    TableSchema,
    make_relationship_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Visual element types that can be described or referenced.
_VISUAL_TYPES: frozenset = frozenset({"table", "image", "chart", "graph", "formula"})

# Reference regex patterns for ``refers_to`` detection.
# Captures the number (and optional letter suffix for figures).
_REF_PATTERNS: Dict[str, re.Pattern] = {
    "table": re.compile(r"(?:Table|Tbl\.?)\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "figure": re.compile(
        r"(?:Figure|Fig\.?)\s*(\d+(?:\.\d+)?(?:[a-z])?)", re.IGNORECASE
    ),
    "equation": re.compile(
        r"(?:Equation|Eq\.?|Eqn\.?)\s*(\d+(?:\.\d+)?)", re.IGNORECASE
    ),
}

# Reading-order window for descriptive relationships.
_DESCRIBES_ORDER_WINDOW: int = 2

# Proximity threshold for spatial relationships (normalised distance).
_SPATIAL_THRESHOLD: float = 0.08


# ===================================================================
#  Internal helpers
# ===================================================================


def _bbox_center_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Euclidean distance between centres of two normalized bounding boxes."""
    c1_x = (bbox1.left + bbox1.right) / 2.0
    c1_y = (bbox1.top + bbox1.bottom) / 2.0
    c2_x = (bbox2.left + bbox2.right) / 2.0
    c2_y = (bbox2.top + bbox2.bottom) / 2.0
    return ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5


def _extract_label_number(text: str) -> Optional[str]:
    """Extract a numbered label (e.g. 'Table 3' -> '3') from text.

    Returns the first matching number string, or ``None``.
    """
    if not text:
        return None
    for pattern in _REF_PATTERNS.values():
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


# ===================================================================
#  1.  Structural relationships  (belongs_to inverse of contains)
# ===================================================================


def generate_structural_relationships(
    doc: DocumentSchema,
    registry: ElementRegistry,  # noqa: ARG001
) -> List[RelationshipSchema]:
    """Generate ``belongs_to`` relationships as inverses of ``contains``.

    Scans existing ``contains`` relationships on *doc* and creates a
    matching ``belongs_to`` edge in the opposite direction for each.
    """
    relationships: List[RelationshipSchema] = []
    seen: Set[Tuple[uuid.UUID, uuid.UUID, str]] = set()

    for rel in doc.relationships:
        if rel.relationship_type != "contains":
            continue
        key = (rel.target_id, rel.source_id, "belongs_to")
        if key in seen:
            continue
        seen.add(key)
        relationships.append(
            RelationshipSchema(
                relationship_id=make_relationship_id(
                    rel.target_id, rel.source_id, "belongs_to"
                ),
                source_id=rel.target_id,
                target_id=rel.source_id,
                relationship_type="belongs_to",
                metadata=copy.deepcopy(rel.metadata),
                weight=rel.weight,
            )
        )

    return relationships


# ===================================================================
#  2.  Sequential relationships  (follows / precedes)
# ===================================================================


def generate_sequential_relationships(
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate ``follows`` and ``precedes`` between consecutive elements.

    Elements on the same page that are adjacent in reading order
    receive:
    - ``follows``: the later element **follows** the earlier one.
    - ``precedes``: the earlier element **precedes** the later one.
    """
    relationships: List[RelationshipSchema] = []
    seen: Set[Tuple[uuid.UUID, uuid.UUID, str]] = set()

    # Group by page, sorted by reading order globally.
    page_groups: Dict[int, List[ElementSchema]] = {}
    for elem in registry.iter_in_reading_order():
        page_groups.setdefault(elem.page_num, []).append(elem)

    for page_num in sorted(page_groups):
        elems = page_groups[page_num]
        for i in range(len(elems) - 1):
            a, b = elems[i], elems[i + 1]

            # follows: b follows a
            key_f = (b.element_id, a.element_id, "follows")
            if key_f not in seen:
                seen.add(key_f)
                relationships.append(
                    RelationshipSchema(
                        relationship_id=make_relationship_id(
                            b.element_id, a.element_id, "follows"
                        ),
                        source_id=b.element_id,
                        target_id=a.element_id,
                        relationship_type="follows",
                        metadata={"page_num": page_num},
                        weight=1.0,
                    )
                )

            # precedes: a precedes b
            key_p = (a.element_id, b.element_id, "precedes")
            if key_p not in seen:
                seen.add(key_p)
                relationships.append(
                    RelationshipSchema(
                        relationship_id=make_relationship_id(
                            a.element_id, b.element_id, "precedes"
                        ),
                        source_id=a.element_id,
                        target_id=b.element_id,
                        relationship_type="precedes",
                        metadata={"page_num": page_num},
                        weight=1.0,
                    )
                )

    return relationships


# ===================================================================
#  3.  Caption relationships  (has_caption)
# ===================================================================


def generate_caption_relationships(
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate ``has_caption`` edges from visual parent to caption.

    Scans for ``CaptionSchema`` elements and creates ``has_caption``
    from their visual parent (identified via ``parent_element_id``)
    to the caption itself.
    """
    relationships: List[RelationshipSchema] = []
    seen: Set[uuid.UUID] = set()

    for elem in registry.iter_in_reading_order():
        if not isinstance(elem, CaptionSchema):
            continue
        parent_id = elem.parent_element_id
        if parent_id is None:
            continue

        rid = make_relationship_id(parent_id, elem.element_id, "has_caption")
        if rid not in seen:
            seen.add(rid)
            relationships.append(
                RelationshipSchema(
                    relationship_id=rid,
                    source_id=parent_id,
                    target_id=elem.element_id,
                    relationship_type="has_caption",
                    metadata={"page_num": elem.page_num},
                    weight=1.0,
                )
            )

    return relationships


# ===================================================================
#  4.  Spatial relationships  (nearby)
# ===================================================================


def generate_spatial_relationships(
    registry: ElementRegistry,
    threshold: float = _SPATIAL_THRESHOLD,
) -> List[RelationshipSchema]:
    """Generate ``nearby`` relationships based on spatial proximity.

    Two elements on the same page whose bounding-box centres are within
    *threshold* (normalised distance) receive a ``nearby`` edge stored
    in canonical direction (lower UUID first).
    """
    relationships: List[RelationshipSchema] = []
    seen_pairs: Set[Tuple[uuid.UUID, uuid.UUID]] = set()

    # Group by page
    page_groups: Dict[int, List[ElementSchema]] = {}
    for elem in registry.iter_in_reading_order():
        page_groups.setdefault(elem.page_num, []).append(elem)

    for page_num in sorted(page_groups):
        elems = page_groups[page_num]
        for i, a in enumerate(elems):
            for b in elems[i + 1 :]:
                if a.element_id == b.element_id:
                    continue
                dist = _bbox_center_distance(a.bbox, b.bbox)
                if dist > threshold:
                    continue
                # Canonical pair: smaller UUID first
                if a.element_id < b.element_id:
                    p1, p2 = a.element_id, b.element_id
                    src, tgt = a.element_id, b.element_id
                else:
                    p1, p2 = b.element_id, a.element_id
                    src, tgt = b.element_id, a.element_id

                if (p1, p2) in seen_pairs:
                    continue
                seen_pairs.add((p1, p2))

                relationships.append(
                    RelationshipSchema(
                        relationship_id=make_relationship_id(src, tgt, "nearby"),
                        source_id=src,
                        target_id=tgt,
                        relationship_type="nearby",
                        metadata={
                            "distance": round(dist, 6),
                            "page_num": page_num,
                        },
                        weight=round(1.0 - dist, 6),
                    )
                )

    return relationships


# ===================================================================
#  5.  Section relationships  (same_section_as — group-based)
# ===================================================================


def generate_section_relationships(
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate ``same_section_as`` group-based relationships.

    Elements sharing the same non-empty ``section_path`` are grouped
    into **one relationship per section path** (not O(n²) pairwise
    edges).  The relationship stores member IDs and the section path
    in its ``metadata`` dict.

    Returns one relationship per section path that has at least two
    members.
    """
    relationships: List[RelationshipSchema] = []

    # Group element IDs by section_path (non-empty only).
    section_groups: Dict[str, List[uuid.UUID]] = defaultdict(list)
    for elem in registry.iter_in_reading_order():
        path = elem.section_path
        if path:
            section_groups[path].append(elem.element_id)

    for section_path, member_ids in section_groups.items():
        if len(member_ids) < 2:
            continue  # Single-element sections do not need a group edge.

        # Deterministic group UUID derived from the section path.
        group_id = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"docling-project-samesection:{section_path}",
        )

        relationships.append(
            RelationshipSchema(
                relationship_id=make_relationship_id(
                    group_id, group_id, "same_section_as"
                ),
                source_id=group_id,
                target_id=group_id,
                relationship_type="same_section_as",
                metadata={
                    "group_id": section_path,
                    "member_ids": [str(mid) for mid in member_ids],
                    "element_count": len(member_ids),
                },
                weight=1.0,
            )
        )

    return relationships


# ===================================================================
#  6.  Reference relationships  (refers_to)
# ===================================================================


def _build_label_map(
    registry: ElementRegistry,
) -> Dict[str, List[ElementSchema]]:
    """Build a mapping from normalised labels to visual elements.

    Labels look like ``"table:1"``, ``"figure:2.1"``, ``"equation:5"``.
    Each key may map to multiple elements (rare but possible).
    """
    label_map: Dict[str, List[ElementSchema]] = defaultdict(list)

    # Build a reverse index: parent_element_id -> list of captions (O(n)).
    caption_parent_index: Dict[uuid.UUID, List[CaptionSchema]] = defaultdict(list)
    for elem in registry.iter_in_reading_order():
        if isinstance(elem, CaptionSchema) and elem.parent_element_id is not None:
            caption_parent_index[elem.parent_element_id].append(elem)

    for elem in registry.iter_in_reading_order():
        elem_type = elem.element_type
        if elem_type not in _VISUAL_TYPES:
            continue

        # Collect candidate text that may contain a label.
        candidates: List[str] = []

        if isinstance(elem, TableSchema):
            candidates = [elem.summary, elem.content, elem.markdown]
        elif isinstance(elem, (ImageSchema, ChartSchema, GraphSchema)):
            candidates = [elem.caption or "", elem.content]
        elif isinstance(elem, FormulaSchema):
            candidates = [elem.content, elem.text_approximation, elem.latex]

        # Also include captions whose parent is this element (O(1) lookup).
        for cap in caption_parent_index.get(elem.element_id, []):
            candidates.append(cap.content)

        for text in candidates:
            if not text:
                continue
            for pattern_name, pattern in _REF_PATTERNS.items():
                for m in pattern.finditer(text):
                    num = m.group(1).strip().lower()
                    label = f"{pattern_name}:{num}"
                    if elem not in label_map[label]:
                        label_map[label].append(elem)

        # Also try extracting a label directly from the element content.
        label_num = _extract_label_number(elem.content)
        if label_num:
            type_prefix = elem_type
            if elem_type in ("image", "chart", "graph"):
                type_prefix = "figure"
            elif elem_type == "formula":
                type_prefix = "equation"
            label = f"{type_prefix}:{label_num}"
            if elem not in label_map[label]:
                label_map[label].append(elem)

    return label_map


def generate_reference_relationships(
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate ``refers_to`` by scanning text for cross-references.

    Scans all text-block, list-block, and caption elements for
    patterns like "Table 1", "Figure 2.1", "Equation 5", "Fig. 3a"
    and links them to the matching visual element via the label map
    built by :func:`_build_label_map`.
    """
    relationships: List[RelationshipSchema] = []
    seen: Set[Tuple[uuid.UUID, uuid.UUID, str]] = set()

    label_map = _build_label_map(registry)

    for elem in registry.iter_in_reading_order():
        if elem.element_type not in ("text_block", "list_block", "caption"):
            continue
        text = elem.content
        if not text:
            continue

        for pattern_name, pattern in _REF_PATTERNS.items():
            for m in pattern.finditer(text):
                num = m.group(1).strip().lower()
                label = f"{pattern_name}:{num}"

                targets = label_map.get(label, [])
                for target in targets:
                    if target.element_id == elem.element_id:
                        continue
                    key = (elem.element_id, target.element_id, "refers_to")
                    if key in seen:
                        continue
                    seen.add(key)
                    relationships.append(
                        RelationshipSchema(
                            relationship_id=make_relationship_id(
                                elem.element_id, target.element_id, "refers_to"
                            ),
                            source_id=elem.element_id,
                            target_id=target.element_id,
                            relationship_type="refers_to",
                            metadata={
                                "page_num": elem.page_num,
                                "matched_pattern": pattern_name,
                                "matched_text": m.group(0),
                            },
                            weight=1.0,
                        )
                    )

    return relationships


# ===================================================================
#  7.  Descriptive relationships  (describes)
# ===================================================================


def generate_descriptive_relationships(
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate ``describes`` from text to nearby visual elements.

    A text element ``describes`` a visual element (table, image, chart,
    graph, formula) when:
    - They share the same non-empty ``section_path``.
    - They are on the same page.
    - The text is within ``_DESCRIBES_ORDER_WINDOW`` positions of the
      visual in reading order.
    """
    relationships: List[RelationshipSchema] = []
    seen: Set[Tuple[uuid.UUID, uuid.UUID, str]] = set()

    # Group by page, sorted by reading order.
    page_groups: Dict[int, List[ElementSchema]] = {}
    for elem in registry.iter_in_reading_order():
        page_groups.setdefault(elem.page_num, []).append(elem)

    for page_num in sorted(page_groups):
        elems = page_groups[page_num]
        for i, elem in enumerate(elems):
            if elem.element_type not in ("text_block", "list_block"):
                continue
            text_id = elem.element_id
            text_path = elem.section_path
            if not text_path:
                continue

            start = max(0, i - _DESCRIBES_ORDER_WINDOW)
            end = min(len(elems), i + _DESCRIBES_ORDER_WINDOW + 1)

            for j in range(start, end):
                if j == i:
                    continue
                candidate = elems[j]
                if candidate.element_type not in _VISUAL_TYPES:
                    continue
                if candidate.section_path != text_path:
                    continue

                pair = (text_id, candidate.element_id, "describes")
                if pair in seen:
                    continue
                seen.add(pair)

                relationships.append(
                    RelationshipSchema(
                        relationship_id=make_relationship_id(
                            text_id, candidate.element_id, "describes"
                        ),
                        source_id=text_id,
                        target_id=candidate.element_id,
                        relationship_type="describes",
                        metadata={
                            "page_num": page_num,
                            "reading_order_diff": abs(i - j),
                        },
                        weight=1.0,
                    )
                )

    return relationships


# ===================================================================
#  Deduplication
# ===================================================================


def deduplicate_relationships(
    rels: List[RelationshipSchema],
) -> List[RelationshipSchema]:
    """Remove duplicate and self-referencing relationships.

    Rules
    -----
    1. **Exact duplicates**: same (source_id, target_id, type).
    2. **Self-references**: ``source_id == target_id``  (skipped for
       ``same_section_as`` which intentionally self-references for
       group storage).
    3. **Symmetric types** (``nearby``): the reverse direction is
       also blocked after the canonical direction is seen.

    Args:
        rels: Input relationships (may contain duplicates).

    Returns:
        A new list with duplicates and self-references removed.
    """
    seen: Set[Tuple[uuid.UUID, uuid.UUID, str]] = set()
    result: List[RelationshipSchema] = []

    for rel in rels:
        # Self-reference — skip unless it is a group-based type.
        if (
            rel.source_id == rel.target_id
            and rel.relationship_type != "same_section_as"
        ):
            continue

        key = (rel.source_id, rel.target_id, rel.relationship_type)
        if key in seen:
            continue
        seen.add(key)

        # For symmetric types, also block the reverse direction.
        if rel.relationship_type == "nearby":
            rev_key = (rel.target_id, rel.source_id, rel.relationship_type)
            seen.add(rev_key)

        result.append(rel)

    return result


# ===================================================================
#  Composition
# ===================================================================


def generate_all_relationships(
    doc: DocumentSchema,
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate all typed relationships for a document.

    Composes every relationship generator and deduplicates the
    combined result.  The document's existing relationships are
    included as input so that the structural generator can find
    ``contains`` edges created by earlier stages.

    Args:
        doc: Document with elements and existing relationships.
        registry: Element registry with all elements.

    Returns:
        A deduplicated list of ``RelationshipSchema`` objects.
    """
    all_rels: List[RelationshipSchema] = list(doc.relationships)

    # Each generator returns new relationships without mutation.
    all_rels.extend(generate_structural_relationships(doc, registry))
    all_rels.extend(generate_sequential_relationships(registry))
    all_rels.extend(generate_caption_relationships(registry))
    all_rels.extend(generate_spatial_relationships(registry))
    all_rels.extend(generate_section_relationships(registry))
    all_rels.extend(generate_reference_relationships(registry))
    all_rels.extend(generate_descriptive_relationships(registry))

    return deduplicate_relationships(all_rels)


# ===================================================================
#  Summary
# ===================================================================


def generate_relationship_summary(
    rels: List[RelationshipSchema],
) -> Dict[str, int]:
    """Count relationships by type.

    Args:
        rels: List of relationships to summarise.

    Returns:
        A dict mapping relationship-type strings to counts, sorted by
        count descending (ties broken by type name for determinism).
    """
    counts: Dict[str, int] = {}
    for rel in rels:
        counts[rel.relationship_type] = counts.get(rel.relationship_type, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return dict(sorted_items)
