"""
Tests for ``src.metadata.relationship_generator`` (Sub-Task 11 — Define
and Generate Relationship Metadata).

Uses fakes throughout — no real Docling dependency.

Covers:
- Public imports.
- ``generate_structural_relationships`` — creates ``belongs_to`` from
  existing ``contains``.
- ``generate_sequential_relationships`` — ``follows`` / ``precedes``
  between consecutive same-page elements.
- ``generate_caption_relationships`` — ``has_caption`` from caption
  elements with ``parent_element_id``.
- ``generate_spatial_relationships`` — ``nearby`` using proximity
  heuristics.
- ``generate_section_relationships`` — group-based ``same_section_as``.
- ``generate_reference_relationships`` — ``refers_to`` with pattern
  matching (Table 1, Figure 2.1, Equation 5, Fig. 3a).
- ``generate_descriptive_relationships`` — ``describes`` from text
  near visual elements.
- ``deduplicate_relationships`` — exact duplicates, self-references,
  symmetric duplicate blocking.
- ``generate_all_relationships`` — composition + dedup.
- ``generate_relationship_summary`` — count by type.
- Deterministic output.
- Empty document / empty registry handling.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from src.metadata import (
    deduplicate_relationships,
    generate_all_relationships,
    generate_caption_relationships,
    generate_descriptive_relationships,
    generate_reference_relationships,
    generate_relationship_summary,
    generate_section_relationships,
    generate_sequential_relationships,
    generate_spatial_relationships,
    generate_structural_relationships,
)
from src.normalization import ElementRegistry
from src.schemas import (
    BoundingBox,
    CaptionSchema,
    ChartSchema,
    DocumentSchema,
    ElementSchema,
    FormulaSchema,
    GraphSchema,
    ImageSchema,
    ListBlockSchema,
    RelationshipSchema,
    TableSchema,
    TextBlockSchema,
    make_relationship_id,
)

# ===================================================================
#  Constants
# ===================================================================

_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

_VISUAL_ELEM_IDS = {
    "table": uuid.UUID("10000000-0000-0000-0000-000000000001"),
    "image": uuid.UUID("10000000-0000-0000-0000-000000000002"),
    "chart": uuid.UUID("10000000-0000-0000-0000-000000000003"),
    "graph": uuid.UUID("10000000-0000-0000-0000-000000000004"),
    "formula": uuid.UUID("10000000-0000-0000-0000-000000000005"),
}


def _uid(seed: int) -> uuid.UUID:
    """Deterministic UUID for testing, derived from an integer seed."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"test-rel-{seed}")


# ===================================================================
#  Helpers — synthetic elements and documents
# ===================================================================


def _bbox(
    left: float = 0.0,
    top: float = 0.0,
    right: float = 0.3,
    bottom: float = 0.1,
) -> BoundingBox:
    return BoundingBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        coord_system="normalized",
    )


def _make_doc(page_count: int = 5) -> DocumentSchema:
    return DocumentSchema(
        doc_id=_DOC_ID,
        title="Test Relationship Doc",
        source_path="/fake/rel_test.pdf",
        file_hash="rel1234",
        page_count=page_count,
        created_at=datetime(2025, 1, 1),
    )


def _make_text(
    elem_id: Optional[uuid.UUID] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    content: str = "Some text content.",
    bbox: Optional[BoundingBox] = None,
) -> TextBlockSchema:
    return TextBlockSchema(
        element_id=elem_id or _uid(100),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="text_block",
        content=content,
    )


def _make_list_block(
    elem_id: Optional[uuid.UUID] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    content: str = "• List item",
    bbox: Optional[BoundingBox] = None,
    items: Optional[List[str]] = None,
) -> ListBlockSchema:
    return ListBlockSchema(
        element_id=elem_id or _uid(101),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="list_block",
        content=content,
        items=items or ["List item"],
    )


def _make_table(
    elem_id: Optional[uuid.UUID] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    content: str = "Table data",
    summary: str = "",
    bbox: Optional[BoundingBox] = None,
) -> TableSchema:
    return TableSchema(
        element_id=elem_id or _VISUAL_ELEM_IDS["table"],
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="table",
        content=content,
        summary=summary,
    )


def _make_image(
    elem_id: Optional[uuid.UUID] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    caption: Optional[str] = None,
    bbox: Optional[BoundingBox] = None,
) -> ImageSchema:
    return ImageSchema(
        element_id=elem_id or _VISUAL_ELEM_IDS["image"],
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="image",
        content="",
        caption=caption,
    )


def _make_formula(
    elem_id: Optional[uuid.UUID] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    content: str = "E = mc^2",
    bbox: Optional[BoundingBox] = None,
) -> FormulaSchema:
    return FormulaSchema(
        element_id=elem_id or _VISUAL_ELEM_IDS["formula"],
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="formula",
        content=content,
    )


def _make_caption(
    elem_id: Optional[uuid.UUID] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    content: str = "Figure 1: A caption",
    parent_element_id: Optional[uuid.UUID] = None,
    bbox: Optional[BoundingBox] = None,
) -> CaptionSchema:
    return CaptionSchema(
        element_id=elem_id or _uid(300),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="caption",
        content=content,
        parent_element_id=parent_element_id,
    )


def _make_registry(elements: List[ElementSchema]) -> ElementRegistry:
    reg = ElementRegistry()
    for el in elements:
        reg.add(el)
    return reg


def _make_contains_rel(parent_id: uuid.UUID, child_id: uuid.UUID) -> RelationshipSchema:
    return RelationshipSchema(
        relationship_id=make_relationship_id(parent_id, child_id, "contains"),
        source_id=parent_id,
        target_id=child_id,
        relationship_type="contains",
        weight=1.0,
    )


# ===================================================================
#  1.  Public imports
# ===================================================================


class TestPublicImports:
    """All public API symbols are importable from ``src.metadata``."""

    def test_imports(self) -> None:
        assert callable(generate_all_relationships)
        assert callable(generate_structural_relationships)
        assert callable(generate_sequential_relationships)
        assert callable(generate_caption_relationships)
        assert callable(generate_spatial_relationships)
        assert callable(generate_section_relationships)
        assert callable(generate_reference_relationships)
        assert callable(generate_descriptive_relationships)
        assert callable(deduplicate_relationships)
        assert callable(generate_relationship_summary)

    def test_import_from_package(self) -> None:
        from src.metadata import (
            generate_all_relationships as f1,
            deduplicate_relationships as f2,
            generate_relationship_summary as f3,
        )

        assert callable(f1)
        assert callable(f2)
        assert callable(f3)


# ===================================================================
#  2.  Structural relationships
# ===================================================================


class TestStructuralRelationships:
    """``generate_structural_relationships`` creates ``belongs_to`` inverses."""

    def test_belongs_to_from_contains(self) -> None:
        """Each ``contains`` generates an inverse ``belongs_to``."""
        doc = _make_doc()
        reg = _make_registry([])
        parent, child = uuid.uuid4(), uuid.uuid4()
        # Add a contains relationship to the doc
        contains = _make_contains_rel(parent, child)
        doc = doc.model_copy(update={"relationships": [contains]})

        rels = generate_structural_relationships(doc, reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "belongs_to"
        assert rel.source_id == child
        assert rel.target_id == parent

    def test_no_contains_no_belongs_to(self) -> None:
        """No ``contains`` → no ``belongs_to``."""
        doc = _make_doc()
        reg = _make_registry([])
        rels = generate_structural_relationships(doc, reg)
        assert len(rels) == 0

    def test_ignores_non_contains(self) -> None:
        """Only ``contains`` is inverted; other types are ignored."""
        doc = _make_doc()
        reg = _make_registry([])
        a, b = uuid.uuid4(), uuid.uuid4()
        other = RelationshipSchema(
            relationship_id=make_relationship_id(a, b, "follows"),
            source_id=a,
            target_id=b,
            relationship_type="follows",
        )
        doc = doc.model_copy(update={"relationships": [other]})
        rels = generate_structural_relationships(doc, reg)
        assert len(rels) == 0


# ===================================================================
#  3.  Sequential relationships
# ===================================================================


class TestSequentialRelationships:
    """``generate_sequential_relationships`` creates ``follows``/``precedes``."""

    def test_two_elements_on_same_page(self) -> None:
        """Consecutive elements on the same page get both edges."""
        t1 = _make_text(elem_id=_uid(10), reading_order=0, page_num=1)
        t2 = _make_text(elem_id=_uid(11), reading_order=1, page_num=1)
        reg = _make_registry([t1, t2])
        rels = generate_sequential_relationships(reg)

        types = {(r.relationship_type, r.source_id, r.target_id) for r in rels}
        assert ("precedes", t1.element_id, t2.element_id) in types
        assert ("follows", t2.element_id, t1.element_id) in types
        assert len(rels) == 2

    def test_three_elements_sequential(self) -> None:
        """Three elements produce four relationships."""
        t1 = _make_text(elem_id=_uid(20), reading_order=0, page_num=1)
        t2 = _make_text(elem_id=_uid(21), reading_order=1, page_num=1)
        t3 = _make_text(elem_id=_uid(22), reading_order=2, page_num=1)
        reg = _make_registry([t1, t2, t3])
        rels = generate_sequential_relationships(reg)

        assert len(rels) == 4  # 2 precedes + 2 follows

    def test_no_edges_across_pages(self) -> None:
        """Elements on different pages are not linked."""
        t1 = _make_text(elem_id=_uid(30), reading_order=0, page_num=1)
        t2 = _make_text(elem_id=_uid(31), reading_order=0, page_num=2)
        reg = _make_registry([t1, t2])
        rels = generate_sequential_relationships(reg)
        assert len(rels) == 0

    def test_no_duplicates(self) -> None:
        """No duplicate sequential edges."""
        t1 = _make_text(elem_id=_uid(40), reading_order=0, page_num=1)
        t2 = _make_text(elem_id=_uid(41), reading_order=1, page_num=1)
        reg = _make_registry([t1, t2])
        rels = generate_sequential_relationships(reg)
        assert len(rels) == 2
        # Running twice should not create extras
        rels2 = generate_sequential_relationships(reg)
        assert len(rels2) == 2

    def test_single_element(self) -> None:
        """A single element produces no sequential edges."""
        t1 = _make_text(reading_order=0, page_num=1)
        reg = _make_registry([t1])
        rels = generate_sequential_relationships(reg)
        assert len(rels) == 0


# ===================================================================
#  4.  Caption relationships
# ===================================================================


class TestCaptionRelationships:
    """``generate_caption_relationships`` creates ``has_caption`` edges."""

    def test_caption_with_parent(self) -> None:
        """A caption with ``parent_element_id`` gets a ``has_caption`` edge."""
        parent_id = uuid.uuid4()
        cap = _make_caption(parent_element_id=parent_id)
        reg = _make_registry([cap])
        rels = generate_caption_relationships(reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "has_caption"
        assert rel.source_id == parent_id
        assert rel.target_id == cap.element_id

    def test_caption_without_parent(self) -> None:
        """A caption without ``parent_element_id`` produces no edge."""
        cap = _make_caption(parent_element_id=None)
        reg = _make_registry([cap])
        rels = generate_caption_relationships(reg)
        assert len(rels) == 0

    def test_no_duplicates(self) -> None:
        """No duplicate caption edges."""
        parent_id = uuid.uuid4()
        cap = _make_caption(parent_element_id=parent_id)
        reg = _make_registry([cap])
        rels = generate_caption_relationships(reg)
        assert len(rels) == 1
        rels2 = generate_caption_relationships(reg)
        assert len(rels2) == 1


# ===================================================================
#  5.  Spatial relationships
# ===================================================================


class TestSpatialRelationships:
    """``generate_spatial_relationships`` creates ``nearby`` edges."""

    def test_nearby_elements(self) -> None:
        """Elements within threshold get a ``nearby`` edge."""
        a = _make_text(
            elem_id=_uid(50),
            page_num=1,
            reading_order=0,
            bbox=_bbox(left=0.0, top=0.0, right=0.1, bottom=0.1),
        )
        b = _make_text(
            elem_id=_uid(51),
            page_num=1,
            reading_order=1,
            bbox=_bbox(left=0.05, top=0.0, right=0.15, bottom=0.1),
        )
        reg = _make_registry([a, b])
        rels = generate_spatial_relationships(reg, threshold=0.2)
        assert len(rels) == 1
        assert rels[0].relationship_type == "nearby"

    def test_far_elements_no_edge(self) -> None:
        """Elements far apart produce no edge."""
        a = _make_text(
            elem_id=_uid(60),
            page_num=1,
            reading_order=0,
            bbox=_bbox(left=0.0, top=0.0, right=0.01, bottom=0.01),
        )
        b = _make_text(
            elem_id=_uid(61),
            page_num=1,
            reading_order=1,
            bbox=_bbox(left=0.9, top=0.9, right=0.99, bottom=0.99),
        )
        reg = _make_registry([a, b])
        rels = generate_spatial_relationships(reg, threshold=0.2)
        assert len(rels) == 0

    def test_different_pages_no_edge(self) -> None:
        """Elements on different pages get no edge."""
        a = _make_text(elem_id=_uid(70), page_num=1)
        b = _make_text(elem_id=_uid(71), page_num=2)
        reg = _make_registry([a, b])
        rels = generate_spatial_relationships(reg)
        assert len(rels) == 0


# ===================================================================
#  6.  Section relationships  (same_section_as)
# ===================================================================


class TestSectionRelationships:
    """``generate_section_relationships`` creates group-based edges."""

    def test_group_based_storage(self) -> None:
        """Elements in the same section produce one group relationship."""
        t1 = _make_text(elem_id=_uid(80), reading_order=0, section_path="1.0")
        t2 = _make_text(elem_id=_uid(81), reading_order=1, section_path="1.0")
        reg = _make_registry([t1, t2])
        rels = generate_section_relationships(reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "same_section_as"
        assert "member_ids" in rel.metadata
        assert rel.metadata["element_count"] == 2
        assert rel.metadata["group_id"] == "1.0"
        # source_id == target_id for group relationships
        assert rel.source_id == rel.target_id

    def test_single_element_no_group(self) -> None:
        """A section with only one element produces no edge."""
        t1 = _make_text(reading_order=0, section_path="1.0")
        reg = _make_registry([t1])
        rels = generate_section_relationships(reg)
        assert len(rels) == 0

    def test_empty_section_path_ignored(self) -> None:
        """Elements with empty section paths are not grouped."""
        t1 = _make_text(reading_order=0, section_path="")
        t2 = _make_text(reading_order=1, section_path="")
        reg = _make_registry([t1, t2])
        rels = generate_section_relationships(reg)
        assert len(rels) == 0

    def test_multiple_sections(self) -> None:
        """Multiple sections each get their own group."""
        elements = [
            _make_text(elem_id=_uid(90), reading_order=0, section_path="1.0"),
            _make_text(elem_id=_uid(91), reading_order=1, section_path="1.0"),
            _make_text(elem_id=_uid(92), reading_order=2, section_path="2.0"),
            _make_text(elem_id=_uid(93), reading_order=3, section_path="2.0"),
        ]
        reg = _make_registry(elements)
        rels = generate_section_relationships(reg)
        assert len(rels) == 2
        paths = {r.metadata["group_id"] for r in rels}
        assert paths == {"1.0", "2.0"}

    def test_deterministic_group_id(self) -> None:
        """Same section path always produces the same group UUID."""
        t1 = _make_text(elem_id=_uid(94), reading_order=0, section_path="1.0")
        t2 = _make_text(elem_id=_uid(95), reading_order=1, section_path="1.0")
        reg1 = _make_registry([t1, t2])
        reg2 = _make_registry([t1, t2])
        r1 = generate_section_relationships(reg1)
        r2 = generate_section_relationships(reg2)
        assert r1[0].source_id == r2[0].source_id
        assert r1[0].relationship_id == r2[0].relationship_id


# ===================================================================
#  7.  Reference relationships  (refers_to)
# ===================================================================


class TestReferenceRelationships:
    """``generate_reference_relationships`` pattern matching."""

    def test_table_reference(self) -> None:
        """ "see Table 1" links to a table with label "Table 1"."""
        table = _make_table(summary="Table 1: Results")
        text = _make_text(
            content="The results in Table 1 show a clear trend.",
            reading_order=0,
        )
        reg = _make_registry([table, text])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "refers_to"
        assert rel.source_id == text.element_id
        assert rel.target_id == table.element_id

    def test_figure_reference(self) -> None:
        """ "as shown in Figure 2.1" links to an image with that label."""
        image = _make_image(caption="Figure 2.1: Architecture")
        text = _make_text(
            content="As shown in Figure 2.1, the architecture is complex.",
            reading_order=0,
        )
        reg = _make_registry([image, text])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 1
        assert rels[0].relationship_type == "refers_to"

    def test_fig_abbreviation(self) -> None:
        """ "Fig. 3a" links to an image with label "Fig. 3a"."""
        image = _make_image(caption="Fig. 3a: Details")
        text = _make_text(
            content="See Fig. 3a for more details.",
            reading_order=0,
        )
        reg = _make_registry([image, text])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 1
        assert rels[0].relationship_type == "refers_to"

    def test_equation_reference(self) -> None:
        """ "Equation 5" links to a formula with that label."""
        formula = _make_formula(content="Equation 5: E = mc^2")
        text = _make_text(
            content="From Equation 5 we can derive...",
            reading_order=0,
        )
        reg = _make_registry([formula, text])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 1
        assert rels[0].relationship_type == "refers_to"

    def test_no_reference_when_missing_target(self) -> None:
        """A reference with no matching target produces no edge."""
        text = _make_text(
            content="See Table 99 for details.",
            reading_order=0,
        )
        reg = _make_registry([text])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 0

    def test_no_false_positive_without_number(self) -> None:
        """ "see Table" without a number should not match."""
        table = _make_table(summary="Table 1: Results")
        text = _make_text(
            content="Please refer to the table for details.",
            reading_order=0,
        )
        reg = _make_registry([table, text])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 0

    def test_list_block_reference(self) -> None:
        """A list_block referencing "Table 1" links to the matching table."""
        table = _make_table(summary="Table 1: Results")
        lst = _make_list_block(
            content="• See Table 1 for the quarterly breakdown.",
            reading_order=0,
        )
        reg = _make_registry([table, lst])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "refers_to"
        assert rel.source_id == lst.element_id
        assert rel.target_id == table.element_id

    def test_list_block_reference_figure(self) -> None:
        """A list_block referencing "Figure 2.1" links to the matching image."""
        image = _make_image(caption="Figure 2.1: Architecture")
        lst = _make_list_block(
            content="• The system architecture is shown in Figure 2.1.",
            reading_order=0,
        )
        reg = _make_registry([image, lst])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 1
        assert rels[0].relationship_type == "refers_to"

    def test_list_block_no_false_reference(self) -> None:
        """A list_block with no numbered reference produces no edge."""
        table = _make_table(summary="Table 1: Results")
        lst = _make_list_block(
            content="• Refer to the table for details.",
            reading_order=0,
        )
        reg = _make_registry([table, lst])
        rels = generate_reference_relationships(reg)
        assert len(rels) == 0


# ===================================================================
#  8.  Descriptive relationships  (describes)
# ===================================================================


class TestDescriptiveRelationships:
    """``generate_descriptive_relationships`` creates ``describes`` edges."""

    def test_text_describes_nearby_table(self) -> None:
        """Text near a visual element in the same section describes it."""
        table = _make_table(
            reading_order=1,
            page_num=1,
            section_path="1.0",
        )
        text = _make_text(
            elem_id=_uid(110),
            content="This table shows the results.",
            reading_order=0,
            page_num=1,
            section_path="1.0",
            bbox=_bbox(left=0.0, top=0.0, right=0.5, bottom=0.05),
        )
        reg = _make_registry([text, table])
        rels = generate_descriptive_relationships(reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "describes"
        assert rel.source_id == text.element_id
        assert rel.target_id == table.element_id

    def test_different_sections_no_edge(self) -> None:
        """Text and visual in different sections produce no edge."""
        table = _make_table(
            reading_order=1,
            page_num=1,
            section_path="1.0",
        )
        text = _make_text(
            elem_id=_uid(120),
            reading_order=0,
            page_num=1,
            section_path="2.0",
        )
        reg = _make_registry([text, table])
        rels = generate_descriptive_relationships(reg)
        assert len(rels) == 0

    def test_far_reading_order_no_edge(self) -> None:
        """Text far from visual in reading order produces no edge."""
        # Two elements with 10 intervening fillers on different sections
        # so the gap exceeds the ±2 window.
        text = _make_text(
            elem_id=_uid(130),
            reading_order=0,
            page_num=1,
            section_path="1.0",
        )
        table = _make_table(
            reading_order=20,
            page_num=1,
            section_path="1.0",
        )
        # Fillers with a different section path won't match the table
        # but they ensure the table is at a far index.
        fillers = [
            _make_text(
                elem_id=_uid(131 + i),
                reading_order=1 + i,
                page_num=1,
                section_path="other",
            )
            for i in range(10)
        ]
        reg = _make_registry([text, *fillers, table])
        rels = generate_descriptive_relationships(reg)
        # text is at index 0, table at index 11 — gap exceeds window
        assert len(rels) == 0

    def test_list_block_describes_nearby_table(self) -> None:
        """A list_block near a visual in the same section describes it."""
        table = _make_table(
            reading_order=1,
            page_num=1,
            section_path="1.0",
        )
        lst = _make_list_block(
            elem_id=_uid(170),
            content="• This table shows the results:",
            reading_order=0,
            page_num=1,
            section_path="1.0",
        )
        reg = _make_registry([lst, table])
        rels = generate_descriptive_relationships(reg)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "describes"
        assert rel.source_id == lst.element_id
        assert rel.target_id == table.element_id

    def test_list_block_different_section_no_describes(self) -> None:
        """A list_block in a different section does not describe a visual."""
        table = _make_table(
            reading_order=1,
            page_num=1,
            section_path="1.0",
        )
        lst = _make_list_block(
            elem_id=_uid(171),
            reading_order=0,
            page_num=1,
            section_path="2.0",
        )
        reg = _make_registry([lst, table])
        rels = generate_descriptive_relationships(reg)
        assert len(rels) == 0


# ===================================================================
#  9.  Deduplication
# ===================================================================


class TestDeduplication:
    """``deduplicate_relationships`` removes duplicates and self-refs."""

    def _rel(
        self,
        source: uuid.UUID,
        target: uuid.UUID,
        rtype: str,
    ) -> RelationshipSchema:
        return RelationshipSchema(
            relationship_id=make_relationship_id(source, target, rtype),
            source_id=source,
            target_id=target,
            relationship_type=rtype,  # type: ignore[arg-type]
            weight=1.0,
        )

    def test_exact_duplicate_removed(self) -> None:
        """Exact duplicate (same source, target, type) is removed."""
        a, b = uuid.uuid4(), uuid.uuid4()
        rel = self._rel(a, b, "follows")
        result = deduplicate_relationships([rel, rel])
        assert len(result) == 1

    def test_self_reference_removed(self) -> None:
        """Self-reference (source == target) is removed."""
        a = uuid.uuid4()
        rel = self._rel(a, a, "follows")
        result = deduplicate_relationships([rel])
        assert len(result) == 0

    def test_same_section_as_self_ref_preserved(self) -> None:
        """``same_section_as`` self-references are preserved (group-based)."""
        a = uuid.uuid4()
        rel = RelationshipSchema(
            relationship_id=make_relationship_id(a, a, "same_section_as"),
            source_id=a,
            target_id=a,
            relationship_type="same_section_as",
            metadata={
                "group_id": "1.0",
                "member_ids": ["id1", "id2"],
                "element_count": 2,
            },
            weight=1.0,
        )
        result = deduplicate_relationships([rel])
        assert len(result) == 1

    def test_nearby_symmetric_duplicate_blocked(self) -> None:
        """Reverse of ``nearby`` is blocked (symmetric type)."""
        a, b = uuid.uuid4(), uuid.uuid4()
        rel1 = self._rel(a, b, "nearby")
        rel2 = self._rel(b, a, "nearby")
        result = deduplicate_relationships([rel1, rel2])
        assert len(result) == 1

    def test_multiple_duplicates(self) -> None:
        """Mix of duplicates, self-refs, and valid relationships."""
        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        rels = [
            self._rel(a, b, "follows"),
            self._rel(a, b, "follows"),  # duplicate
            self._rel(a, a, "follows"),  # self-ref
            self._rel(b, c, "precedes"),
            self._rel(b, c, "nearby"),
            self._rel(c, b, "nearby"),  # symmetric duplicate
        ]
        result = deduplicate_relationships(rels)
        assert len(result) == 3  # follows, precedes, nearby


# ===================================================================
#  10.  Composition via generate_all_relationships
# ===================================================================


class TestGenerateAllRelationships:
    """``generate_all_relationships`` composes and deduplicates."""

    def test_basic_composition(self) -> None:
        """Multiple generators contribute relationships."""
        t1 = _make_text(
            elem_id=_uid(140), reading_order=0, page_num=1, section_path="1.0"
        )
        t2 = _make_text(
            elem_id=_uid(141),
            reading_order=1,
            page_num=1,
            section_path="1.0",
            bbox=_bbox(left=0.01, top=0.0, right=0.11, bottom=0.1),
        )
        doc = _make_doc()
        reg = _make_registry([t1, t2])

        rels = generate_all_relationships(doc, reg)
        # Should have: follows, precedes, (maybe nearby if close), same_section_as
        types = {r.relationship_type for r in rels}
        assert "follows" in types
        assert "precedes" in types
        # t1 and t2 have overlapping bboxes, so nearby is expected
        # same_section_as requires ≥2 elements
        assert "same_section_as" in types

    def test_deterministic_output(self) -> None:
        """Same input produces identical relationship lists."""
        t1 = _make_text(
            elem_id=_uid(150), reading_order=0, page_num=1, section_path="1.0"
        )
        t2 = _make_text(
            elem_id=_uid(151),
            reading_order=1,
            page_num=1,
            section_path="1.0",
        )
        doc = _make_doc()
        reg = _make_registry([t1, t2])

        rels1 = generate_all_relationships(doc, reg)
        rels2 = generate_all_relationships(doc, reg)

        # Same number of relationships
        assert len(rels1) == len(rels2)
        # Same IDs (ordered comparison)
        ids1 = [r.relationship_id for r in rels1]
        ids2 = [r.relationship_id for r in rels2]
        assert ids1 == ids2

    def test_empty_registry(self) -> None:
        """An empty registry produces no relationships."""
        doc = _make_doc()
        reg = _make_registry([])
        rels = generate_all_relationships(doc, reg)
        assert len(rels) == 0

    def test_empty_document(self) -> None:
        """An empty document (no elements) produces no relationships."""
        doc = _make_doc()
        reg = _make_registry([])
        rels = generate_all_relationships(doc, reg)
        assert len(rels) == 0

    def test_no_duplicates_in_output(self) -> None:
        """The composed output has no duplicate relationships."""
        t1 = _make_text(
            elem_id=_uid(160), reading_order=0, page_num=1, section_path="1.0"
        )
        t2 = _make_text(
            elem_id=_uid(161),
            reading_order=1,
            page_num=1,
            section_path="1.0",
        )
        doc = _make_doc()
        reg = _make_registry([t1, t2])

        rels = generate_all_relationships(doc, reg)
        seen: Set[Tuple[uuid.UUID, uuid.UUID, str]] = set()
        for r in rels:
            key = (r.source_id, r.target_id, r.relationship_type)
            assert key not in seen, f"Duplicate: {key}"
            seen.add(key)


# ===================================================================
#  11.  Summary
# ===================================================================


class TestRelationshipSummary:
    """``generate_relationship_summary`` counts by type."""

    def test_counts_by_type(self) -> None:
        """Summary returns correct counts per type, sorted desc."""
        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        def _rel(s, t, rtype):
            return RelationshipSchema(
                relationship_id=make_relationship_id(s, t, rtype),
                source_id=s,
                target_id=t,
                relationship_type=rtype,  # type: ignore[arg-type]
                weight=1.0,
            )

        rels = [
            _rel(a, b, "follows"),
            _rel(b, c, "follows"),
            _rel(a, b, "precedes"),
            _rel(a, b, "contains"),
            _rel(b, a, "contains"),
        ]

        summary = generate_relationship_summary(rels)
        assert summary["follows"] == 2
        assert summary["precedes"] == 1
        assert summary["contains"] == 2
        assert len(summary) == 3

    def test_empty_list(self) -> None:
        """Empty list produces empty dict."""
        assert generate_relationship_summary([]) == {}

    def test_deterministic_order(self) -> None:
        """Summary keys are in deterministic order (desc count, then name)."""
        a, b = uuid.uuid4(), uuid.uuid4()

        def _rel(s, t, rtype):
            return RelationshipSchema(
                relationship_id=make_relationship_id(s, t, rtype),
                source_id=s,
                target_id=t,
                relationship_type=rtype,  # type: ignore[arg-type]
                weight=1.0,
            )

        # Use valid relationship types sorted by name.
        rels = [
            _rel(a, b, "nearby"),
            _rel(a, b, "follows"),
            _rel(a, b, "describes"),
        ]

        summary = generate_relationship_summary(rels)
        # All have count 1, so sorted by name alphabetically
        keys = list(summary.keys())
        assert keys == ["describes", "follows", "nearby"]

    def test_summary_all_types_present(self) -> None:
        """All relationship types should be countable."""
        a, b = uuid.uuid4(), uuid.uuid4()

        def _rel(s, t, rtype):
            return RelationshipSchema(
                relationship_id=make_relationship_id(s, t, rtype),
                source_id=s,
                target_id=t,
                relationship_type=rtype,  # type: ignore[arg-type]
                weight=1.0,
            )

        # Create one of every valid type
        types = [
            "contains",
            "belongs_to",
            "follows",
            "precedes",
            "refers_to",
            "describes",
            "has_caption",
            "nearby",
            "same_section_as",
        ]
        rels = [_rel(a, b, t) for t in types]
        summary = generate_relationship_summary(rels)
        assert len(summary) == len(types)
        for t in types:
            assert summary[t] == 1, f"Missing type: {t}"


# ===================================================================
#  12.  Integration — end-to-end scenario
# ===================================================================


class TestEndToEnd:
    """End-to-end relationship generation with realistic elements."""

    def test_complex_document(self) -> None:
        """A document with multiple element types produces all rel types."""
        doc = _make_doc()

        # Build elements on page 1 with section "1.0" and some on "2.0"
        elements = [
            _make_text(
                elem_id=_uid(200),
                reading_order=0,
                page_num=1,
                section_path="1.0",
                content="Introduction text.",
                bbox=_bbox(left=0.0, top=0.0, right=0.5, bottom=0.05),
            ),
            _make_table(
                reading_order=1,
                page_num=1,
                section_path="1.0",
                content="Table data",
                summary="Table 1: Results summary",
                bbox=_bbox(left=0.0, top=0.1, right=0.5, bottom=0.3),
            ),
            _make_caption(
                elem_id=_uid(202),
                reading_order=2,
                page_num=1,
                section_path="1.0",
                content="Table 1: Results",
                parent_element_id=_VISUAL_ELEM_IDS["table"],
                bbox=_bbox(left=0.0, top=0.32, right=0.5, bottom=0.35),
            ),
            _make_text(
                elem_id=_uid(203),
                reading_order=3,
                page_num=1,
                section_path="1.0",
                content="As shown in Table 1, the results are clear.",
                bbox=_bbox(left=0.0, top=0.36, right=0.5, bottom=0.4),
            ),
            _make_formula(
                reading_order=4,
                page_num=1,
                section_path="1.0",
                content="Equation 1: y = mx + b",
                bbox=_bbox(left=0.0, top=0.42, right=0.3, bottom=0.45),
            ),
            _make_text(
                elem_id=_uid(205),
                reading_order=5,
                page_num=1,
                section_path="2.0",
                content="Different section content.",
                bbox=_bbox(left=0.0, top=0.5, right=0.5, bottom=0.55),
            ),
        ]

        # Add a contains relationship (parent -> table)
        sec_id = uuid.uuid4()
        contains_rel = _make_contains_rel(sec_id, _VISUAL_ELEM_IDS["table"])
        doc = doc.model_copy(update={"relationships": [contains_rel]})

        reg = _make_registry(elements)
        rels = generate_all_relationships(doc, reg)

        # Verify we have all expected types
        types = {r.relationship_type for r in rels}
        assert "contains" in types  # from doc.relationships
        assert "belongs_to" in types  # inverse of contains
        assert "follows" in types  # sequential
        assert "precedes" in types  # sequential
        assert "has_caption" in types  # caption parent -> caption
        assert "refers_to" in types  # "Table 1" -> table
        assert "same_section_as" in types  # section grouping

        # No duplicates
        seen_ids = set()
        for r in rels:
            assert r.relationship_id not in seen_ids
            seen_ids.add(r.relationship_id)

        # No self-references (except same_section_as which is group-based)
        for r in rels:
            if r.relationship_type == "same_section_as":
                assert r.source_id == r.target_id
            else:
                assert r.source_id != r.target_id

    def test_deterministic_across_runs(self) -> None:
        """Running twice on the same input yields identical output."""
        doc = _make_doc()
        elements = [
            _make_text(
                elem_id=_uid(210),
                reading_order=0,
                page_num=1,
                section_path="1.0",
            ),
            _make_text(
                elem_id=_uid(211),
                reading_order=1,
                page_num=1,
                section_path="1.0",
            ),
        ]
        reg = _make_registry(elements)

        rels1 = generate_all_relationships(doc, reg)
        rels2 = generate_all_relationships(doc, reg)

        ids1 = [
            (r.relationship_id, r.source_id, r.target_id, r.relationship_type)
            for r in rels1
        ]
        ids2 = [
            (r.relationship_id, r.source_id, r.target_id, r.relationship_type)
            for r in rels2
        ]
        assert ids1 == ids2
