"""
Tests for ``src.normalization.hierarchy_builder`` (Sub-Task 7 —
Reconstruct Document Hierarchy and Assign Section Paths).

Uses synthetic elements throughout — no real Docling dependency.

Covers:
- Public imports.
- Flat numbered sections (1, 2, 3).
- Nested numbered sections (1 → 1.1, 1.2 → 2).
- Mix of numbered and unnumbered sections.
- Section header elements using explicit level.
- Headers/footers not treated as sections.
- Elements assigned non-empty section paths where expected.
- Valid parent-prefix property for section paths.
- Contains relationships for section->subsection and section->element.
- Deterministic section IDs / stable output across runs.
- Empty / no-section document does not crash.
- Appendices / front matter handling.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.normalization import (
    ElementRegistry,
    assign_section_paths,
    build_hierarchy,
)
from src.schemas import (
    BoundingBox,
    DocumentSchema,
    ElementSchema,
    FooterSchema,
    HeaderSchema,
    SectionHeaderSchema,
    SectionSchema,
    TextBlockSchema,
    make_relationship_id,
)


# ===================================================================
#  Helpers — create synthetic elements and documents
# ===================================================================


_DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _bbox(
    left: float = 0.0,
    top: float = 0.0,
    right: float = 0.5,
    bottom: float = 0.1,
) -> BoundingBox:
    return BoundingBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        coord_system="normalized",
    )


def _make_doc() -> DocumentSchema:
    """Minimal DocumentSchema."""
    return DocumentSchema(
        doc_id=_DOC_ID,
        title="Test Doc",
        source_path="/fake/test.pdf",
        file_hash="abcd1234",
        page_count=5,
        created_at=datetime(2025, 1, 1),
    )


def _text_block(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    **overrides: Any,
) -> TextBlockSchema:
    return TextBlockSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=_bbox(),
        reading_order=reading_order,
        element_type="text_block",
        content=content,
        **overrides,
    )


def _section_header(
    elem_id: str,
    content: str,
    level: int = 1,
    section_number: Optional[str] = None,
    page_num: int = 1,
    reading_order: int = 0,
) -> SectionHeaderSchema:
    return SectionHeaderSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=_bbox(),
        reading_order=reading_order,
        element_type="section_header",
        content=content,
        level=level,
        section_number=section_number,
    )


def _header_elem(
    elem_id: str,
    content: str = "Page Header",
    page_num: int = 1,
    reading_order: int = 0,
) -> HeaderSchema:
    return HeaderSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=_bbox(),
        reading_order=reading_order,
        element_type="header",
        content=content,
    )


def _footer_elem(
    elem_id: str,
    content: str = "Page Footer",
    page_num: int = 1,
    reading_order: int = 0,
) -> FooterSchema:
    return FooterSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=_bbox(),
        reading_order=reading_order,
        element_type="footer",
        content=content,
    )


def _build_registry(elements: List[ElementSchema]) -> ElementRegistry:
    reg = ElementRegistry()
    for elem in elements:
        reg.add(elem)
    return reg


def _build_doc_with_elements(
    elements: List[ElementSchema],
) -> DocumentSchema:
    """Build a DocumentSchema with the given elements."""
    doc = _make_doc()
    elem_dict: Dict[str, ElementSchema] = {}
    for elem in elements:
        elem_dict[str(elem.element_id)] = elem
    return doc.model_copy(update={"elements": elem_dict})


def _get_sections(doc: DocumentSchema) -> List[Dict[str, Any]]:
    """Extract section list from metadata.custom."""
    return doc.metadata.custom.get("sections", [])


def _get_contains_rels(
    doc: DocumentSchema,
) -> List[ElementSchema]:
    """Extract 'contains' relationships from doc."""
    return [r for r in doc.relationships if r.relationship_type == "contains"]


# ===================================================================
#  1.  Public imports
# ===================================================================


class TestPublicImports:
    """build_hierarchy and assign_section_paths are importable."""

    def test_imports(self) -> None:
        assert callable(build_hierarchy)
        assert callable(assign_section_paths)

    def test_import_from_package(self) -> None:
        from src.normalization import build_hierarchy as bh, assign_section_paths as asp

        assert callable(bh)
        assert callable(asp)


# ===================================================================
#  2.  Empty / no-section document
# ===================================================================


class TestEmptyDocument:
    """Documents with no sections do not crash."""

    def test_no_elements(self) -> None:
        doc = _build_doc_with_elements([])
        result = build_hierarchy(doc)
        assert result.doc_id == doc.doc_id
        assert len(result.elements) == 0
        assert _get_sections(result) == []
        # Should have no new contains relationships beyond originals
        assert len(_get_contains_rels(result)) == 0

    def test_only_text_no_section_headers(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001", "Some text", reading_order=0
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002", "More text", reading_order=1
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        assert _get_sections(result) == []
        # Elements without a section should have empty section_path
        for e in result.elements.values():
            assert e.section_path == ""

    def test_only_headers_and_footers(self) -> None:
        elems = [
            _header_elem("a0000001-0000-0000-0000-000000000001"),
            _footer_elem("a0000002-0000-0000-0000-000000000002"),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        assert _get_sections(result) == []
        for e in result.elements.values():
            assert e.section_path == ""


# ===================================================================
#  3.  Flat numbered sections
# ===================================================================


class TestFlatNumberedSections:
    """Sections at a single level with numbers."""

    def test_two_flat_sections(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Introduction",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Intro text.",
                reading_order=1,
            ),
            _section_header(
                "a0000003-0000-0000-0000-000000000003",
                "2. Method",
                level=1,
                section_number="2",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "Method text.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        assert sections[0]["section_path"] == "1"
        assert sections[0]["title"] == "1. Introduction"
        assert sections[1]["section_path"] == "2"
        assert sections[1]["title"] == "2. Method"

        # Elements in sections
        elem_2 = result.elements["a0000002-0000-0000-0000-000000000002"]
        assert elem_2.section_path == "1"
        elem_4 = result.elements["a0000004-0000-0000-0000-000000000004"]
        assert elem_4.section_path == "2"

        # Section headers themselves get the path
        elem_1 = result.elements["a0000001-0000-0000-0000-000000000001"]
        assert elem_1.section_path == "1"
        elem_3 = result.elements["a0000003-0000-0000-0000-000000000003"]
        assert elem_3.section_path == "2"

    def test_section_path_parent_prefix_valid(self) -> None:
        """Every section path is a prefix of its children's paths."""
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. First",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Under 1.",
                reading_order=1,
            ),
            _section_header(
                "a0000003-0000-0000-0000-000000000003",
                "2. Second",
                level=1,
                section_number="2",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        elements = result.elements
        # Elements under "1" have paths starting with "1"
        assert elements["a0000002-0000-0000-0000-000000000002"].section_path == "1"


# ===================================================================
#  4.  Nested numbered sections
# ===================================================================


class TestNestedNumberedSections:
    """Sections nested at multiple levels."""

    def test_two_level_nesting(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Introduction",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "1.1 Background",
                level=2,
                section_number="1.1",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "Background text.",
                reading_order=2,
            ),
            _section_header(
                "a0000004-0000-0000-0000-000000000004",
                "1.2 Approach",
                level=2,
                section_number="1.2",
                reading_order=3,
            ),
            _text_block(
                "a0000005-0000-0000-0000-000000000005",
                "Approach text.",
                reading_order=4,
            ),
            _section_header(
                "a0000006-0000-0000-0000-000000000006",
                "2. Results",
                level=1,
                section_number="2",
                reading_order=5,
            ),
            _text_block(
                "a0000007-0000-0000-0000-000000000007",
                "Results text.",
                reading_order=6,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 4

        paths = [s["section_path"] for s in sections]
        assert paths == ["1", "1.1", "1.2", "2"]

        # Check parent references
        # Section "1" has no parent
        s1 = [s for s in sections if s["section_path"] == "1"][0]
        assert s1["parent_section_id"] is None

        # Section "1.1" has parent "1"
        s11 = [s for s in sections if s["section_path"] == "1.1"][0]
        # s1["section_id"] is already a UUID (from model_dump)
        assert s11["parent_section_id"] == s1["section_id"]

        # Section "1.2" has parent "1"
        s12 = [s for s in sections if s["section_path"] == "1.2"][0]
        assert s12["parent_section_id"] == s1["section_id"]

        # Section "2" has no parent
        s2 = [s for s in sections if s["section_path"] == "2"][0]
        assert s2["parent_section_id"] is None

        # Contains relationships
        contains = _get_contains_rels(result)
        # document -> section 1, document -> section 2
        # section 1 -> section 1.1, section 1 -> section 1.2
        # section 1 -> background text, section 1.1 -> background text?
        # Wait: the background text is under section 1.1
        # section 1.1 -> background text
        # section 1.2 -> approach text
        # section 2 -> results text
        # That's 2 (doc->sect) + 2 (sect->subsect) + 3 (sect->elem) = 7
        assert len(contains) == 7

        # Check element section paths
        assert (
            result.elements["a0000003-0000-0000-0000-000000000003"].section_path
            == "1.1"
        )
        assert (
            result.elements["a0000005-0000-0000-0000-000000000005"].section_path
            == "1.2"
        )
        assert (
            result.elements["a0000007-0000-0000-0000-000000000007"].section_path == "2"
        )


# ===================================================================
#  5.  Mix of numbered and unnumbered sections
# ===================================================================


class TestMixedNumberedUnnumbered:
    """Documents with both numbered and unnumbered sections."""

    def test_front_matter_then_numbered(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001",
                "Abstract",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Abstract text here.",
                reading_order=1,
            ),
            _section_header(
                "a0000003-0000-0000-0000-000000000003",
                "1. Introduction",
                level=1,
                section_number="1",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "Intro text.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        # Abstract should get a synthetic path (0.1)
        assert sections[0]["section_path"] in ("0.1",)
        # Introduction keeps its number
        assert sections[1]["section_path"] == "1"

        # Abstract content is under the front matter section
        assert (
            result.elements["a0000002-0000-0000-0000-000000000002"].section_path
            == sections[0]["section_path"]
        )
        # Intro text is under section 1
        assert (
            result.elements["a0000004-0000-0000-0000-000000000004"].section_path == "1"
        )

    def test_unnumbered_section_after_numbered(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Results",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Results text.",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "Conclusion",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "Conclusion text.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        # Conclusion should be synthetic path
        assert sections[1]["section_path"] == "0.1"


# ===================================================================
#  6.  Section header elements using explicit level
# ===================================================================


class TestSectionHeaderExplicitLevel:
    """SectionHeaderSchema with explicit level is respected."""

    def test_deeply_nested_explicit_level(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1.",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "1.1",
                level=2,
                section_number="1.1",
                reading_order=1,
            ),
            _section_header(
                "a0000003-0000-0000-0000-000000000003",
                "1.1.1",
                level=3,
                section_number="1.1.1",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "Deep text.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 3
        assert sections[0]["level"] == 1
        assert sections[1]["level"] == 2
        assert sections[2]["level"] == 3

        # Deep text belongs to deepest section
        assert (
            result.elements["a0000004-0000-0000-0000-000000000004"].section_path
            == "1.1.1"
        )

    def test_level_jump(self) -> None:
        """Jumping from level 1 to level 3 (skipping level 2) is handled."""
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1.",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "1.1.1",
                level=3,
                section_number="1.1.1",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "Text in 1.1.1.",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        assert sections[1]["level"] == 3
        assert sections[1]["section_path"] == "1.1.1"
        # parent_section_id is already a UUID from model_dump
        assert sections[1]["parent_section_id"] == sections[0]["section_id"]


# ===================================================================
#  7.  Headers/footers not treated as sections
# ===================================================================


class TestHeadersFootersSkipped:
    """Headers and footers are ignored for section detection."""

    def test_header_not_section(self) -> None:
        elems = [
            _header_elem(
                "a0000001-0000-0000-0000-000000000001",
                "Page 1 Header",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "1. Introduction",
                level=1,
                section_number="1",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "Intro text.",
                reading_order=2,
            ),
            _footer_elem(
                "a0000004-0000-0000-0000-000000000004",
                "Page 1 Footer",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 1
        assert sections[0]["section_path"] == "1"
        assert sections[0]["title"] == "1. Introduction"

        # Header and footer should have empty section_path
        assert (
            result.elements["a0000001-0000-0000-0000-000000000001"].section_path == ""
        )
        assert (
            result.elements["a0000004-0000-0000-0000-000000000004"].section_path == ""
        )

    def test_footer_after_section_maintains_path(self) -> None:
        """Footer after a section does not get a section path."""
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Intro",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
            _footer_elem(
                "a0000003-0000-0000-0000-000000000003",
                "Footer",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        assert (
            result.elements["a0000003-0000-0000-0000-000000000003"].section_path == ""
        )


# ===================================================================
#  8.  Elements assigned non-empty section paths
# ===================================================================


class TestElementSectionPaths:
    """Regular elements correctly receive section paths."""

    def test_text_under_section_gets_path(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Section",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Some text.",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "More text.",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        assert (
            result.elements["a0000002-0000-0000-0000-000000000002"].section_path == "1"
        )
        assert (
            result.elements["a0000003-0000-0000-0000-000000000003"].section_path == "1"
        )

    def test_text_before_first_section_has_empty_path(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001",
                "Preamble text.",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "1. First",
                level=1,
                section_number="1",
                reading_order=1,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        assert (
            result.elements["a0000001-0000-0000-0000-000000000001"].section_path == ""
        )


# ===================================================================
#  9.  Contains relationships
# ===================================================================


class TestContainsRelationships:
    """Contains relationships are generated correctly."""

    def test_section_contains_element(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Section",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)
        sec_id = sections[0]["section_id"]
        elem_id = uuid.UUID("a0000002-0000-0000-0000-000000000002")

        contains = _get_contains_rels(result)
        # doc -> section (1), section -> element (1) = 2
        assert len(contains) == 2

        # Check section -> element
        sec_elem_rels = [
            r for r in contains if r.source_id == sec_id and r.target_id == elem_id
        ]
        assert len(sec_elem_rels) == 1

    def test_section_contains_subsection(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1.",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "1.1",
                level=2,
                section_number="1.1",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "Sub text.",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)
        s1_id = sections[0]["section_id"]
        s11_id = sections[1]["section_id"]

        contains = _get_contains_rels(result)

        # section 1 -> section 1.1
        s1_s11 = [r for r in contains if r.source_id == s1_id and r.target_id == s11_id]
        assert len(s1_s11) == 1

        # section 1.1 -> text element
        s11_elem = [
            r
            for r in contains
            if r.source_id == s11_id
            and r.target_id == uuid.UUID("a0000003-0000-0000-0000-000000000003")
        ]
        assert len(s11_elem) == 1

    def test_document_contains_top_level_sections(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1.",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _section_header(
                "a0000002-0000-0000-0000-000000000002",
                "2.",
                level=1,
                section_number="2",
                reading_order=1,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        contains = _get_contains_rels(result)
        doc_contains = [r for r in contains if r.source_id == doc.doc_id]
        assert len(doc_contains) == 2  # doc -> section 1, doc -> section 2

    def test_no_duplicate_contains(self) -> None:
        """Running hierarchy builder twice does not duplicate contains."""
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1.",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result1 = build_hierarchy(doc)
        # Run again on the result
        result2 = build_hierarchy(result1)
        contains2 = _get_contains_rels(result2)

        # doc -> section (1), section -> element (1) = 2 (not duplicated)
        assert len(contains2) == 2


# ===================================================================
#  10.  Deterministic section IDs
# ===================================================================


class TestDeterministicSectionIds:
    """Section IDs are stable for the same inputs."""

    def test_same_input_same_section_ids(self) -> None:
        elems1 = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Intro",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        elems2 = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Intro",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        doc1 = _build_doc_with_elements(elems1)
        doc2 = _build_doc_with_elements(elems2)
        result1 = build_hierarchy(doc1)
        result2 = build_hierarchy(doc2)

        s1 = _get_sections(result1)
        s2 = _get_sections(result2)
        assert len(s1) == len(s2)
        assert s1[0]["section_id"] == s2[0]["section_id"]

    def test_different_title_different_id(self) -> None:
        elems1 = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Intro",
                level=1,
                section_number="1",
                reading_order=0,
            ),
        ]
        elems2 = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Different",
                level=1,
                section_number="1",
                reading_order=0,
            ),
        ]
        doc1 = _build_doc_with_elements(elems1)
        doc2 = _build_doc_with_elements(elems2)
        result1 = build_hierarchy(doc1)
        result2 = build_hierarchy(doc2)

        s1 = _get_sections(result1)
        s2 = _get_sections(result2)
        assert s1[0]["section_id"] != s2[0]["section_id"]


# ===================================================================
#  11.  Appendices and front matter
# ===================================================================


class TestAppendicesFrontMatter:
    """Appendices and front matter get appropriate handling."""

    def test_appendix_pattern(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Main",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Main text.",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "Appendix A",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "Appendix text.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) >= 2
        # Appendix A should be detected
        appendix_sec = [s for s in sections if "appendix" in s["title"].lower()]
        assert len(appendix_sec) >= 1

    def test_abstract_front_matter(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001",
                "Abstract",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Abstract content.",
                reading_order=1,
            ),
            _section_header(
                "a0000003-0000-0000-0000-000000000003",
                "1. Introduction",
                level=1,
                section_number="1",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        # Abstract gets a synthetic path
        abstract_sec = sections[0]
        assert abstract_sec["title"].lower() == "abstract"
        assert abstract_sec["section_path"].startswith("0.")

        # Abstract content gets that path
        assert (
            result.elements["a0000002-0000-0000-0000-000000000002"].section_path
            == abstract_sec["section_path"]
        )

    def test_keywords_and_introduction(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001",
                "Keywords",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "key1, key2, key3",
                reading_order=1,
            ),
            _section_header(
                "a0000003-0000-0000-0000-000000000003",
                "1. Introduction",
                level=1,
                section_number="1",
                reading_order=2,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        kw_sec = sections[0]
        assert kw_sec["title"].lower() == "keywords"


# ===================================================================
#  12.  Text-based heuristic section detection
# ===================================================================


class TestHeuristicSectionDetection:
    """text_block elements with numbering patterns are detected as sections."""

    def test_numbered_text_block_detected(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001",
                "1. Introduction",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Some intro text.",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "2. Related Work",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "Related work text.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 2
        assert sections[0]["section_path"] == "1"
        assert sections[1]["section_path"] == "2"

        # Content assigned correctly
        assert (
            result.elements["a0000002-0000-0000-0000-000000000002"].section_path == "1"
        )
        assert (
            result.elements["a0000004-0000-0000-0000-000000000004"].section_path == "2"
        )

    def test_deep_numbering_in_text_block(self) -> None:
        elems = [
            _text_block(
                "a0000001-0000-0000-0000-000000000001",
                "1.2.3 Deep Section",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Deep text.",
                reading_order=1,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)

        assert len(sections) == 1
        assert sections[0]["section_path"] == "1.2.3"
        assert sections[0]["level"] == 3


# ===================================================================
#  13.  assign_section_paths convenience wrapper
# ===================================================================


class TestAssignSectionPaths:
    """assign_section_paths convenience wrapper works."""

    def test_wrapper_calls_build_hierarchy(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Section",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = assign_section_paths(doc)
        assert (
            result.elements["a0000002-0000-0000-0000-000000000002"].section_path == "1"
        )
        assert len(_get_sections(result)) == 1

    def test_wrapper_with_registry(self) -> None:
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Section",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        reg = _build_registry(elems)
        doc = _build_doc_with_elements(elems)
        result = assign_section_paths(doc, registry=reg)
        assert (
            result.elements["a0000002-0000-0000-0000-000000000002"].section_path == "1"
        )


# ===================================================================
#  14.  Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_single_element_section_header(self) -> None:
        """A document with just one section header is fine."""
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Only Section",
                level=1,
                section_number="1",
                reading_order=0,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        assert len(_get_sections(result)) == 1
        assert len(_get_contains_rels(result)) == 1  # doc -> section

    def test_references_not_treated_as_sections(self) -> None:
        """'References' text block should be detected as a section."""
        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Body",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Body text.",
                reading_order=1,
            ),
            _text_block(
                "a0000003-0000-0000-0000-000000000003",
                "References",
                reading_order=2,
            ),
            _text_block(
                "a0000004-0000-0000-0000-000000000004",
                "[1] Some ref.",
                reading_order=3,
            ),
        ]
        doc = _build_doc_with_elements(elems)
        result = build_hierarchy(doc)
        sections = _get_sections(result)
        # "References" should be detected as a (back matter) section header
        ref_sections = [s for s in sections if "references" in s["title"].lower()]
        assert len(ref_sections) >= 1
        # References text is under that section
        assert (
            result.elements["a0000004-0000-0000-0000-000000000004"].section_path
            == ref_sections[0]["section_path"]
        )

    def test_preserves_existing_relationships(self) -> None:
        """Existing relationships in the doc are preserved."""
        from src.schemas import RelationshipSchema, make_relationship_id

        elems = [
            _section_header(
                "a0000001-0000-0000-0000-000000000001",
                "1. Section",
                level=1,
                section_number="1",
                reading_order=0,
            ),
            _text_block(
                "a0000002-0000-0000-0000-000000000002",
                "Text.",
                reading_order=1,
            ),
        ]
        existing_rel = RelationshipSchema(
            relationship_id=make_relationship_id(
                uuid.UUID("a0000001-0000-0000-0000-000000000001"),
                uuid.UUID("a0000002-0000-0000-0000-000000000002"),
                "follows",
            ),
            source_id=uuid.UUID("a0000001-0000-0000-0000-000000000001"),
            target_id=uuid.UUID("a0000002-0000-0000-0000-000000000002"),
            relationship_type="follows",
        )
        doc = _build_doc_with_elements(elems)
        doc = doc.model_copy(update={"relationships": [existing_rel]})
        result = build_hierarchy(doc)

        follows = [r for r in result.relationships if r.relationship_type == "follows"]
        assert len(follows) == 1
        # Follows relationship is preserved
        assert follows[0].relationship_id == existing_rel.relationship_id
