"""
Tests for ``src.normalization.table_processor`` (Sub-Task 8 — Process
Tables into Structured Representations).

Uses fakes throughout — no real Docling dependency.

Covers:
- Public imports.
- process_table from list-of-dicts / dataframe-like / list-of-lists /
  existing json_data.
- Markdown / HTML / JSON / summary population.
- row_count / col_count / header extraction.
- Empty table handling.
- detect_spanning_tables positive and negative cases.
- Deterministic span_group_id.
- generate_table_relationships for nearby caption / text / footnote
  elements.
- No duplicate relationships / no self-reference.
- Optional process_tables document-wide helper.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.normalization import (
    ElementRegistry,
    detect_spanning_tables,
    generate_table_relationships,
    process_table,
    process_tables,
)
from src.schemas import (
    BoundingBox,
    CaptionSchema,
    DocumentSchema,
    ElementSchema,
    FootnoteSchema,
    RelationshipSchema,
    TableSchema,
    TextBlockSchema,
)


# ===================================================================
#  Helpers — synthetic elements and documents
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


def _make_doc(page_count: int = 5) -> DocumentSchema:
    return DocumentSchema(
        doc_id=_DOC_ID,
        title="Test Doc",
        source_path="/fake/test.pdf",
        file_hash="abcd1234",
        page_count=page_count,
        created_at=datetime(2025, 1, 1),
    )


def _make_table(
    elem_id: Optional[str] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    markdown: str = "",
    html: str = "",
    json_data: Optional[Dict[str, Any]] = None,
    row_count: int = 0,
    col_count: int = 0,
    headers: Optional[List[str]] = None,
    summary: str = "",
    content: str = "",
    is_spanning: bool = False,
    span_group_id: Optional[str] = None,
    bbox: Optional[BoundingBox] = None,
) -> TableSchema:
    return TableSchema(
        element_id=uuid.UUID(elem_id or "22222222-2222-2222-2222-222222222222"),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="table",
        content=content,
        markdown=markdown,
        html=html,
        json_data=json_data or {},
        row_count=row_count,
        col_count=col_count,
        headers=headers or [],
        summary=summary,
        is_spanning=is_spanning,
        span_group_id=span_group_id,
    )


def _make_text_block(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    bbox: Optional[BoundingBox] = None,
) -> TextBlockSchema:
    return TextBlockSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        element_type="text_block",
        content=content,
    )


def _make_caption(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    bbox: Optional[BoundingBox] = None,
) -> CaptionSchema:
    return CaptionSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        element_type="caption",
        content=content,
    )


def _make_footnote(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    bbox: Optional[BoundingBox] = None,
) -> FootnoteSchema:
    return FootnoteSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        element_type="footnote",
        content=content,
    )


def _build_registry(elements: List[ElementSchema]) -> ElementRegistry:
    reg = ElementRegistry()
    for elem in elements:
        reg.add(elem)
    return reg


# ===================================================================
#  Fake DataFrame-like object (no pandas)
# ===================================================================


class FakeDataFrame:
    """A minimal DataFrame-like object for testing, without pandas."""

    def __init__(self, columns: List[str], rows: List[List[str]]) -> None:
        self._columns = columns
        self._rows = rows

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def values(self) -> List[List[str]]:
        return self._rows

    @property
    def iloc(self) -> Any:
        return self  # simplified

    def head(self, n: int = 5) -> FakeDataFrame:
        return FakeDataFrame(self._columns, self._rows[:n])

    def to_dict(self, orient: str = "records") -> List[Dict[str, str]]:
        return [
            {col: row[i] for i, col in enumerate(self._columns)} for row in self._rows
        ]

    def iterrows(self) -> Any:
        for i, row in enumerate(self._rows):
            yield i, row

    def __len__(self) -> int:
        return len(self._rows)


# ===================================================================
#  1.  Public imports
# ===================================================================


class TestPublicImports:
    """All public API symbols are importable."""

    def test_imports(self) -> None:
        assert callable(process_table)
        assert callable(detect_spanning_tables)
        assert callable(generate_table_relationships)
        assert callable(process_tables)

    def test_import_from_normalization(self) -> None:
        from src.normalization import (
            detect_spanning_tables as d,
            generate_table_relationships as g,
            process_table as p,
            process_tables as pt,
        )

        assert callable(p)
        assert callable(d)
        assert callable(g)
        assert callable(pt)


# ===================================================================
#  2.  process_table — data source extraction
# ===================================================================


class TestProcessTableFromListOfDicts:
    """process_table with list-of-dicts data (via dl_doc)."""

    def test_basic(self) -> None:
        elem = _make_table()
        dl_doc = {
            "data": [
                {"Name": "Alice", "Age": "30"},
                {"Name": "Bob", "Age": "25"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        assert result.row_count == 2
        assert result.col_count == 2
        assert result.headers == ["Name", "Age"]
        assert "Alice" in result.markdown
        assert "Bob" in result.markdown
        assert "<table>" in result.html
        assert result.json_data.get("rows") == [
            {"Name": "Alice", "Age": "30"},
            {"Name": "Bob", "Age": "25"},
        ]
        assert "2 row(s)" in result.summary
        assert "2 column(s)" in result.summary
        assert "Name" in result.summary


class TestProcessTableFromDataFrameLike:
    """process_table with a DataFrame-like object."""

    def test_basic(self) -> None:
        elem = _make_table()
        df = FakeDataFrame(
            columns=["A", "B"],
            rows=[["1", "2"], ["3", "4"]],
        )
        dl_doc = {"data": df}
        result = process_table(elem, dl_doc=dl_doc)
        assert result.row_count == 2
        assert result.col_count == 2
        assert result.headers == ["A", "B"]
        assert "1" in result.markdown
        assert "3" in result.markdown
        assert result.json_data.get("rows") == [
            {"A": "1", "B": "2"},
            {"A": "3", "B": "4"},
        ]


class TestProcessTableFromListOfLists:
    """process_table with list-of-lists data."""

    def test_basic(self) -> None:
        elem = _make_table()
        dl_doc = {
            "data": [
                ["Name", "Age"],
                ["Alice", "30"],
                ["Bob", "25"],
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        # First row looks like a header
        assert result.headers == ["Name", "Age"]
        assert result.row_count == 2
        assert result.col_count == 2

    def test_no_headers(self) -> None:
        elem = _make_table()
        dl_doc = {
            "data": [
                ["1", "2"],
                ["3", "4"],
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        # Numeric first row — not header-like
        assert result.headers == ["Column 1", "Column 2"]
        assert result.row_count == 2


class TestProcessTableFromExistingJsonData:
    """process_table with existing json_data."""

    def test_basic(self) -> None:
        elem = _make_table(
            json_data={
                "rows": [
                    {"X": "10", "Y": "20"},
                    {"X": "30", "Y": "40"},
                ]
            }
        )
        result = process_table(elem)
        assert result.row_count == 2
        assert result.col_count == 2
        assert result.headers == ["X", "Y"]
        assert "10" in result.markdown


class TestProcessTableFromHtml:
    """process_table from HTML table string."""

    def test_basic(self) -> None:
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>Alpha</td><td>100</td></tr>"
            "</table>"
        )
        elem = _make_table(html=html)
        result = process_table(elem)
        assert result.row_count == 1
        assert result.col_count == 2
        assert result.headers == ["Name", "Value"]
        assert "Alpha" in result.markdown
        assert "100" in result.html


class TestProcessTableFromMarkdown:
    """process_table from markdown table string."""

    def test_basic(self) -> None:
        md = "| H1 | H2 |\n|---|---|\n| A | B |\n| C | D |"
        elem = _make_table(markdown=md)
        result = process_table(elem)
        assert result.row_count == 2
        assert result.col_count == 2
        assert result.headers == ["H1", "H2"]
        assert "A" in result.markdown
        assert "C" in result.markdown


class TestProcessTableFromTextFallback:
    """process_table fallback to plain content text."""

    def test_tab_delimited(self) -> None:
        content = "Name\tAge\nAlice\t30\nBob\t25"
        elem = _make_table(content=content)
        result = process_table(elem)
        assert result.row_count == 2
        assert result.col_count == 2
        # First row "Name\tAge" is header-like
        assert result.headers == ["Name", "Age"]

    def test_space_delimited(self) -> None:
        content = "Name  Age\nAlice  30\nBob  25"
        elem = _make_table(content=content)
        result = process_table(elem)
        assert result.row_count == 2
        assert result.col_count == 2


# ===================================================================
#  3.  process_table — metadata extraction
# ===================================================================


class TestProcessTableMetadata:
    """row_count, col_count, headers, and summary population."""

    def test_row_count_and_col_count(self) -> None:
        elem = _make_table()
        dl_doc = {
            "data": [
                {"A": "1", "B": "2", "C": "3"},
                {"A": "4", "B": "5", "C": "6"},
                {"A": "7", "B": "8", "C": "9"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        assert result.row_count == 3
        assert result.col_count == 3

    def test_headers_preserved(self) -> None:
        elem = _make_table(headers=["Existing", "Headers"])
        dl_doc = {
            "data": [
                {"X": "1", "Y": "2"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        # Existing headers should be replaced with extracted ones
        assert result.headers == ["X", "Y"]

    def test_summary_includes_details(self) -> None:
        elem = _make_table()
        dl_doc = {
            "data": [
                {"Name": "Alice", "Score": "95"},
                {"Name": "Bob", "Score": "87"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        assert "2 row(s)" in result.summary
        assert "2 column(s)" in result.summary
        assert "Name" in result.summary
        assert "Score" in result.summary

    def test_summary_preserves_existing_prefix(self) -> None:
        elem = _make_table(summary="Table 1: Results")
        dl_doc = {
            "data": [
                {"A": "1"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        assert "Table 1: Results" in result.summary
        assert "1 row(s)" in result.summary


# ===================================================================
#  4.  process_table — empty/malformed table handling
# ===================================================================


class TestProcessTableEmpty:
    """Empty or malformed tables do not crash."""

    def test_empty_table(self) -> None:
        elem = _make_table()
        result = process_table(elem)
        # No data to extract — should return the element unchanged
        assert result is elem or (
            result.row_count == 0
            and result.col_count == 0
            and result.markdown == ""
            and result.html == ""
            and result.json_data == {}
        )

    def test_empty_list(self) -> None:
        elem = _make_table()
        dl_doc = {"data": []}
        result = process_table(elem, dl_doc=dl_doc)
        # Empty list — no rows
        assert result.row_count == 0

    def test_empty_string_content(self) -> None:
        elem = _make_table(content="")
        result = process_table(elem)
        assert result.row_count == 0

    def test_single_row_only(self) -> None:
        elem = _make_table()
        dl_doc = {
            "data": [
                {"Only": "row"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        assert result.row_count == 1
        assert result.col_count == 1
        assert result.headers == ["Only"]
        assert result.markdown
        assert result.html
        assert result.json_data.get("rows") == [{"Only": "row"}]

    def test_malformed_data_none(self) -> None:
        elem = _make_table()
        dl_doc = {"data": None}
        result = process_table(elem, dl_doc=dl_doc)
        assert result.row_count == 0
        # Should not raise


# ===================================================================
#  5.  detect_spanning_tables
# ===================================================================


class TestDetectSpanningTables:
    """Positive and negative cases for spanning table detection."""

    def test_positive_consecutive_pages(self) -> None:
        """Two tables on consecutive pages with same structure."""
        t1 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333331",
            page_num=1,
            section_path="3.1",
            col_count=3,
            headers=["A", "B", "C"],
        )
        t2 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333332",
            page_num=2,
            section_path="3.1",
            col_count=3,
            headers=["A", "B", "C"],
        )
        result = detect_spanning_tables([t1, t2])
        assert len(result) == 2
        assert result[0].is_spanning is True
        assert result[1].is_spanning is True
        # Both share the same span_group_id
        assert result[0].span_group_id is not None
        assert result[0].span_group_id == result[1].span_group_id

    def test_positive_three_consecutive(self) -> None:
        """Three tables across three pages."""
        tables = [
            _make_table(
                elem_id=f"33333333-3333-3333-3333-33333333333{i}",
                page_num=i,
                section_path="2",
                col_count=2,
                headers=["X", "Y"],
            )
            for i in range(1, 4)
        ]
        result = detect_spanning_tables(tables)
        assert all(r.is_spanning for r in result)
        ids = {r.span_group_id for r in result}
        assert len(ids) == 1  # all same group

    def test_negative_different_section(self) -> None:
        """Different section paths should not be marked."""
        t1 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333341",
            page_num=1,
            section_path="2.1",
            col_count=2,
            headers=["A", "B"],
        )
        t2 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333342",
            page_num=2,
            section_path="3.1",
            col_count=2,
            headers=["A", "B"],
        )
        result = detect_spanning_tables([t1, t2])
        assert not result[0].is_spanning
        assert not result[1].is_spanning

    def test_negative_different_col_count(self) -> None:
        """Different column counts should not be paired."""
        t1 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333351",
            page_num=1,
            section_path="2",
            col_count=3,
            headers=["A", "B", "C"],
        )
        t2 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333352",
            page_num=2,
            section_path="2",
            col_count=2,
            headers=["A", "B"],
        )
        result = detect_spanning_tables([t1, t2])
        assert not result[0].is_spanning
        assert not result[1].is_spanning

    def test_negative_not_consecutive(self) -> None:
        """Tables far apart should not be paired."""
        t1 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333361",
            page_num=1,
            section_path="2",
            col_count=2,
            headers=["A", "B"],
        )
        t2 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333362",
            page_num=10,
            section_path="2",
            col_count=2,
            headers=["A", "B"],
        )
        result = detect_spanning_tables([t1, t2])
        assert not result[0].is_spanning
        assert not result[1].is_spanning

    def test_single_table_not_spanned(self) -> None:
        """A lone table should never be marked spanning."""
        t1 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333371",
            page_num=1,
            section_path="2",
            col_count=2,
        )
        result = detect_spanning_tables([t1])
        assert not result[0].is_spanning
        assert result[0].span_group_id is None

    def test_empty_list(self) -> None:
        """Empty input returns empty list."""
        result = detect_spanning_tables([])
        assert result == []


class TestSpanGroupIdDeterminism:
    """span_group_id values are stable across runs."""

    def test_deterministic(self) -> None:
        t1 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333381",
            page_num=1,
            section_path="2",
            col_count=2,
            headers=["A", "B"],
        )
        t2 = _make_table(
            elem_id="33333333-3333-3333-3333-333333333382",
            page_num=2,
            section_path="2",
            col_count=2,
            headers=["A", "B"],
        )
        result1 = detect_spanning_tables([t1, t2])
        result2 = detect_spanning_tables([t1, t2])
        assert result1[0].span_group_id == result2[0].span_group_id
        assert result1[1].span_group_id == result2[1].span_group_id

    def test_different_section_different_id(self) -> None:
        """Tables in different sections get different span IDs."""
        t1a = _make_table(
            elem_id="33333333-3333-3333-3333-333333333391",
            page_num=1,
            section_path="2.1",
            col_count=2,
            headers=["A", "B"],
        )
        t1b = _make_table(
            elem_id="33333333-3333-3333-3333-333333333392",
            page_num=2,
            section_path="2.1",
            col_count=2,
            headers=["A", "B"],
        )
        t2a = _make_table(
            elem_id="33333333-3333-3333-3333-333333333393",
            page_num=3,
            section_path="2.2",
            col_count=2,
            headers=["A", "B"],
        )
        t2b = _make_table(
            elem_id="33333333-3333-3333-3333-333333333394",
            page_num=4,
            section_path="2.2",
            col_count=2,
            headers=["A", "B"],
        )
        result = detect_spanning_tables([t1a, t1b, t2a, t2b])
        group1 = result[0].span_group_id
        group2 = result[2].span_group_id
        assert group1 != group2


# ===================================================================
#  6.  generate_table_relationships
# ===================================================================


class TestGenerateTableRelationships:
    """Relationship generation for tables."""

    def test_has_caption_nearby(self) -> None:
        """Caption element spatially close to table."""
        tbl_bbox = _bbox(left=0.1, top=0.2, right=0.5, bottom=0.4)
        cap_bbox = _bbox(left=0.1, top=0.41, right=0.5, bottom=0.45)
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444401",
            page_num=1,
            bbox=tbl_bbox,
        )
        cap = _make_caption(
            elem_id="44444444-4444-4444-4444-444444444402",
            content="Table 1: Results",
            page_num=1,
            bbox=cap_bbox,
        )
        registry = _build_registry([tbl, cap])

        rels = generate_table_relationships(tbl, registry)
        has_caption_rels = [r for r in rels if r.relationship_type == "has_caption"]
        assert len(has_caption_rels) >= 1
        assert has_caption_rels[0].source_id == tbl.element_id
        assert has_caption_rels[0].target_id == cap.element_id

    def test_has_caption_mentions_table(self) -> None:
        """Caption mentions 'Table' — relationship created even if far."""
        tbl_bbox = _bbox(left=0.1, top=0.2, right=0.5, bottom=0.4)
        cap_bbox = _bbox(left=0.7, top=0.8, right=0.9, bottom=0.85)  # far
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444411",
            page_num=1,
            bbox=tbl_bbox,
        )
        cap = _make_caption(
            elem_id="44444444-4444-4444-4444-444444444412",
            content="Table 2: Summary",
            page_num=1,
            bbox=cap_bbox,
        )
        registry = _build_registry([tbl, cap])
        rels = generate_table_relationships(tbl, registry)
        has_caption = [r for r in rels if r.relationship_type == "has_caption"]
        assert len(has_caption) >= 1

    def test_describes_nearby_text_mentioning_table(self) -> None:
        """Text block near table mentioning 'table' gets describes."""
        tbl_bbox = _bbox(left=0.1, top=0.3, right=0.5, bottom=0.5)
        txt_bbox = _bbox(left=0.1, top=0.15, right=0.5, bottom=0.28)
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444421",
            page_num=1,
            bbox=tbl_bbox,
        )
        txt = _make_text_block(
            elem_id="44444444-4444-4444-4444-444444444422",
            content="As shown in Table 3, the results indicate...",
            page_num=1,
            bbox=txt_bbox,
        )
        registry = _build_registry([tbl, txt])
        rels = generate_table_relationships(tbl, registry)
        describes = [r for r in rels if r.relationship_type == "describes"]
        assert len(describes) >= 1
        assert describes[0].source_id == txt.element_id
        assert describes[0].target_id == tbl.element_id

    def test_refers_to_footnote_nearby(self) -> None:
        """Footnote spatially close to table."""
        tbl_bbox = _bbox(left=0.1, top=0.2, right=0.5, bottom=0.4)
        fn_bbox = _bbox(left=0.1, top=0.38, right=0.3, bottom=0.42)
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444431",
            page_num=1,
            bbox=tbl_bbox,
        )
        fn = _make_footnote(
            elem_id="44444444-4444-4444-4444-444444444432",
            content="Note: values in thousands.",
            page_num=1,
            bbox=fn_bbox,
        )
        registry = _build_registry([tbl, fn])
        rels = generate_table_relationships(tbl, registry)
        refers_to = [r for r in rels if r.relationship_type == "refers_to"]
        assert len(refers_to) >= 1

    def test_no_self_reference(self) -> None:
        """Table should not relate to itself."""
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444441",
            page_num=1,
        )
        registry = _build_registry([tbl])
        rels = generate_table_relationships(tbl, registry)
        self_refs = [
            r
            for r in rels
            if r.source_id == tbl.element_id and r.target_id == tbl.element_id
        ]
        assert len(self_refs) == 0

    def test_no_duplicate_relationships(self) -> None:
        """Calling generate_table_relationships twice on same inputs
        produces same relationships (deterministic IDs avoid dupes)."""
        tbl_bbox = _bbox(left=0.1, top=0.2, right=0.5, bottom=0.4)
        cap_bbox = _bbox(left=0.1, top=0.41, right=0.5, bottom=0.45)
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444451",
            page_num=1,
            bbox=tbl_bbox,
        )
        cap = _make_caption(
            elem_id="44444444-4444-4444-4444-444444444452",
            content="Table 4: Data",
            page_num=1,
            bbox=cap_bbox,
        )
        registry = _build_registry([tbl, cap])
        rels1 = generate_table_relationships(tbl, registry)
        rels2 = generate_table_relationships(tbl, registry)
        ids1 = {r.relationship_id for r in rels1}
        ids2 = {r.relationship_id for r in rels2}
        assert ids1 == ids2

    def test_different_page_no_relationship(self) -> None:
        """Elements on different pages should not get relationships."""
        tbl = _make_table(
            elem_id="44444444-4444-4444-4444-444444444461",
            page_num=1,
        )
        cap = _make_caption(
            elem_id="44444444-4444-4444-4444-444444444462",
            content="Table 5",
            page_num=2,  # Different page
        )
        registry = _build_registry([tbl, cap])
        rels = generate_table_relationships(tbl, registry)
        assert len(rels) == 0  # No same-page elements


# ===================================================================
#  7.  process_tables — document-wide helper
# ===================================================================


class TestProcessTables:
    """Document-wide table processing helper."""

    def test_processes_all_tables(self) -> None:
        doc = _make_doc(page_count=2)
        tbl1 = _make_table(
            elem_id="55555555-5555-5555-5555-555555555501",
            page_num=1,
            reading_order=0,
        )
        tbl2 = _make_table(
            elem_id="55555555-5555-5555-5555-555555555502",
            page_num=2,
            reading_order=0,
        )
        # Add elements to doc
        elements = {
            str(tbl1.element_id): tbl1,
            str(tbl2.element_id): tbl2,
        }
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([tbl1, tbl2])

        dl_doc = {
            "data": [
                {"Col1": "A", "Col2": "B"},
            ]
        }
        result = process_tables(doc, registry, dl_doc=dl_doc)
        # Both tables should be processed
        for elem in result.elements.values():
            assert isinstance(elem, TableSchema)
            assert elem.markdown != ""  # Should be populated
            assert elem.html != ""
            assert elem.row_count >= 1

    def test_preserves_non_table_elements(self) -> None:
        doc = _make_doc()
        tbl = _make_table(
            elem_id="55555555-5555-5555-5555-555555555511",
            page_num=1,
        )
        txt = _make_text_block(
            elem_id="55555555-5555-5555-5555-555555555512",
            content="Some text.",
            page_num=1,
        )
        elements = {
            str(tbl.element_id): tbl,
            str(txt.element_id): txt,
        }
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([tbl, txt])

        result = process_tables(doc, registry)
        assert str(tbl.element_id) in result.elements
        assert str(txt.element_id) in result.elements
        text_elem = result.elements[str(txt.element_id)]
        assert text_elem.content == "Some text."

    def test_empty_document(self) -> None:
        doc = _make_doc()
        registry = _build_registry([])
        result = process_tables(doc, registry)
        assert len(result.elements) == 0
        assert len(result.relationships) == 0

    def test_no_duplicate_relationships_across_tables(self) -> None:
        """Multiple tables produce non-duplicate relationships."""
        doc = _make_doc(page_count=1)
        tbl1 = _make_table(
            elem_id="55555555-5555-5555-5555-555555555521",
            page_num=1,
            reading_order=0,
        )
        tbl2 = _make_table(
            elem_id="55555555-5555-5555-5555-555555555522",
            page_num=1,
            reading_order=1,
            bbox=_bbox(left=0.6, top=0.2, right=0.9, bottom=0.4),
        )
        cap = _make_caption(
            elem_id="55555555-5555-5555-5555-555555555523",
            content="Table 6: Comparison",
            page_num=1,
            reading_order=2,
            bbox=_bbox(left=0.1, top=0.5, right=0.5, bottom=0.55),
        )
        elements = {
            str(tbl1.element_id): tbl1,
            str(tbl2.element_id): tbl2,
            str(cap.element_id): cap,
        }
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([tbl1, tbl2, cap])

        result = process_tables(doc, registry)
        rel_ids = [r.relationship_id for r in result.relationships]
        assert len(rel_ids) == len(set(rel_ids)), "Duplicate relationship IDs found"


# ===================================================================
#  8.  Edge cases
# ===================================================================


class TestEdgeCases:
    """Additional edge cases."""

    def test_process_table_preserves_spanning_flags(self) -> None:
        """Existing is_spanning / span_group_id should survive."""
        elem = _make_table(
            is_spanning=True,
            span_group_id="test-span-id",
        )
        dl_doc = {
            "data": [
                {"A": "1"},
            ]
        }
        result = process_table(elem, dl_doc=dl_doc)
        assert result.is_spanning is True
        assert result.span_group_id == "test-span-id"

    def test_detect_spanning_mixed_section_and_col_count(self) -> None:
        """Complex case with multiple groups."""
        tables = (
            [
                # Group 1: spanning, section "2.1", 2 cols
                _make_table(
                    elem_id=f"66666666-6666-6666-6666-66666666660{i}",
                    page_num=i,
                    section_path="2.1",
                    col_count=2,
                    headers=["A", "B"],
                )
                for i in range(1, 3)
            ]
            + [
                # Group 2: spanning, section "2.2", 3 cols
                _make_table(
                    elem_id=f"66666666-6666-6666-6666-66666666661{i}",
                    page_num=i,
                    section_path="2.2",
                    col_count=3,
                    headers=["X", "Y", "Z"],
                )
                for i in range(1, 3)
            ]
            + [
                # Lone table, not spanning
                _make_table(
                    elem_id="66666666-6666-6666-6666-666666666620",
                    page_num=5,
                    section_path="3",
                    col_count=1,
                    headers=["Single"],
                ),
            ]
        )
        result = detect_spanning_tables(tables)
        # Group 1: both spanning
        assert result[0].is_spanning is True
        assert result[1].is_spanning is True
        # Group 2: both spanning
        assert result[2].is_spanning is True
        assert result[3].is_spanning is True
        # Lone table: not spanning
        assert result[4].is_spanning is False

        # Different groups have different span_group_id
        ids = {r.span_group_id for r in result if r.is_spanning}
        assert len(ids) == 2
