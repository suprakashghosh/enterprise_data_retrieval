"""
Tests for ``src.validation`` (Sub-Task 5 — Validate Docling Outputs).

Uses fakes throughout so that tests are fast and do **not** require a real
Docling installation.

Covers:
- Public imports work without invoking Docling.
- Happy path passes all critical checks.
- Page count mismatch fails.
- Empty document fails.
- All invalid bboxes fails.
- Individual invalid page/bbox warning does not fail when other critical
  checks pass.
- Unknown item type warns.
- Reading order gap/non-monotonic warns but report remains valid.
- Zero-row/zero-column table warns but report remains valid.
- No tables does not fail.
- Report serialises to JSON/dict.
- Malformed weird input returns report instead of crashing.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import pytest

from src.schemas import DocumentSchema
from src.validation import (
    DoclingValidator,
    ValidationCheck,
    ValidationReport,
    validate_docling_output,
)

# ===================================================================
#  Fakes — simulate Docling objects without importing docling
# ===================================================================


class FakeBBox:
    """A fake bounding-box object supporting multiple representations."""

    def __init__(
        self,
        left: float = 0,
        top: float = 0,
        right: float = 100,
        bottom: float = 200,
    ) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class FakePage:
    """Simulates a Docling page object."""

    def __init__(
        self,
        page_num: int = 1,
        width: float = 612,
        height: float = 792,
        items: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        tables: Optional[List[Any]] = None,
    ) -> None:
        self.page_num = page_num
        self.width = width
        self.height = height
        self._items = items or []
        self._texts = texts or []
        self._tables = tables or []

    @property
    def items(self) -> List[Any]:
        return self._items

    @property
    def texts(self) -> List[Any]:
        return self._texts

    @property
    def tables(self) -> List[Any]:
        return self._tables


class FakeDoclingItem:
    """A generic fake Docling item (text, table, figure, etc.)."""

    def __init__(
        self,
        type: str = "text",  # noqa: A002
        text: str = "Hello",
        bbox: Any = None,
        page_num: int = 1,
        order: Optional[int] = None,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.type = type
        self.text = text
        self._bbox = bbox
        self.page_num = page_num
        self.order = order
        if caption is not None:
            self.caption = caption
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def bbox(self) -> Any:
        return self._bbox


class FakeDoclingDocument:
    """A fake Docling document for testing validation.

    Provides ``pages`` (as a dict), optional ``export_to_dict`` /
    ``export_to_markdown``, and ``page_count``.
    """

    def __init__(
        self,
        pages: Optional[Dict[int, FakePage]] = None,
        page_count: Optional[int] = None,
        texts: Optional[List[Any]] = None,
        tables: Optional[List[Any]] = None,
        markdown_output: str = "",
    ) -> None:
        self._pages = pages if pages is not None else {}
        self._page_count = page_count
        self._texts = texts or []
        self._tables = tables or []
        self._markdown_output = markdown_output

    @property
    def pages(self) -> Dict[int, FakePage]:
        return self._pages

    @property
    def page_count(self) -> Optional[int]:
        return self._page_count

    @property
    def texts(self) -> List[Any]:
        return self._texts

    @property
    def tables(self) -> List[Any]:
        return self._tables

    def export_to_dict(self) -> Dict[str, Any]:
        return {"type": "document", "pages": list(self._pages.keys())}

    def export_to_markdown(self) -> str:
        return self._markdown_output


# ===================================================================
#  Helper: build a minimal doc schema with a given page count
# ===================================================================


def _make_doc(page_count: int = 1, doc_id: Optional[str] = None) -> DocumentSchema:
    """Return a minimal ``DocumentSchema`` with the given *page_count*."""
    return DocumentSchema(
        doc_id=UUID(doc_id or "11111111-1111-1111-1111-111111111111"),
        title="Test Document",
        source_path="/fake/test.pdf",
        file_hash="abcd",
        page_count=page_count,
        created_at=datetime(2025, 1, 1),
    )


# ===================================================================
#  Fixtures
# ===================================================================


@pytest.fixture
def valid_doc() -> DocumentSchema:
    """A ``DocumentSchema`` with 2 pages (matching the happy-path doc)."""
    return _make_doc(page_count=2)


@pytest.fixture
def one_page_doc() -> DocumentSchema:
    """A ``DocumentSchema`` with 1 page."""
    return _make_doc(page_count=1)


@pytest.fixture
def valid_dl_doc() -> FakeDoclingDocument:
    """A well-formed fake Docling document with two pages of content."""
    page1 = FakePage(
        page_num=1,
        items=[
            FakeDoclingItem(type="text", text="Page 1 text", page_num=1, order=0),
            FakeDoclingItem(
                type="table",
                text="",
                page_num=1,
                order=1,
                rows=3,
                cols=4,
                caption="Table 1: results",
            ),
        ],
    )
    page2 = FakePage(
        page_num=2,
        items=[
            FakeDoclingItem(type="text", text="Page 2 text", page_num=2, order=0),
            FakeDoclingItem(
                type="figure",
                text="Figure 1",
                page_num=2,
                order=1,
                caption="A caption",
            ),
        ],
    )
    return FakeDoclingDocument(
        pages={1: page1, 2: page2},
        page_count=2,
        markdown_output="# Test\n\nSome content.",
    )


# ===================================================================
#  1.  Public imports
# ===================================================================


def test_public_imports() -> None:
    """All public API symbols are importable from ``src.validation``."""
    assert ValidationCheck is not None
    assert ValidationReport is not None
    assert DoclingValidator is not None
    assert callable(validate_docling_output)


# ===================================================================
#  2.  Happy path
# ===================================================================


class TestHappyPath:
    """A well-formed document passes all critical checks."""

    def test_report_is_valid(
        self, valid_dl_doc: FakeDoclingDocument, valid_doc: DocumentSchema
    ) -> None:
        report = validate_docling_output(valid_dl_doc, valid_doc)
        assert report.is_valid is True, f"Expected valid, got errors: {report.errors}"

    def test_all_checks_present(
        self, valid_dl_doc: FakeDoclingDocument, valid_doc: DocumentSchema
    ) -> None:
        report = validate_docling_output(valid_dl_doc, valid_doc)
        check_names = {c.check_name for c in report.checks}
        expected = {
            "page_count",
            "not_empty",
            "all_bboxes_valid",
            "page_number_valid",
            "item_types",
            "individual_bboxes",
            "reading_order",
            "tables",
            "captions",
        }
        assert check_names == expected, f"Missing checks: {expected - check_names}"

    def test_no_errors(
        self, valid_dl_doc: FakeDoclingDocument, valid_doc: DocumentSchema
    ) -> None:
        report = validate_docling_output(valid_dl_doc, valid_doc)
        assert len(report.errors) == 0, f"Unexpected errors: {report.errors}"

    def test_summary_present(
        self, valid_dl_doc: FakeDoclingDocument, valid_doc: DocumentSchema
    ) -> None:
        report = validate_docling_output(valid_dl_doc, valid_doc)
        assert "PASSED" in report.summary


# ===================================================================
#  3.  Page count mismatch
# ===================================================================


class TestPageCountMismatch:
    """Page count mismatch causes a critical failure."""

    def test_mismatch_detected(self, valid_dl_doc: FakeDoclingDocument) -> None:
        """Docling has 2 pages, schema says 5 → critical failure."""
        doc = _make_doc(page_count=5, doc_id="22222222-2222-2222-2222-222222222222")
        report = validate_docling_output(valid_dl_doc, doc)
        assert report.is_valid is False
        check = next(c for c in report.checks if c.check_name == "page_count")
        assert check.passed is False
        assert check.severity == "critical"
        assert "mismatch" in check.message.lower()

    def test_match_when_doc_page_count_zero(
        self, valid_dl_doc: FakeDoclingDocument
    ) -> None:
        """When ``doc.page_count <= 0``, the check is skipped."""
        doc = _make_doc(page_count=0, doc_id="33333333-3333-3333-3333-333333333333")
        report = validate_docling_output(valid_dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "page_count")
        assert check.passed is True


# ===================================================================
#  4.  Empty document
# ===================================================================


class TestEmptyDocument:
    """An empty Docling document fails the critical ``not_empty`` check."""

    def test_no_pages_fails(self) -> None:
        doc = _make_doc(page_count=2)
        dl_doc = FakeDoclingDocument(pages={}, page_count=0)
        report = validate_docling_output(dl_doc, doc)
        assert report.is_valid is False
        check = next(c for c in report.checks if c.check_name == "not_empty")
        assert check.passed is False
        assert check.severity == "critical"

    def test_pages_with_no_items_fails(self) -> None:
        """Pages exist but contain no items and no exported content."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(page_num=1, items=[])
        dl_doc = FakeDoclingDocument(pages={1: page1}, page_count=1)
        report = validate_docling_output(dl_doc, doc)
        assert report.is_valid is False

    def test_exported_markdown_counts_as_content(self) -> None:
        """Non-empty markdown export counts as content."""
        doc = _make_doc(page_count=0)
        dl_doc = FakeDoclingDocument(
            pages={},
            page_count=0,
            markdown_output="# Has content",
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "not_empty")
        assert check.passed is True

    def test_export_to_dict_alone_not_content(self) -> None:
        """A bare ``export_to_dict`` returning a trivial dict is not content."""
        doc = _make_doc(page_count=0)
        dl_doc = FakeDoclingDocument(pages={}, page_count=0)
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "not_empty")
        assert check.passed is False, (
            "Trivial export_to_dict should not count as content"
        )


# ===================================================================
#  5.  All invalid bounding boxes
# ===================================================================


class TestAllInvalidBboxes:
    """When every discovered bbox is invalid, the report fails critically."""

    def test_all_invalid_fails(self) -> None:
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="bad",
                    bbox=FakeBBox(left=-1, top=0, right=100, bottom=200),
                ),
                FakeDoclingItem(
                    type="text",
                    text="also bad",
                    bbox=FakeBBox(left=0, top=0, right=-1, bottom=200),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "all_bboxes_valid")
        assert check.passed is False
        assert check.severity == "critical"
        assert report.is_valid is False

    def test_some_valid_some_invalid_passes_critical(self) -> None:
        """Mixed valid/invalid bboxes pass the critical check but warn."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="good",
                    bbox=FakeBBox(left=0, top=0, right=100, bottom=200),
                ),
                FakeDoclingItem(
                    type="text",
                    text="bad",
                    bbox=FakeBBox(left=-5, top=0, right=100, bottom=200),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        critical_check = next(
            c for c in report.checks if c.check_name == "all_bboxes_valid"
        )
        assert critical_check.passed is True  # not all are invalid
        individual_check = next(
            c for c in report.checks if c.check_name == "individual_bboxes"
        )
        assert individual_check.passed is False
        assert individual_check.severity == "warning"


# ===================================================================
#  6.  Individual invalid page/bbox warnings
# ===================================================================


class TestIndividualWarnings:
    """Individual invalid page numbers or bboxes generate warnings only."""

    def test_invalid_page_number_warns(self) -> None:
        """Item with page number outside range should warn but not fail."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="p1", page_num=1),
                FakeDoclingItem(type="text", text="p99", page_num=99),  # invalid
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "page_number_valid")
        assert check.passed is False
        assert check.severity == "warning"
        # Overall is_valid should be True (no critical failures)
        assert report.is_valid is True

    def test_some_invalid_bboxes_warn_but_report_valid(self) -> None:
        """A mix of valid/invalid bboxes produces warnings but the report stays valid."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="good",
                    bbox=FakeBBox(left=0, top=0, right=100, bottom=200),
                ),
                FakeDoclingItem(
                    type="text",
                    text="bad",
                    bbox=FakeBBox(left=0, top=0, right=-10, bottom=200),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        individual = next(
            c for c in report.checks if c.check_name == "individual_bboxes"
        )
        assert individual.passed is False
        assert individual.severity == "warning"
        assert report.is_valid is True


# ===================================================================
#  7.  Unknown item types
# ===================================================================


class TestUnknownItemTypes:
    """Unrecognised item types generate warnings."""

    def test_unknown_type_warns(self) -> None:
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="normal"),
                FakeDoclingItem(type="gizmo", text="weird"),  # unknown
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "item_types")
        assert check.passed is False
        assert check.severity == "warning"
        assert "gizmo" in check.details
        assert report.is_valid is True


# ===================================================================
#  8.  Reading order gaps / non-monotonic
# ===================================================================


class TestReadingOrder:
    """Reading-order gaps and non-monotonic order generate warnings."""

    def test_gaps_warn(self) -> None:
        """Items with order values 0, 1, 5 — gap between 1 and 5."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="a", page_num=1, order=0),
                FakeDoclingItem(type="text", text="b", page_num=1, order=1),
                FakeDoclingItem(type="text", text="c", page_num=1, order=5),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "reading_order")
        assert check.passed is False
        assert check.severity == "warning"
        assert "gap" in check.message.lower() or "missing" in check.message.lower()
        assert report.is_valid is True

    def test_non_monotonic_warns(self) -> None:
        """Duplicate order values (non-monotonic / duplicate)."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="a", page_num=1, order=0),
                FakeDoclingItem(
                    type="text", text="b", page_num=1, order=0
                ),  # duplicate
                FakeDoclingItem(type="text", text="c", page_num=1, order=2),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "reading_order")
        assert check.passed is False
        assert check.severity == "warning"
        assert report.is_valid is True

    def test_no_order_info_passes(self) -> None:
        """Items without explicit order fields should pass (info)."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="a"),  # no order field
                FakeDoclingItem(type="text", text="b"),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "reading_order")
        assert check.passed is True


# ===================================================================
#  9.  Tables with zero rows/columns
# ===================================================================


class TestTables:
    """Tables with zero rows or columns generate warnings."""

    def test_zero_rows_warns(self) -> None:
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="table", text="", page_num=1, rows=0, cols=3),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "tables")
        assert check.passed is False
        assert check.severity == "warning"
        assert report.is_valid is True

    def test_zero_cols_warns(self) -> None:
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="table", text="", page_num=1, rows=3, cols=0),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "tables")
        assert check.passed is False
        assert check.severity == "warning"

    def test_no_tables_passes(self) -> None:
        """A document with no tables should pass."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="only text"),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        check = next(c for c in report.checks if c.check_name == "tables")
        assert check.passed is True


# ===================================================================
#  10.  JSON serialisation
# ===================================================================


class TestSerialisation:
    """ValidationReport serialises to JSON and dict."""

    def test_model_dump(
        self, valid_dl_doc: FakeDoclingDocument, valid_doc: DocumentSchema
    ) -> None:
        report = validate_docling_output(valid_dl_doc, valid_doc)
        data = report.model_dump()
        assert isinstance(data, dict)
        assert "doc_id" in data
        assert "is_valid" in data
        assert "checks" in data
        assert "errors" in data
        assert "warnings" in data
        assert "summary" in data
        assert "created_at" in data

    def test_json_roundtrip(
        self, valid_dl_doc: FakeDoclingDocument, valid_doc: DocumentSchema
    ) -> None:
        report = validate_docling_output(valid_dl_doc, valid_doc)
        json_str = report.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["doc_id"] == str(valid_doc.doc_id)
        assert parsed["is_valid"] is True
        assert len(parsed["checks"]) > 0

    def test_failed_report_serialises(self) -> None:
        """A failed report should also serialise cleanly."""
        doc = _make_doc(page_count=2)
        dl_doc = FakeDoclingDocument(pages={}, page_count=0)
        report = validate_docling_output(dl_doc, doc)
        data = report.model_dump()
        assert data["is_valid"] is False
        json_str = report.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["is_valid"] is False


# ===================================================================
#  11.  Malformed / weird input
# ===================================================================


class TestMalformedInput:
    """The validator never crashes on malformed input."""

    def test_none_pages(self) -> None:
        """When pages is ``None``, the validator should still produce a report."""
        doc = _make_doc(page_count=1)
        dl_doc = FakeDoclingDocument(pages=None, page_count=0)  # type: ignore[arg-type]
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)
        assert report.is_valid is False  # empty document

    def test_empty_dict(self) -> None:
        """Feeding a bare dict instead of an object."""
        doc = _make_doc(page_count=1)
        report = validate_docling_output({}, doc)
        assert isinstance(report, ValidationReport)

    def test_list_instead_of_doc(self) -> None:
        """A list is not a valid Docling doc, but should not crash."""
        doc = _make_doc(page_count=1)
        report = validate_docling_output([1, 2, 3], doc)  # type: ignore[arg-type]
        assert isinstance(report, ValidationReport)

    def test_string_instead_of_doc(self) -> None:
        doc = _make_doc(page_count=1)
        report = validate_docling_output("garbage", doc)  # type: ignore[arg-type]
        assert isinstance(report, ValidationReport)

    def test_items_with_no_type_field(self) -> None:
        """Items missing type/label/kind should not crash."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[{"text": "no type here"}],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)
        type_check = next(c for c in report.checks if c.check_name == "item_types")
        assert type_check.passed is True

    def test_items_with_broken_bbox(self) -> None:
        """Items with bboxes that can't be parsed should not crash."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="x", bbox="not-a-bbox"),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)

    def test_bounding_box_as_tuple(self) -> None:
        """Bounding box as a tuple [left, top, right, bottom]."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="x", bbox=(0, 0, 100, 200)),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)
        all_check = next(c for c in report.checks if c.check_name == "all_bboxes_valid")
        assert all_check.passed is True

    def test_bounding_box_as_list(self) -> None:
        """Bounding box as a list [left, top, right, bottom]."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(type="text", text="x", bbox=[0, 0, 100, 200]),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)

    def test_bounding_box_as_l_tr_b_dict(self) -> None:
        """Bounding box as dict with l/t/r/b keys."""
        doc = _make_doc(page_count=1)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="x",
                    bbox={"l": 0, "t": 0, "r": 100, "b": 200},
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(
            pages={1: page1}, page_count=1, markdown_output="x"
        )
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)

    def test_page_with_dict_style(self) -> None:
        """Pages as dicts (exported format) instead of objects."""
        doc = _make_doc(page_count=1)
        dl_doc = {
            "page_count": 1,
            "pages": {
                1: {
                    "page_num": 1,
                    "width": 612,
                    "height": 792,
                    "items": [
                        {"type": "text", "text": "hello", "bbox": [0, 0, 100, 200]},
                    ],
                }
            },
        }
        report = validate_docling_output(dl_doc, doc)
        assert isinstance(report, ValidationReport)


# ===================================================================
#  12.  Individual check methods (unit tests)
# ===================================================================


class TestCheckMethods:
    """Individual DoclingValidator check methods are independently testable."""

    def test_check_page_count_skip_when_unknown(self) -> None:
        """When page_count is None and pages is empty dict, check passes
        with info (undetermined)."""
        doc = _make_doc(page_count=1)
        validator = DoclingValidator()
        dl_doc = FakeDoclingDocument(pages={}, page_count=None)
        check = validator.check_page_count(dl_doc, doc)
        assert check.passed is True
        assert "could not be determined" in check.message

    def test_captions_no_visual_items(self) -> None:
        """captions check passes when there are no visual items."""
        doc = _make_doc(page_count=1)
        validator = DoclingValidator()
        dl_doc = FakeDoclingDocument(
            pages={1: FakePage(page_num=1, items=[])}, page_count=1
        )
        check = validator.check_captions(dl_doc, doc)
        assert check.passed is True
        assert "No visual items" in check.message


# ===================================================================
#  13.  Edge cases
# ===================================================================


def test_pages_as_list() -> None:
    """When pages is a list instead of dict."""
    doc = _make_doc(page_count=2)
    page1 = FakePage(page_num=1, items=[FakeDoclingItem(type="text", text="hello")])
    page2 = FakePage(page_num=2, items=[FakeDoclingItem(type="text", text="world")])
    dl_doc = FakeDoclingDocument(
        pages={1: page1, 2: page2}, page_count=2, markdown_output="x"
    )
    # Force pages to be a list
    dl_doc._pages = [page1, page2]  # type: ignore[assignment]
    report = validate_docling_output(dl_doc, doc)
    assert report.is_valid is True


def test_bbox_without_page_dimensions() -> None:
    """When page dimensions are unavailable, bbox validation uses only
    non-negative and ordering checks."""
    doc = _make_doc(page_count=1)
    page1 = FakePage(page_num=1, width=None, height=None)  # type: ignore[assignment]
    page1._items = [
        FakeDoclingItem(
            type="text",
            text="x",
            bbox=FakeBBox(left=0, top=0, right=100, bottom=200),
        )
    ]
    dl_doc = FakeDoclingDocument(pages={1: page1}, page_count=1, markdown_output="x")
    report = validate_docling_output(dl_doc, doc)
    assert report.is_valid is True
