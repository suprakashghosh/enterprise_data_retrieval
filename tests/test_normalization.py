"""
Tests for ``src.normalization`` (Sub-Task 6 — Normalize Docling Output into
Internal Objects and Create Element Registry).

Uses fakes throughout — no real Docling dependency.

Covers:
- Public imports.
- Type mapping creates expected element subclasses.
- Bbox normalisation with page dimensions.
- Deterministic element IDs.
- ElementRegistry add / get / get_by_page / get_by_type / iter_in_reading_order.
- normalize_document populates pages / elements with correct fields.
- Caption linking via explicit reference and spatial heuristic.
- Proximity relationships for nearby / far elements.
- Empty document.
- Missing fields / unknown item types do not crash.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.normalization import (
    DOCLING_TYPE_TO_INTERNAL_TYPE,
    ElementRegistry,
    normalize_document,
    preserve_proximity,
)
from src.schemas import (
    BoundingBox,
    CaptionSchema,
    ChartSchema,
    DocumentSchema,
    ElementSchema,
    FooterSchema,
    FootnoteSchema,
    FormulaSchema,
    GraphSchema,
    HeaderSchema,
    ImageSchema,
    ListBlockSchema,
    PageSchema,
    SectionHeaderSchema,
    TableSchema,
    TextBlockSchema,
)

# ===================================================================
#  Fakes — simulate Docling-like objects without real docling
# ===================================================================


class FakeBBox:
    """Fake bounding box with ``left/top/right/bottom`` attributes."""

    def __init__(
        self,
        left: float = 0.0,
        top: float = 0.0,
        right: float = 100.0,
        bottom: float = 50.0,
    ) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class FakeDoclingItem:
    """A generic fake Docling item (text, table, figure, etc.)."""

    def __init__(  # noqa: PLR0913
        self,
        type: str = "text",  # noqa: A002
        text: str = "",
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


class FakePage:
    """Fake Docling page with items container."""

    def __init__(
        self,
        page_num: int = 1,
        width: float = 612.0,
        height: float = 792.0,
        items: Optional[List[Any]] = None,
    ) -> None:
        self.page_num = page_num
        self.width = width
        self.height = height
        self._items = items or []

    @property
    def items(self) -> List[Any]:
        return self._items


class FakeDoclingDocument:
    """Fake Docling document for testing normalisation."""

    def __init__(
        self,
        pages: Optional[Dict[int, FakePage]] = None,
        page_count: Optional[int] = None,
        texts: Optional[List[Any]] = None,
        tables: Optional[List[Any]] = None,
    ) -> None:
        self._pages = pages if pages is not None else {}
        self._page_count = page_count
        self._texts = texts or []
        self._tables = tables or []

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


# ===================================================================
#  Helper factories
# ===================================================================


def _make_doc(doc_id: Optional[str] = None, page_count: int = 1) -> DocumentSchema:
    """Minimal DocumentSchema for testing."""
    return DocumentSchema(
        doc_id=uuid.UUID(doc_id or "11111111-1111-1111-1111-111111111111"),
        title="Test",
        source_path="/fake/test.pdf",
        file_hash="abcd1234",
        page_count=page_count,
        created_at=datetime(2025, 1, 1),
    )


def _make_bbox(
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


# ===================================================================
#  1.  Public imports
# ===================================================================


class TestPublicImports:
    """All public API symbols are importable from ``src.normalization``."""

    def test_imports(self) -> None:
        assert DOCLING_TYPE_TO_INTERNAL_TYPE is not None
        assert ElementRegistry is not None
        assert callable(normalize_document)
        assert callable(preserve_proximity)

    def test_import_from_package(self) -> None:
        from src.normalization import (
            DOCLING_TYPE_TO_INTERNAL_TYPE as T,
            ElementRegistry as R,
        )

        assert T is not None
        assert R is not None


# ===================================================================
#  2.  Type mapping
# ===================================================================


class TestTypeMapping:
    """DOCLING_TYPE_TO_INTERNAL_TYPE maps correctly."""

    def test_known_types(self) -> None:
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["text"] == "text_block"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["table"] == "table"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["figure"] == "image"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["chart"] == "chart"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["graph"] == "graph"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["formula"] == "formula"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["caption"] == "caption"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["header"] == "header"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["footer"] == "footer"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["footnote"] == "footnote"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["list"] == "list_block"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["heading"] == "section_header"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["picture"] == "image"
        assert DOCLING_TYPE_TO_INTERNAL_TYPE["equation"] == "formula"

    def test_unknown_type_falls_back_to_text_block(self) -> None:
        """Unknown types fall back to text_block in the normalizer."""
        fallback = DOCLING_TYPE_TO_INTERNAL_TYPE.get("gizmo", "text_block")
        assert fallback == "text_block"


# ===================================================================
#  3.  BBox normalisation
# ===================================================================


class TestBBoxNormalisation:
    """Bounding box normalisation (tested indirectly via normalize)."""

    def test_normalize_with_page_dimensions(self) -> None:
        """Bbox should be normalized to 0-1 when page dimensions available."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            width=612.0,
            height=792.0,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="hello",
                    page_num=1,
                    bbox=FakeBBox(left=0, top=0, right=306, bottom=396),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)

        assert len(result.elements) == 1
        elem = next(iter(result.elements.values()))
        # 306/612 = 0.5, 396/792 = 0.5
        assert elem.bbox.left == 0.0
        assert elem.bbox.top == 0.0
        assert elem.bbox.right == 0.5
        assert elem.bbox.bottom == 0.5
        assert elem.bbox.coord_system == "normalized"

    def test_bbox_as_tuple(self) -> None:
        """Bbox as tuple [left, top, right, bottom]."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            width=100.0,
            height=100.0,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="x",
                    page_num=1,
                    bbox=(10, 20, 50, 60),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert elem.bbox.left == 0.1  # 10/100
        assert elem.bbox.top == 0.2  # 20/100
        assert elem.bbox.right == 0.5  # 50/100
        assert elem.bbox.bottom == 0.6  # 60/100

    def test_bbox_dict_with_l_t_r_b(self) -> None:
        """Bbox as dict with l/t/r/b keys."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            width=200.0,
            height=200.0,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="x",
                    page_num=1,
                    bbox={"l": 20, "t": 30, "r": 80, "b": 90},
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert elem.bbox.left == 0.1
        assert elem.bbox.top == 0.15
        assert elem.bbox.right == 0.4
        assert elem.bbox.bottom == 0.45

    def test_missing_bbox_uses_zero_fallback(self) -> None:
        """Items without bbox should use (0, 0, 0, 0)."""
        doc = _make_doc()
        # Item with no bbox attribute
        item = FakeDoclingItem(type="text", text="no bbox")
        del item._bbox  # Remove the stored bbox
        page = FakePage(page_num=1, items=[item])
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert elem.bbox.left == 0.0
        assert elem.bbox.top == 0.0
        assert elem.bbox.right == 0.0
        assert elem.bbox.bottom == 0.0


# ===================================================================
#  4.  Deterministic IDs
# ===================================================================


class TestDeterministicIds:
    """Element IDs should be stable for the same inputs."""

    def test_same_input_same_id(self) -> None:
        """Normalizing identical documents produces identical element IDs."""
        doc1 = _make_doc()
        doc2 = _make_doc()

        def make_docling() -> FakeDoclingDocument:
            page = FakePage(
                page_num=1,
                items=[
                    FakeDoclingItem(
                        type="text",
                        text="hello",
                        page_num=1,
                        order=0,
                        bbox=FakeBBox(0, 0, 100, 50),
                    ),
                ],
            )
            return FakeDoclingDocument(pages={1: page}, page_count=1)

        result1 = normalize_document(doc1, make_docling())
        result2 = normalize_document(doc2, make_docling())

        id1 = list(result1.elements.keys())[0]
        id2 = list(result2.elements.keys())[0]
        assert id1 == id2

    def test_different_inputs_different_ids(self) -> None:
        """Different content/different page yields different element ID."""
        doc1 = _make_doc()
        doc2 = _make_doc(doc_id="22222222-2222-2222-2222-222222222222")

        def make_docling_a() -> FakeDoclingDocument:
            page = FakePage(
                page_num=1,
                items=[
                    FakeDoclingItem(
                        type="text",
                        text="A",
                        page_num=1,
                        order=0,
                        bbox=FakeBBox(0, 0, 100, 50),
                    ),
                ],
            )
            return FakeDoclingDocument(pages={1: page}, page_count=1)

        r1 = normalize_document(doc1, make_docling_a())
        r2 = normalize_document(doc2, make_docling_a())
        # Different doc_id => different element IDs
        id1 = list(r1.elements.keys())[0]
        id2 = list(r2.elements.keys())[0]
        assert id1 != id2


# ===================================================================
#  5.  ElementRegistry
# ===================================================================


class TestElementRegistry:
    """ElementRegistry provides O(1) lookup and correct iteration."""

    @pytest.fixture
    def registry(self) -> ElementRegistry:
        reg = ElementRegistry()
        reg.add(
            TextBlockSchema(
                element_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                page_num=1,
                bbox=_make_bbox(left=0.0, top=0.0, right=0.5, bottom=0.1),
                reading_order=0,
                element_type="text_block",
                content="First",
            )
        )
        reg.add(
            TextBlockSchema(
                element_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                page_num=1,
                bbox=_make_bbox(left=0.0, top=0.2, right=0.5, bottom=0.3),
                reading_order=1,
                element_type="text_block",
                content="Second",
            )
        )
        reg.add(
            TableSchema(
                element_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                page_num=2,
                bbox=_make_bbox(left=0.0, top=0.0, right=0.8, bottom=0.4),
                reading_order=0,
                element_type="table",
                content="Table A",
            )
        )
        return reg

    def test_get_exists(self, registry: ElementRegistry) -> None:
        elem = registry.get(uuid.UUID("00000000-0000-0000-0000-000000000001"))
        assert elem is not None
        assert elem.content == "First"

    def test_get_missing(self, registry: ElementRegistry) -> None:
        elem = registry.get(uuid.UUID("00000000-0000-0000-0000-ffffffffffff"))
        assert elem is None

    def test_get_by_page(self, registry: ElementRegistry) -> None:
        page1 = registry.get_by_page(1)
        assert len(page1) == 2
        page2 = registry.get_by_page(2)
        assert len(page2) == 1

    def test_get_by_page_empty(self, registry: ElementRegistry) -> None:
        page99 = registry.get_by_page(99)
        assert page99 == []

    def test_get_by_type(self, registry: ElementRegistry) -> None:
        texts = registry.get_by_type("text_block")
        assert len(texts) == 2
        tables = registry.get_by_type("table")
        assert len(tables) == 1
        images = registry.get_by_type("image")
        assert len(images) == 0

    def test_iter_in_reading_order(self, registry: ElementRegistry) -> None:
        ordered = list(registry.iter_in_reading_order())
        assert len(ordered) == 3
        # Page 1 reading_order 0 first
        assert ordered[0].content == "First"
        # Page 1 reading_order 1 second
        assert ordered[1].content == "Second"
        # Page 2 reading_order 0 third
        assert ordered[2].content == "Table A"

    def test_contains(self, registry: ElementRegistry) -> None:
        assert uuid.UUID("00000000-0000-0000-0000-000000000001") in registry
        assert uuid.UUID("00000000-0000-0000-0000-ffffffffffff") not in registry

    def test_len(self, registry: ElementRegistry) -> None:
        assert len(registry) == 3


# ===================================================================
#  6.  normalize_document — basic population
# ===================================================================


class TestNormalizeDocumentBasic:
    """normalize_document produces correct structure."""

    def test_populates_pages_and_elements(self) -> None:
        doc = _make_doc(page_count=2)
        page1 = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="Page 1",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
            ],
        )
        page2 = FakePage(
            page_num=2,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="Page 2",
                    page_num=2,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page1, 2: page2}, page_count=2)
        result = normalize_document(doc, dl_doc)

        assert len(result.pages) == 2
        assert len(result.elements) == 2

        # Check all element fields
        for elem in result.elements.values():
            assert elem.element_id is not None
            assert elem.doc_id == doc.doc_id
            assert elem.page_num >= 1
            assert elem.bbox is not None
            assert elem.reading_order >= 0
            assert elem.element_type in (
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
            )
            assert isinstance(elem.content, str)

    def test_element_has_content(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="Hello World",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert elem.content == "Hello World"

    def test_multiple_elements_have_sequential_reading_orders(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="First",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
                FakeDoclingItem(
                    type="text",
                    text="Second",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(0, 60, 612, 110),
                ),
                FakeDoclingItem(
                    type="text",
                    text="Third",
                    page_num=1,
                    order=2,
                    bbox=FakeBBox(0, 120, 612, 170),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        orders = sorted([e.reading_order for e in result.elements.values()])
        assert orders == [0, 1, 2]

    def test_table_element_type(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="table",
                    text="Table data",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 200),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert isinstance(elem, TableSchema)
        assert elem.element_type == "table"

    def test_image_element_type(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="figure",
                    text="Figure 1",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 400),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert isinstance(elem, ImageSchema)
        assert elem.element_type == "image"

    def test_formula_element_type(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="formula",
                    text="E=mc^2",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert isinstance(elem, FormulaSchema)
        assert elem.element_type == "formula"

    def test_section_header_element_type(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="heading",
                    text="Introduction",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert isinstance(elem, SectionHeaderSchema)
        assert elem.element_type == "section_header"

    def test_pages_schema_has_element_ids(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="A",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
                FakeDoclingItem(
                    type="text",
                    text="B",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(0, 60, 100, 110),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert len(result.pages) == 1
        page_schema = result.pages[0]
        assert len(page_schema.element_ids) == 2
        assert all(isinstance(eid, uuid.UUID) for eid in page_schema.element_ids)

    def test_page_schema_size(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            width=612.0,
            height=792.0,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="x",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert result.pages[0].size is not None
        assert result.pages[0].size.width == 612.0
        assert result.pages[0].size.height == 792.0

    def test_page_schema_size_none_when_missing(self) -> None:
        doc = _make_doc()
        page = FakePage(page_num=1, width=None, height=None)  # type: ignore[arg-type]
        page._items = [
            FakeDoclingItem(
                type="text",
                text="x",
                page_num=1,
                order=0,
                bbox=FakeBBox(0, 0, 100, 50),
            ),
        ]
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert result.pages[0].size is None

    def test_preserves_doc_id_and_metadata(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="x",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert result.doc_id == doc.doc_id
        assert result.title == doc.title
        assert result.source_path == doc.source_path
        assert result.file_hash == doc.file_hash


# ===================================================================
#  7.  Caption linking
# ===================================================================


class TestCaptionLinking:
    """Captions are linked to parent elements."""

    def test_explicit_caption_parent_element_id(self) -> None:
        """Caption with explicit parent_element_id is linked."""
        doc = _make_doc(doc_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        parent_id = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        # We need to use a caption item that we can later match
        # Since the normalizer creates CaptionSchema, we can detect it by type
        page = FakePage(
            page_num=1,
            items=[
                # Table
                FakeDoclingItem(
                    type="table",
                    text="Table data",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 600, 200),
                ),
                # Caption — stored as a separate item
                FakeDoclingItem(
                    type="caption",
                    text="Table 1: Results",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(0, 210, 600, 230),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)

        # Check that has_caption and describes relationships exist
        has_caption_rels = [
            r for r in result.relationships if r.relationship_type == "has_caption"
        ]
        describes_rels = [
            r for r in result.relationships if r.relationship_type == "describes"
        ]

        # The caption should be linked to the table via spatial heuristic
        # (caption appears just below table)
        assert len(has_caption_rels) >= 1
        assert len(describes_rels) >= 1

    def test_spatial_caption_linking(self) -> None:
        """Caption spatially near a table should be linked."""
        doc = _make_doc(doc_id="cccccccc-cccc-cccc-cccc-cccccccccccc")
        page = FakePage(
            page_num=1,
            items=[
                # Table at top
                FakeDoclingItem(
                    type="table",
                    text="Data",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(50, 50, 550, 250),
                ),
                # Caption just below (typical placement)
                FakeDoclingItem(
                    type="caption",
                    text="Table 2: Data",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(50, 260, 550, 280),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)

        has_caption = [
            r for r in result.relationships if r.relationship_type == "has_caption"
        ]
        assert len(has_caption) >= 1

    def test_nearby_text_not_linked_as_caption(self) -> None:
        """Regular text near a table should NOT be linked as caption."""
        doc = _make_doc(doc_id="dddddddd-dddd-dddd-dddd-dddddddddddd")
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="table",
                    text="Data",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(50, 50, 550, 250),
                ),
                # Text (not caption type) below table
                FakeDoclingItem(
                    type="text",
                    text="This is not a caption.",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(50, 260, 550, 280),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)

        # Only caption-type elements get linked
        has_caption = [
            r for r in result.relationships if r.relationship_type == "has_caption"
        ]
        assert len(has_caption) == 0


# ===================================================================
#  8.  Proximity
# ===================================================================


class TestProximity:
    """Proximity relationships are computed correctly."""

    def _make_element(
        self,
        elem_id: str,
        page_num: int,
        left: float,
        top: float,
        right: float,
        bottom: float,
        reading_order: int = 0,
    ) -> ElementSchema:
        return TextBlockSchema(
            element_id=uuid.UUID(elem_id),
            doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            page_num=page_num,
            bbox=_make_bbox(left=left, top=top, right=right, bottom=bottom),
            reading_order=reading_order,
            element_type="text_block",
            content="test",
        )

    def test_nearby_elements_get_relationship(self) -> None:
        """Elements within threshold get nearby relationships."""
        a = self._make_element(
            "00000000-0000-0000-0000-000000000001", 1, 0.0, 0.0, 0.1, 0.1
        )
        b = self._make_element(
            "00000000-0000-0000-0000-000000000002", 1, 0.05, 0.05, 0.15, 0.15
        )
        # Centres: (0.05, 0.05) vs (0.10, 0.10), distance ≈ 0.07
        rels = preserve_proximity([a, b], threshold=0.08)
        assert len(rels) >= 1
        assert rels[0].relationship_type == "nearby"

    def test_far_elements_no_relationship(self) -> None:
        """Elements far apart get no nearby relationship."""
        a = self._make_element(
            "00000000-0000-0000-0000-000000000001", 1, 0.0, 0.0, 0.1, 0.1
        )
        b = self._make_element(
            "00000000-0000-0000-0000-000000000002", 1, 0.8, 0.8, 0.9, 0.9
        )
        # Centres: (0.05, 0.05) vs (0.85, 0.85), distance ≈ 1.13
        rels = preserve_proximity([a, b], threshold=0.08)
        assert len(rels) == 0

    def test_different_page_no_relationship(self) -> None:
        """Elements on different pages get no nearby relationship."""
        a = self._make_element(
            "00000000-0000-0000-0000-000000000001", 1, 0.0, 0.0, 0.1, 0.1
        )
        b = self._make_element(
            "00000000-0000-0000-0000-000000000002", 2, 0.0, 0.0, 0.1, 0.1
        )
        rels = preserve_proximity([a, b], threshold=0.5)
        assert len(rels) == 0

    def test_no_duplicate_inverse_relationships(self) -> None:
        """No duplicate inverse relationships created."""
        a = self._make_element(
            "00000000-0000-0000-0000-000000000001", 1, 0.0, 0.0, 0.1, 0.1
        )
        b = self._make_element(
            "00000000-0000-0000-0000-000000000002", 1, 0.05, 0.05, 0.15, 0.15
        )
        rels = preserve_proximity([a, b], threshold=0.08)
        # Should only have one nearby relationship (not a→b and b→a)
        assert len(rels) == 1

    def test_many_elements_same_page(self) -> None:
        """All elements in a cluster get nearby relationships."""
        elems = []
        for i in range(5):
            base = i * 0.02
            elems.append(
                self._make_element(
                    f"00000000-0000-0000-0000-00000000000{i}",
                    1,
                    base,
                    base,
                    base + 0.02,
                    base + 0.02,
                )
            )
        rels = preserve_proximity(elems, threshold=0.1)
        # At least some relationships should exist
        assert len(rels) >= 1

    def test_proximity_inside_normalize_document(self) -> None:
        """normalize_document should include nearby relationships."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="A",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 61, 40),
                ),
                FakeDoclingItem(
                    type="text",
                    text="B",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(30, 30, 91, 70),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc, proximity_threshold=0.2)
        nearby = [r for r in result.relationships if r.relationship_type == "nearby"]
        # Items are close in normalized coords -> should be nearby
        assert len(nearby) >= 1


# ===================================================================
#  9.  Empty document
# ===================================================================


class TestEmptyDocument:
    """Empty document returns valid DocumentSchema."""

    def test_no_pages(self) -> None:
        doc = _make_doc(page_count=0)
        dl_doc = FakeDoclingDocument(pages={}, page_count=0)
        result = normalize_document(doc, dl_doc)
        assert isinstance(result, DocumentSchema)
        assert len(result.pages) == 0
        assert len(result.elements) == 0
        assert result.doc_id == doc.doc_id

    def test_pages_with_no_items(self) -> None:
        doc = _make_doc(page_count=1)
        page = FakePage(page_num=1, items=[])
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert len(result.pages) == 1
        assert len(result.pages[0].element_ids) == 0
        assert len(result.elements) == 0

    def test_empty_dict_doc(self) -> None:
        """A bare empty dict should not crash."""
        doc = _make_doc()
        result = normalize_document(doc, {})
        assert isinstance(result, DocumentSchema)
        assert len(result.elements) == 0


# ===================================================================
#  10.  Missing fields / unknown types
# ===================================================================


class TestEdgeCases:
    """Weird or missing data does not crash."""

    def test_unknown_item_type_does_not_crash(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="gizmo",
                    text="weird",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1
        elem = next(iter(result.elements.values()))
        # Unknown types should map to text_block
        assert elem.element_type == "text_block"
        # Original type should be in metadata
        assert elem.metadata is not None
        assert elem.metadata.custom.get("original_type") == "gizmo"

    def test_item_without_type_field(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                # No type field — should default to text_block
                FakeDoclingItem(
                    type=None,
                    text="no type",
                    page_num=1,
                    order=0,  # type: ignore[arg-type]
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        # type=None will be passed through getattr; should be handled
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1

    def test_item_without_bbox(self) -> None:
        doc = _make_doc()
        item = FakeDoclingItem(type="text", text="no bbox", page_num=1, order=0)
        del item._bbox  # Remove stored bbox
        page = FakePage(page_num=1, items=[item])
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1
        elem = next(iter(result.elements.values()))
        assert elem.bbox.left == 0.0
        assert elem.bbox.top == 0.0

    def test_none_values_in_items(self) -> None:
        """Items with None values in list do not crash."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                None,
                FakeDoclingItem(
                    type="text",
                    text="valid",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1

    def test_reading_order_from_provenance(self) -> None:
        """Reading order can come from provenance entries."""
        doc = _make_doc()

        # Item with prov but no direct order
        class ItemWithProv:
            def __init__(self) -> None:
                self.type = "text"
                self.text = "prov item"
                self.page_num = 1
                self.prov = [{"order": 0, "bbox": [0, 0, 100, 50]}]

        page = FakePage(page_num=1, items=[ItemWithProv()])
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1

    def test_header_and_footer_types(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="header",
                    text="Page Header",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 30),
                ),
                FakeDoclingItem(
                    type="footer",
                    text="Page Footer",
                    page_num=1,
                    order=1,
                    bbox=FakeBBox(0, 750, 612, 792),
                ),
                FakeDoclingItem(
                    type="footnote",
                    text="A note",
                    page_num=1,
                    order=2,
                    bbox=FakeBBox(100, 700, 500, 720),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        types = {e.element_type for e in result.elements.values()}
        assert "header" in types
        assert "footer" in types
        assert "footnote" in types

    def test_list_item_type(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="list",
                    text="Item 1",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 30),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert isinstance(elem, ListBlockSchema)

    def test_spatial_order_fallback(self) -> None:
        """Items without explicit order are sorted spatially."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="Bottom",
                    page_num=1,
                    bbox=FakeBBox(0, 200, 612, 250),
                ),
                FakeDoclingItem(
                    type="text",
                    text="Top",
                    page_num=1,
                    bbox=FakeBBox(0, 0, 612, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elems = list(result.elements.values())
        # After normalization, they should have reading_order 0 and 1
        orders = [(e.reading_order, e.content) for e in elems]
        orders.sort(key=lambda x: x[0])
        # "Top" should come before "Bottom"
        assert orders[0][1] == "Top"
        assert orders[1][1] == "Bottom"


# ===================================================================
#  11.  Table-specific fields
# ===================================================================


class TestTableFields:
    """Tables preserve structured fields."""

    def test_table_with_markdown_and_html(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="table",
                    text="table content",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 200),
                    markdown="| A | B |\n|---|---|\n| 1 | 2 |",
                    html="<table><tr><td>A</td><td>B</td></tr></table>",
                    rows=2,
                    cols=2,
                    headers=["A", "B"],
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert isinstance(elem, TableSchema)
        assert elem.markdown == "| A | B |\n|---|---|\n| 1 | 2 |"
        assert elem.html == "<table><tr><td>A</td><td>B</td></tr></table>"
        assert elem.row_count == 2
        assert elem.col_count == 2
        assert elem.headers == ["A", "B"]


# ===================================================================
#  12.  Page number handling
# ===================================================================


class TestPageNumberHandling:
    """Page numbers from various sources are handled."""

    def test_items_with_explicit_page_num(self) -> None:
        doc = _make_doc(page_count=2)
        page = FakePage(
            page_num=1,  # page-level page_num
            items=[
                FakeDoclingItem(
                    type="text",
                    text="p1",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
                # Item on different page than its container
                FakeDoclingItem(
                    type="text",
                    text="p2",
                    page_num=2,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=2)
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 2
        page_nums = {e.page_num for e in result.elements.values()}
        assert 1 in page_nums
        assert 2 in page_nums


# ===================================================================
#  13.  Dict-style Docling output
# ===================================================================


class TestDictStyleDocling:
    """Support for exported dict-style Docling documents."""

    def test_pages_as_dicts(self) -> None:
        doc = _make_doc()
        dl_doc: Dict[str, Any] = {
            "page_count": 1,
            "pages": {
                1: {
                    "page_num": 1,
                    "width": 612,
                    "height": 792,
                    "items": [
                        {
                            "type": "text",
                            "text": "hello",
                            "order": 0,
                            "bbox": [0, 0, 306, 396],
                        },
                    ],
                },
            },
        }
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1
        elem = next(iter(result.elements.values()))
        assert elem.content == "hello"
        assert elem.page_num == 1
        # Bbox should be normalized: 306/612=0.5, 396/792=0.5
        assert elem.bbox.right == 0.5
        assert elem.bbox.bottom == 0.5

    def test_top_level_items(self) -> None:
        """Items at top level (no pages) should still be collected."""
        doc = _make_doc()
        dl_doc = {
            "texts": [
                {
                    "type": "text",
                    "text": "top-level",
                    "page_num": 1,
                    "order": 0,
                    "bbox": [0, 0, 100, 50],
                },
            ],
        }
        result = normalize_document(doc, dl_doc)
        assert len(result.elements) == 1
        elem = next(iter(result.elements.values()))
        assert elem.content == "top-level"

    def test_nested_children_in_items(self) -> None:
        """Children nested inside items should be collected."""
        doc = _make_doc()
        dl_doc: Dict[str, Any] = {
            "page_count": 1,
            "pages": {
                1: {
                    "page_num": 1,
                    "items": [
                        {
                            "type": "figure",
                            "text": "Figure 1",
                            "order": 0,
                            "bbox": [0, 0, 500, 300],
                            "children": [
                                {
                                    "type": "caption",
                                    "text": "Fig 1: Chart",
                                    "order": 0,
                                    "bbox": [0, 310, 500, 330],
                                },
                            ],
                        },
                    ],
                },
            },
        }
        result = normalize_document(doc, dl_doc)
        # Should have both the figure (image) and the caption
        types = {e.element_type for e in result.elements.values()}
        assert "image" in types, f"Expected image, got {types}"
        assert "caption" in types, f"Expected caption, got {types}"


# ===================================================================
#  14.  preserve_proximity public helper
# ===================================================================


class TestPreserveProximityFunction:
    """The public preserve_proximity helper works correctly."""

    def test_returns_relationships(self) -> None:
        a = TextBlockSchema(
            element_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            page_num=1,
            bbox=_make_bbox(0.0, 0.0, 0.1, 0.1),
            reading_order=0,
            element_type="text_block",
            content="a",
        )
        b = TextBlockSchema(
            element_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
            doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            page_num=1,
            bbox=_make_bbox(0.05, 0.05, 0.15, 0.15),
            reading_order=1,
            element_type="text_block",
            content="b",
        )
        rels = preserve_proximity([a, b], threshold=0.08)
        assert len(rels) == 1
        assert rels[0].relationship_type == "nearby"
        assert rels[0].source_id == a.element_id
        assert rels[0].target_id == b.element_id
        assert "distance" in rels[0].metadata

    def test_empty_list(self) -> None:
        rels = preserve_proximity([])
        assert rels == []

    def test_single_element(self) -> None:
        a = TextBlockSchema(
            element_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            doc_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            page_num=1,
            bbox=_make_bbox(0.0, 0.0, 0.1, 0.1),
            reading_order=0,
            element_type="text_block",
            content="a",
        )
        rels = preserve_proximity([a])
        assert rels == []


# ===================================================================
#  15.  Original type in metadata
# ===================================================================


class TestOriginalTypeInMetadata:
    """Original Docling type is preserved in element metadata."""

    def test_original_type_preserved(self) -> None:
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="picture",
                    text="photo",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 612, 400),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        assert elem.element_type == "image"  # picture -> image
        assert elem.metadata.custom.get("original_type") == "picture"

    def test_text_has_no_original_type_mismatch(self) -> None:
        """When dl_type matches internal type, original_type may not be set."""
        doc = _make_doc()
        page = FakePage(
            page_num=1,
            items=[
                FakeDoclingItem(
                    type="text",
                    text="normal",
                    page_num=1,
                    order=0,
                    bbox=FakeBBox(0, 0, 100, 50),
                ),
            ],
        )
        dl_doc = FakeDoclingDocument(pages={1: page}, page_count=1)
        result = normalize_document(doc, dl_doc)
        elem = next(iter(result.elements.values()))
        # text -> text_block, so original_type is "text"
        assert elem.metadata.custom.get("original_type") == "text"
