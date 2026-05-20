"""
Tests for the ``src.schemas`` package (Sub-Task 1).

Covers:
- Public import paths.
- ``DocumentSchema`` instantiation with all fields populated.
- Deterministic UUID generation (same input → same ID, different input → different ID).
- All relationship types accepted by ``RelationshipSchema``.
- Invalid relationship types rejected.
- Element subclass construction.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# 1.  Public imports
# ---------------------------------------------------------------------------


def test_public_imports() -> None:
    """Verify that all key classes and helpers are importable from
    ``src.schemas``."""
    from src.schemas import (
        BoundingBox,
        ChunkMetadata,
        ChunkSchema,
        DocumentMetadata,
        DocumentSchema,
        ElementMetadata,
        ElementSchema,
        Point,
        RelationshipSchema,
        Size,
        TableSchema,
        make_chunk_id,
        make_doc_id,
        make_element_id,
        make_relationship_id,
    )

    # Smoke: every symbol resolves to the expected type.
    assert DocumentSchema is not None
    assert ElementSchema is not None
    assert TableSchema is not None
    assert ChunkSchema is not None
    assert RelationshipSchema is not None
    assert BoundingBox is not None
    assert Size is not None
    assert Point is not None
    assert DocumentMetadata is not None
    assert ElementMetadata is not None
    assert ChunkMetadata is not None
    assert callable(make_doc_id)
    assert callable(make_element_id)
    assert callable(make_chunk_id)
    assert callable(make_relationship_id)


# ---------------------------------------------------------------------------
# 2.  DocumentSchema instantiation
# ---------------------------------------------------------------------------


def test_document_schema_full_instantiation() -> None:
    """Create a ``DocumentSchema`` with every field populated and verify
    all values are stored correctly."""
    from src.schemas import (
        BoundingBox,
        ChunkMetadata,
        ChunkSchema,
        DocumentMetadata,
        DocumentSchema,
        ElementMetadata,
        PageSchema,
        RelationshipSchema,
        SectionSchema,
        Size,
        TableSchema,
        TextBlockSchema,
        make_relationship_id,
    )

    doc_id = uuid.uuid4()
    element_id = uuid.uuid4()
    created = datetime(2025, 6, 1, 12, 0, 0)

    # --- Element ---
    text_block = TextBlockSchema(
        element_id=element_id,
        doc_id=doc_id,
        page_num=1,
        bbox=BoundingBox(left=10, top=20, right=300, bottom=50),
        reading_order=0,
        content="Hello, world.",
        metadata=ElementMetadata(confidence_score=0.98),
    )

    # --- Page ---
    page = PageSchema(
        page_num=1,
        size=Size(width=595, height=842),
        element_ids=[element_id],
    )

    # --- Section ---
    section = SectionSchema(
        section_id=uuid.uuid4(),
        section_path="1",
        title="Introduction",
        level=1,
        element_ids=[element_id],
    )

    # --- Chunk ---
    chunk = ChunkSchema(
        chunk_id=uuid.uuid4(),
        doc_id=doc_id,
        chunk_type="hierarchical",
        content="Hello, world.",
        element_refs=[element_id],
        page_range=(1, 1),
        metadata=ChunkMetadata(token_count=3),
    )

    # --- Relationship ---
    rel = RelationshipSchema(
        relationship_id=make_relationship_id(element_id, element_id, "contains"),
        source_id=element_id,
        target_id=element_id,
        relationship_type="contains",
        weight=1.0,
    )

    # --- DocumentMetadata ---
    doc_meta = DocumentMetadata(
        source_format="pdf",
        processing_status="extracted",
        confidence_score=0.95,
        processing_started_at=created,
    )

    # --- Document ---
    doc = DocumentSchema(
        doc_id=doc_id,
        title="Test Document",
        source_path="/path/to/test.pdf",
        file_hash="abc123deadbeef",
        page_count=1,
        created_at=created,
        pages=[page],
        elements={str(element_id): text_block},
        chunks=[chunk],
        relationships=[rel],
        metadata=doc_meta,
    )

    # Assertions
    assert doc.doc_id == doc_id
    assert doc.title == "Test Document"
    assert doc.source_path == "/path/to/test.pdf"
    assert doc.file_hash == "abc123deadbeef"
    assert doc.page_count == 1
    assert doc.created_at == created
    assert len(doc.pages) == 1
    assert doc.pages[0].page_num == 1
    assert doc.pages[0].size == Size(width=595, height=842)
    assert doc.pages[0].element_ids == [element_id]
    assert str(element_id) in doc.elements
    assert doc.elements[str(element_id)].content == "Hello, world."
    assert len(doc.chunks) == 1
    assert doc.chunks[0].chunk_type == "hierarchical"
    assert len(doc.relationships) == 1
    assert doc.relationships[0].relationship_type == "contains"
    assert doc.metadata.source_format == "pdf"
    assert doc.metadata.confidence_score == 0.95


# ---------------------------------------------------------------------------
# 3.  Deterministic ID generation
# ---------------------------------------------------------------------------


class TestDeterministicIds:
    """Verify that ID helpers produce stable, deterministic UUIDs."""

    def test_same_input_same_doc_id(self) -> None:
        from src.schemas import make_doc_id

        hash_a = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        id_1 = make_doc_id(hash_a)
        id_2 = make_doc_id(hash_a)
        assert id_1 == id_2, "Same file hash must produce the same doc ID"

    def test_different_input_different_doc_id(self) -> None:
        from src.schemas import make_doc_id

        id_1 = make_doc_id("aaaa")
        id_2 = make_doc_id("bbbb")
        assert id_1 != id_2, "Different file hashes must produce different doc IDs"

    def test_same_input_same_element_id(self) -> None:
        from src.schemas import make_element_id

        doc_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        id_1 = make_element_id(doc_id, 1, 0, "text_block")
        id_2 = make_element_id(doc_id, 1, 0, "text_block")
        assert id_1 == id_2

    def test_different_input_different_element_id(self) -> None:
        from src.schemas import make_element_id

        doc_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        id_1 = make_element_id(doc_id, 1, 0, "text_block")
        id_2 = make_element_id(doc_id, 1, 1, "text_block")
        assert id_1 != id_2

    def test_same_input_same_chunk_id(self) -> None:
        from src.schemas import make_chunk_id

        doc_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        id_1 = make_chunk_id(doc_id, "hierarchical", "1", 5)
        id_2 = make_chunk_id(doc_id, "hierarchical", "1", 5)
        assert id_1 == id_2

    def test_different_input_different_chunk_id(self) -> None:
        from src.schemas import make_chunk_id

        doc_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        id_1 = make_chunk_id(doc_id, "hierarchical", "1", 5)
        id_2 = make_chunk_id(doc_id, "semantic", "1", 5)
        assert id_1 != id_2

    def test_same_input_same_relationship_id(self) -> None:
        from src.schemas import make_relationship_id

        src = uuid.UUID("00000000-0000-0000-0000-000000000001")
        tgt = uuid.UUID("00000000-0000-0000-0000-000000000002")
        id_1 = make_relationship_id(src, tgt, "contains")
        id_2 = make_relationship_id(src, tgt, "contains")
        assert id_1 == id_2

    def test_different_input_different_relationship_id(self) -> None:
        from src.schemas import make_relationship_id

        src = uuid.UUID("00000000-0000-0000-0000-000000000001")
        tgt = uuid.UUID("00000000-0000-0000-0000-000000000002")
        id_1 = make_relationship_id(src, tgt, "contains")
        id_2 = make_relationship_id(src, tgt, "follows")
        assert id_1 != id_2

    def test_namespace_constants_are_uuid5(self) -> None:
        from src.schemas import (
            CHUNK_NAMESPACE,
            DOC_NAMESPACE,
            ELEM_NAMESPACE,
            REL_NAMESPACE,
        )

        for ns in (DOC_NAMESPACE, ELEM_NAMESPACE, CHUNK_NAMESPACE, REL_NAMESPACE):
            assert isinstance(ns, uuid.UUID)
            assert ns.version == 5


# ---------------------------------------------------------------------------
# 4.  Relationship type validation
# ---------------------------------------------------------------------------


class TestRelationshipTypes:
    """Ensure all approved relationship types are accepted and invalid
    values are rejected."""

    ALL_TYPES = [
        "contains",
        "belongs_to",
        "relates_to",
        "refers_to",
        "describes",
        "follows",
        "precedes",
        "summarizes",
        "supports",
        "explains",
        "has_caption",
        "has_table",
        "has_image",
        "has_formula",
        "same_section_as",
        "nearby",
    ]

    def test_all_relationship_types_accepted(self) -> None:
        """Every valid relationship type string should construct without error."""
        from src.schemas import RelationshipSchema

        src = uuid.uuid4()
        tgt = uuid.uuid4()

        for rel_type in self.ALL_TYPES:
            rel = RelationshipSchema(
                relationship_id=uuid.uuid4(),
                source_id=src,
                target_id=tgt,
                relationship_type=rel_type,  # type: ignore[arg-type]
            )
            assert rel.relationship_type == rel_type

    def test_invalid_relationship_type_rejected(self) -> None:
        """A relationship type not in the approved set should raise
        ``ValidationError``."""
        from src.schemas import RelationshipSchema

        with pytest.raises(ValidationError):
            RelationshipSchema(
                relationship_id=uuid.uuid4(),
                source_id=uuid.uuid4(),
                target_id=uuid.uuid4(),
                relationship_type="invalid_type",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 5.  Element subclass construction
# ---------------------------------------------------------------------------


class TestElementSubclasses:
    """Verify that each ``ElementSchema`` subclass can be constructed and
    carries the correct default ``element_type``."""

    @pytest.mark.parametrize(
        ("cls_name", "expected_type"),
        [
            ("TextBlockSchema", "text_block"),
            ("TableSchema", "table"),
            ("ImageSchema", "image"),
            ("ChartSchema", "chart"),
            ("GraphSchema", "graph"),
            ("FormulaSchema", "formula"),
            ("CaptionSchema", "caption"),
            ("FootnoteSchema", "footnote"),
            ("HeaderSchema", "header"),
            ("FooterSchema", "footer"),
            ("ListBlockSchema", "list_block"),
            ("SectionHeaderSchema", "section_header"),
        ],
    )
    def test_element_type_default(self, cls_name: str, expected_type: str) -> None:
        import src.schemas.elements as mod
        from src.schemas.geometry import BoundingBox

        cls = getattr(mod, cls_name)
        instance = cls(
            element_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            page_num=1,
            bbox=BoundingBox(left=0, top=0, right=100, bottom=50),
            reading_order=0,
        )
        assert instance.element_type == expected_type


# ---------------------------------------------------------------------------
# 6.  Geometry model construction
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_bounding_box_coord_system_default(self) -> None:
        from src.schemas import BoundingBox

        bb = BoundingBox(left=0, top=0, right=100, bottom=200)
        assert bb.coord_system == "pdf"

    def test_bounding_box_all_coord_systems(self) -> None:
        from src.schemas import BoundingBox

        for cs in ("pdf", "image", "normalized"):
            bb = BoundingBox(left=0, top=0, right=100, bottom=200, coord_system=cs)  # type: ignore[arg-type]
            assert bb.coord_system == cs

    def test_invalid_coord_system_rejected(self) -> None:
        from src.schemas import BoundingBox

        with pytest.raises(ValidationError):
            BoundingBox(left=0, top=0, right=100, bottom=200, coord_system="screen")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7.  ChunkSchema page_range
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_chunk_with_full_fields(self) -> None:
        from src.schemas import ChunkSchema

        chunk = ChunkSchema(
            chunk_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            chunk_type="semantic",
            content="Some chunk content",
            element_refs=[uuid.uuid4(), uuid.uuid4()],
            section_path="3.2.1",
            page_range=(3, 5),
        )
        assert chunk.chunk_type == "semantic"
        assert chunk.page_range == (3, 5)
        assert len(chunk.element_refs) == 2

    def test_chunk_defaults(self) -> None:
        from src.schemas import ChunkSchema

        chunk = ChunkSchema(
            chunk_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            chunk_type="cluster",
        )
        assert chunk.content == ""
        assert chunk.element_refs == []
        assert chunk.section_path == ""
        assert chunk.page_range is None
        assert chunk.relationships == []
