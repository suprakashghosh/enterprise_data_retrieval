"""Tests for cross-reference resolution."""

from unittest.mock import Mock

from src.chunking.models import ChunkMetadata, make_chunk_id
from src.chunking.cross_reference_resolver import (
    _build_caption_index,
    resolve_cross_references,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC_HASH = 12345
DOC_NAME = "test_doc.pdf"
DOC_TYPE = "application/pdf"


def _make_chunk(
    seq: int,
    chunk_text: str,
    *,
    element_self_refs: list[str] | None = None,
    chunk_id: str | None = None,
    image_type: list[str] | None = None,
    caption_text: list[str] | None = None,
    caption_number: list[str] | None = None,
) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id or make_chunk_id(str(DOC_HASH), seq, ["text"]),
        document_name=DOC_NAME,
        document_type=DOC_TYPE,
        document_hash=DOC_HASH,
        embedding_type="text",
        chunk_text=chunk_text,
        chunk_types=["text"],
        sequence_number=seq,
        element_self_refs=element_self_refs or [f"#/texts/{seq}"],
        image_type=image_type or [],
        caption_text=caption_text or [],
        caption_number=caption_number or [],
    )


# ---------------------------------------------------------------------------
# _build_caption_index tests
# ---------------------------------------------------------------------------


class TestBuildCaptionIndex:
    def test_single_picture(self):
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {
            "#/texts/99": "Figure 1: Architecture overview.",
        }
        chunks = [
            ChunkMetadata(
                chunk_id="chunk-A",
                document_name=DOC_NAME,
                document_type=DOC_TYPE,
                document_hash=DOC_HASH,
                embedding_type="text",
                chunk_text="...",
                sequence_number=0,
                element_self_refs=["#/pictures/0"],
            ),
        ]
        index = _build_caption_index(pic_lookup, text_lookup, chunks)
        assert index == {"figure 1": "chunk-A"}

    def test_multiple_visuals_in_same_chunk(self):
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
            "#/tables/0": Mock(
                captions=[Mock(cref="#/texts/100")],
                children=None,
            ),
        }
        text_lookup = {
            "#/texts/99": "Figure 1: Overview.",
            "#/texts/100": "Table IV: Results.",
        }
        chunks = [
            ChunkMetadata(
                chunk_id="chunk-B",
                document_name=DOC_NAME,
                document_type=DOC_TYPE,
                document_hash=DOC_HASH,
                embedding_type="text",
                chunk_text="...",
                sequence_number=0,
                element_self_refs=["#/pictures/0", "#/tables/0"],
            ),
        ]
        index = _build_caption_index(pic_lookup, text_lookup, chunks)
        assert index == {"figure 1": "chunk-B", "table IV": "chunk-B"}

    def test_missing_caption_skipped(self):
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=None,
                children=None,
            ),
        }
        text_lookup = {}
        chunks = [
            ChunkMetadata(
                chunk_id="chunk-C",
                document_name=DOC_NAME,
                document_type=DOC_TYPE,
                document_hash=DOC_HASH,
                embedding_type="text",
                chunk_text="...",
                sequence_number=0,
                element_self_refs=["#/pictures/0"],
            ),
        ]
        index = _build_caption_index(pic_lookup, text_lookup, chunks)
        assert index == {}

    def test_label_not_in_caption_extractor_skipped(self):
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {
            "#/texts/99": "This is not a caption.",
        }
        chunks = [
            ChunkMetadata(
                chunk_id="chunk-D",
                document_name=DOC_NAME,
                document_type=DOC_TYPE,
                document_hash=DOC_HASH,
                embedding_type="text",
                chunk_text="...",
                sequence_number=0,
                element_self_refs=["#/pictures/0"],
            ),
        ]
        index = _build_caption_index(pic_lookup, text_lookup, chunks)
        assert index == {}

    def test_pic_not_in_any_chunk_skipped(self):
        pic_lookup = {
            "#/pictures/99": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {
            "#/texts/99": "Figure 42: Missing chunk.",
        }
        chunks = [
            ChunkMetadata(
                chunk_id="chunk-E",
                document_name=DOC_NAME,
                document_type=DOC_TYPE,
                document_hash=DOC_HASH,
                embedding_type="text",
                chunk_text="...",
                sequence_number=0,
                element_self_refs=["#/pictures/0"],
            ),
        ]
        index = _build_caption_index(pic_lookup, text_lookup, chunks)
        assert index == {}


# ---------------------------------------------------------------------------
# resolve_cross_references tests
# ---------------------------------------------------------------------------


class TestResolveCrossReferences:
    def test_simple_reference(self):
        chunks = [
            _make_chunk(
                0, "Please see Figure 1 for details.", element_self_refs=["#/texts/0"]
            ),
            _make_chunk(
                1, "Some description here.", element_self_refs=["#/pictures/0"]
            ),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {"#/texts/99": "Figure 1: Overview."}
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert result[0].refers_to == [chunks[1].chunk_id]
        assert result[1].refers_to == []  # visual chunk doesn't reference itself

    def test_abbreviation_matching(self):
        """Fig. 3 in text should match Figure 3 in caption."""
        chunks = [
            _make_chunk(
                0, "As shown in Fig. 3, the results...", element_self_refs=["#/texts/0"]
            ),
            _make_chunk(1, "Figure 3", element_self_refs=["#/pictures/0"]),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {"#/texts/99": "Figure 3: Results."}
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert result[0].refers_to == [chunks[1].chunk_id]

    def test_multiple_references(self):
        chunks = [
            _make_chunk(
                0,
                "See Figure 1 and Table II for reference.",
                element_self_refs=["#/texts/0"],
            ),
            _make_chunk(1, "Fig 1", element_self_refs=["#/pictures/0"]),
            _make_chunk(2, "Table II", element_self_refs=["#/tables/0"]),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
            "#/tables/0": Mock(
                captions=[Mock(cref="#/texts/100")],
                children=None,
            ),
        }
        text_lookup = {
            "#/texts/99": "Figure 1: Architecture.",
            "#/texts/100": "Table II: Measurements.",
        }
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert sorted(result[0].refers_to) == sorted(
            [chunks[1].chunk_id, chunks[2].chunk_id]
        )

    def test_self_reference_filtered(self):
        chunks = [
            _make_chunk(
                0,
                "The caption of this figure is Figure 1.",
                element_self_refs=["#/texts/0", "#/pictures/0"],
            ),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {"#/texts/99": "Figure 1: Self-referencing."}
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert result[0].refers_to == []  # self-reference filtered

    def test_no_references(self):
        chunks = [
            _make_chunk(
                0,
                "This paragraph has no references to figures.",
                element_self_refs=["#/texts/0"],
            ),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {"#/texts/99": "Figure 1: Something."}
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert result[0].refers_to == []

    def test_reference_to_unknown_caption(self):
        chunks = [
            _make_chunk(
                0, "Please see Figure 99 for details.", element_self_refs=["#/texts/0"]
            ),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {"#/texts/99": "Figure 1: Known thing."}
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert result[0].refers_to == []  # "Figure 99" not in index

    def test_empty_pic_table_lookup(self):
        chunks = [_make_chunk(0, "See Figure 1.", element_self_refs=["#/texts/0"])]
        result = resolve_cross_references(chunks, {}, {})
        assert result[0].refers_to == []

    def test_original_not_mutated(self):
        chunks = [_make_chunk(0, "See Figure 1.", element_self_refs=["#/texts/0"])]
        original = chunks[0].model_copy(deep=True)
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
        }
        text_lookup = {"#/texts/99": "Figure 1: Overview."}
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        # Original unchanged
        assert chunks[0].refers_to == []
        assert chunks[0] == original

    def test_chunk_with_multiple_visual_matches(self):
        """When multiple visual items in one chunk, all indexed."""
        chunks = [
            _make_chunk(0, "See Figure 2.", element_self_refs=["#/texts/0"]),
            _make_chunk(
                1,
                "multi visual",
                element_self_refs=["#/pictures/0", "#/pictures/1"],
            ),
        ]
        pic_lookup = {
            "#/pictures/0": Mock(
                captions=[Mock(cref="#/texts/99")],
                children=None,
            ),
            "#/pictures/1": Mock(
                captions=[Mock(cref="#/texts/100")],
                children=None,
            ),
        }
        text_lookup = {
            "#/texts/99": "Figure 1: First.",
            "#/texts/100": "Figure 2: Second.",
        }
        result = resolve_cross_references(chunks, pic_lookup, text_lookup)
        assert result[0].refers_to == [chunks[1].chunk_id]
