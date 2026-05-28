"""Tests for top-k cosine similarity relates_to population."""

import numpy as np
import pytest

from src.chunking.models import ChunkMetadata, make_chunk_id
from src.retrieval.similarity import (
    compute_cosine_similarity_matrix,
    populate_relates_to,
    _build_sibling_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC_HASH = 42
DOC_NAME = "test.pdf"
DOC_TYPE = "application/pdf"


def _make_chunk(
    seq: int,
    *,
    chunk_text: str = "",
    element_self_refs: list[str] | None = None,
    chunk_id: str | None = None,
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
        element_self_refs=element_self_refs or [],
    )


# ---------------------------------------------------------------------------
# compute_cosine_similarity_matrix tests
# ---------------------------------------------------------------------------


class TestCosineSimilarityMatrix:
    def test_identity(self):
        """Identical vectors -> sim == 1.0."""
        emb = np.array([[1.0, 0.0], [1.0, 0.0]])
        sim = compute_cosine_similarity_matrix(emb)
        assert sim.shape == (2, 2)
        np.testing.assert_allclose(np.diag(sim), [1.0, 1.0])
        np.testing.assert_allclose(sim[0, 1], 1.0)

    def test_orthogonal(self):
        """Orthogonal vectors -> sim == 0.0."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_cosine_similarity_matrix(emb)
        np.testing.assert_allclose(sim[0, 1], 0.0, atol=1e-10)

    def test_opposite(self):
        """Opposite vectors -> sim == -1.0."""
        emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
        sim = compute_cosine_similarity_matrix(emb)
        np.testing.assert_allclose(sim[0, 1], -1.0)

    def test_auto_normalize(self):
        """Unnormalized vectors are auto-normalized."""
        emb = np.array([[3.0, 4.0], [6.0, 8.0]])  # both point in same direction
        sim = compute_cosine_similarity_matrix(emb)
        np.testing.assert_allclose(sim[0, 1], 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_cosine_similarity_matrix(np.array([]))

    def test_single_vector(self):
        emb = np.array([[1.0, 2.0, 3.0]])
        sim = compute_cosine_similarity_matrix(emb)
        assert sim.shape == (1, 1)
        np.testing.assert_allclose(sim[0, 0], 1.0)


# ---------------------------------------------------------------------------
# _build_sibling_mask tests
# ---------------------------------------------------------------------------


class TestSiblingMask:
    def test_no_siblings(self):
        chunks = [
            _make_chunk(0, element_self_refs=["#/texts/0"]),
            _make_chunk(1, element_self_refs=["#/texts/1"]),
        ]
        mask = _build_sibling_mask(chunks)
        assert mask.shape == (2, 2)
        assert not mask.any()  # no siblings

    def test_shared_ref_are_siblings(self):
        chunks = [
            _make_chunk(0, element_self_refs=["#/pictures/0"]),
            _make_chunk(1, element_self_refs=["#/pictures/0"]),  # same ref
        ]
        mask = _build_sibling_mask(chunks)
        assert mask[0, 1]
        assert mask[1, 0]

    def test_mixed_refs(self):
        chunks = [
            _make_chunk(0, element_self_refs=["#/texts/0", "#/pictures/0"]),
            _make_chunk(1, element_self_refs=["#/pictures/0"]),  # shares #/pictures/0
            _make_chunk(2, element_self_refs=["#/texts/1"]),  # unrelated
        ]
        mask = _build_sibling_mask(chunks)
        assert mask[0, 1]  # share ref
        assert not mask[0, 2]  # no shared ref
        assert not mask[1, 2]

    def test_empty_refs(self):
        chunks = [
            _make_chunk(0, element_self_refs=[]),
            _make_chunk(1, element_self_refs=[]),
        ]
        mask = _build_sibling_mask(chunks)
        assert not mask.any()


# ---------------------------------------------------------------------------
# populate_relates_to tests
# ---------------------------------------------------------------------------


class TestPopulateRelatesTo:
    def test_empty_chunks(self):
        result = populate_relates_to([], np.array([]).reshape(0, 2))
        assert result == []

    def test_single_chunk(self):
        chunks = [_make_chunk(0)]
        emb = np.array([[1.0, 0.0]])
        result = populate_relates_to(chunks, emb)
        assert result[0].relates_to == []

    def test_top_k_neighbors(self):
        """Chunk 0 and 2 are very similar, chunk 1 is far away."""
        chunks = [
            _make_chunk(0, chunk_id="A"),
            _make_chunk(1, chunk_id="B"),
            _make_chunk(2, chunk_id="C"),
            _make_chunk(3, chunk_id="D"),
        ]
        # A=[1,0,0], B=[0,1,0], C=[0.99,0.01,0], D=[0,0,1]
        emb = np.array(
            [
                [1.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0],  # B
                [0.99, 0.01, 0.0],  # C (very close to A)
                [0.0, 0.0, 1.0],  # D
            ],
            dtype=np.float64,
        )
        result = populate_relates_to(chunks, emb, top_k=2, min_similarity=0.7)
        # A should relate to C (closest)
        assert "C" in result[0].relates_to
        # C should relate to A
        assert "A" in result[2].relates_to
        # B and D are far from everyone — no relates_to
        assert result[1].relates_to == []
        assert result[3].relates_to == []

    def test_excludes_self(self):
        chunks = [_make_chunk(0, chunk_id="X"), _make_chunk(1, chunk_id="Y")]
        emb = np.array([[1.0, 0.0], [0.999, 0.001]])
        result = populate_relates_to(chunks, emb, top_k=3)
        # Each chunk should NOT include itself
        assert "X" not in result[0].relates_to
        assert "Y" not in result[1].relates_to

    def test_sibling_exclusion(self):
        """Chunks sharing element_self_refs should not relate to each other."""
        chunks = [
            _make_chunk(0, chunk_id="img", element_self_refs=["#/pictures/0"]),
            _make_chunk(1, chunk_id="desc", element_self_refs=["#/pictures/0"]),
        ]
        # Very similar embeddings (sim ≈ 1.0)
        emb = np.array([[1.0, 0.0], [0.999, 0.001]])
        result = populate_relates_to(chunks, emb, top_k=3, min_similarity=0.7)
        # Siblings excluded — no relates_to between them
        assert result[0].relates_to == []
        assert result[1].relates_to == []

    def test_min_similarity_threshold(self):
        chunks = [
            _make_chunk(0, chunk_id="A"),
            _make_chunk(1, chunk_id="B"),
        ]
        # Similarity ≈ 0.5 (45 degrees)
        emb = np.array([[1.0, 0.0], [0.5, 0.8660254]])  # cos ≈ 0.5
        result = populate_relates_to(chunks, emb, top_k=3, min_similarity=0.75)
        # Below threshold — no relates_to
        assert result[0].relates_to == []
        assert result[1].relates_to == []

    def test_deterministic(self):
        chunks = [
            _make_chunk(0, chunk_id="A"),
            _make_chunk(1, chunk_id="B"),
            _make_chunk(2, chunk_id="C"),
        ]
        emb = np.array([[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]])
        r1 = populate_relates_to(chunks, emb)
        r2 = populate_relates_to(chunks, emb)
        for a, b in zip(r1, r2):
            assert a.relates_to == b.relates_to

    def test_original_not_mutated(self):
        chunks = [
            _make_chunk(0, chunk_id="A"),
            _make_chunk(1, chunk_id="B"),
        ]
        original = [cm.model_copy(deep=True) for cm in chunks]
        emb = np.array([[1.0, 0.0], [0.9, 0.1]])
        result = populate_relates_to(chunks, emb)
        # Originals unchanged
        for orig, cm in zip(original, chunks):
            assert orig.relates_to == cm.relates_to
            assert orig == cm  # full equality

    def test_shape_mismatch_raises(self):
        chunks = [_make_chunk(0), _make_chunk(1)]
        emb = np.array([[1.0, 0.0]])  # only 1 row, 2 chunks
        with pytest.raises(ValueError, match="does not match"):
            populate_relates_to(chunks, emb)

    def test_sorted_descending(self):
        """relates_to should be sorted by descending similarity."""
        chunks = [
            _make_chunk(0, chunk_id="A"),
            _make_chunk(1, chunk_id="B"),
            _make_chunk(2, chunk_id="C"),
            _make_chunk(3, chunk_id="D"),
        ]
        # A → closest to C, then B, then D
        emb = np.array(
            [
                [1.0, 0.0],  # A
                [0.7, 0.3],  # B (sim with A ≈ 0.7)
                [0.9, 0.1],  # C (sim with A ≈ 0.9)
                [0.5, 0.5],  # D (sim with A ≈ 0.5)
            ]
        )
        result = populate_relates_to(chunks, emb, top_k=3, min_similarity=0.4)
        # A's relates_to should be [C, B, D] in that order
        assert result[0].relates_to == ["C", "B", "D"]

    def test_json_export(self, tmp_path):
        """populate_relates_to writes JSON when output_dir is provided."""
        chunks = [
            _make_chunk(0, chunk_id="A"),
            _make_chunk(1, chunk_id="B"),
            _make_chunk(2, chunk_id="C"),
        ]
        emb = np.array([[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]])
        out = tmp_path / "exports"
        result = populate_relates_to(
            chunks,
            emb,
            document_name="mydoc.pdf",
            output_dir=out,
        )
        export_path = out / "mydoc_chunks_metadata_with_relates_to.json"
        assert export_path.exists()

        import json

        with export_path.open() as f:
            data = json.load(f)
        assert len(data) == 3
        # Verify relates_to is in the exported data
        assert "relates_to" in data[0]

    def test_no_export_when_output_dir_none(self):
        chunks = [_make_chunk(0, chunk_id="A"), _make_chunk(1, chunk_id="B")]
        emb = np.array([[1.0, 0.0], [0.5, 0.5]])
        result = populate_relates_to(chunks, emb)
        assert len(result) == 2  # just returns, doesn't crash
