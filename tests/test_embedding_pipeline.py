"""Tests for the batch multimodal embedding pipeline."""

import numpy as np
import pytest

from src.chunking.models import ChunkMetadata, make_chunk_id
from src.retrieval.embedding_pipeline import (
    attach_embeddings,
    build_encode_items,
    encode_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC_HASH = 12345
DOC_NAME = "test.pdf"
DOC_TYPE = "application/pdf"


def _make_chunk(
    seq: int,
    *,
    embedding_type: str = "text",
    chunk_text: str = "",
    chunk_id: str | None = None,
    image_uri: list[str] | None = None,
    image_type: list[str] | None = None,
    caption_text: list[str] | None = None,
    caption_number: list[str] | None = None,
) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id or make_chunk_id(str(DOC_HASH), seq, [embedding_type]),
        document_name=DOC_NAME,
        document_type=DOC_TYPE,
        document_hash=DOC_HASH,
        embedding_type=embedding_type,  # type: ignore[arg-type]
        chunk_text=chunk_text,
        chunk_types=[embedding_type],
        sequence_number=seq,
        image_uri=image_uri or [],
        image_type=image_type or [],
        caption_text=caption_text or [],
        caption_number=caption_number or [],
    )


class _MockModel:
    """Returns fixed-dimension embeddings for testing dispatch logic."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.text_calls: list[list[str]] = []
        self.image_calls: list[str] = []

    def encode(
        self,
        sentences,
        *,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=False,
        **kwargs,
    ):
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            # Text batch
            self.text_calls.append(sentences)
            raw = np.array([[float(ord(s[0]) % 10)] * self.dim for s in sentences])
        elif isinstance(sentences, str):
            # Single image path
            self.image_calls.append(sentences)
            raw = np.array([float(hash(sentences) % 10)] * self.dim)
        else:
            raw = np.zeros((1, self.dim))
        # Return normalized
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return raw / norms


# ---------------------------------------------------------------------------
# build_encode_items tests
# ---------------------------------------------------------------------------


class TestBuildEncodeItems:
    def test_text_chunks(self):
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="Hello world."),
            _make_chunk(
                1, embedding_type="textual_description", chunk_text="A description."
            ),
        ]
        items = build_encode_items(chunks)
        assert len(items) == 2
        assert items[0]["embedding_type"] == "text"
        assert items[0]["content"] == "Hello world."
        assert items[0]["chunk_index"] == 0
        assert items[1]["embedding_type"] == "textual_description"
        assert items[1]["content"] == "A description."

    def test_image_chunk_with_valid_uri(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")  # create a file
        chunks = [
            _make_chunk(
                0,
                embedding_type="image",
                chunk_text="Caption",
                image_uri=[str(img)],
                image_type=["picture"],
            ),
        ]
        items = build_encode_items(chunks)
        assert len(items) == 1
        assert items[0]["embedding_type"] == "image"
        assert items[0]["content"] == str(img)

    def test_image_chunk_missing_uri(self):
        chunks = [
            _make_chunk(
                0,
                embedding_type="image",
                chunk_text="Caption",
                image_uri=[],
                image_type=["picture"],
            ),
        ]
        items = build_encode_items(chunks)
        assert len(items) == 0

    def test_image_chunk_nonexistent_file(self):
        chunks = [
            _make_chunk(
                0,
                embedding_type="image",
                chunk_text="Caption",
                image_uri=["/nonexistent/path.png"],
                image_type=["picture"],
            ),
        ]
        items = build_encode_items(chunks)
        assert len(items) == 0

    def test_mixed_chunks(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_text("fake")
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="Text content."),
            _make_chunk(
                1,
                embedding_type="image",
                chunk_text="Caption",
                image_uri=[str(img)],
                image_type=["picture"],
            ),
            _make_chunk(
                2, embedding_type="textual_description", chunk_text="LLM desc."
            ),
        ]
        items = build_encode_items(chunks)
        assert len(items) == 3
        assert items[0]["chunk_index"] == 0
        assert items[1]["chunk_index"] == 1
        assert items[2]["chunk_index"] == 2

    def test_empty_chunks(self):
        assert build_encode_items([]) == []


# ---------------------------------------------------------------------------
# encode_batch tests
# ---------------------------------------------------------------------------


class TestEncodeBatch:
    def test_text_only(self):
        model = _MockModel(dim=8)
        items = [
            {
                "chunk_id": "a",
                "embedding_type": "text",
                "content": "hello",
                "chunk_index": 0,
            },
            {
                "chunk_id": "b",
                "embedding_type": "text",
                "content": "world",
                "chunk_index": 1,
            },
        ]
        embeddings = encode_batch(items, model, batch_size=32)
        assert len(embeddings) == 2
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape == (8,) for e in embeddings)
        # Verify normalization
        for e in embeddings:
            assert abs(np.linalg.norm(e) - 1.0) < 1e-6 or np.allclose(e, 0)

    def test_image_only(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")
        model = _MockModel(dim=8)
        items = [
            {
                "chunk_id": "a",
                "embedding_type": "image",
                "content": str(img),
                "chunk_index": 0,
            },
        ]
        embeddings = encode_batch(items, model, batch_size=32)
        assert len(embeddings) == 1
        assert embeddings[0].shape == (8,)

    def test_mixed_modalities(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")
        model = _MockModel(dim=8)
        items = [
            {
                "chunk_id": "a",
                "embedding_type": "text",
                "content": "text",
                "chunk_index": 0,
            },
            {
                "chunk_id": "b",
                "embedding_type": "image",
                "content": str(img),
                "chunk_index": 1,
            },
            {
                "chunk_id": "c",
                "embedding_type": "textual_description",
                "content": "desc",
                "chunk_index": 2,
            },
        ]
        embeddings = encode_batch(items, model, batch_size=32)
        assert len(embeddings) == 3
        # Order preserved
        assert all(e.shape == (8,) for e in embeddings)

    def test_empty_items(self):
        model = _MockModel()
        assert encode_batch([], model) == []


# ---------------------------------------------------------------------------
# attach_embeddings tests
# ---------------------------------------------------------------------------


class TestAttachEmbeddings:
    def test_basic(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="Hello."),
            _make_chunk(
                1,
                embedding_type="image",
                chunk_text="Caption",
                image_uri=[str(img)],
                image_type=["picture"],
            ),
        ]
        items = build_encode_items(chunks)
        model = _MockModel(dim=4)
        embeddings = encode_batch(items, model)
        docs = attach_embeddings(chunks, embeddings)
        assert len(docs) == 2
        assert docs[0]["chunk_id"] == chunks[0].chunk_id
        assert docs[0]["embedding_type"] == "text"
        assert "embedding" in docs[0]
        assert len(docs[0]["embedding"]) == 4
        assert "metadata" in docs[0]
        assert docs[0]["metadata"]["page_numbers"] == []

    def test_mismatch_lengths(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="A"),
            _make_chunk(1, embedding_type="text", chunk_text="B"),
        ]
        items = build_encode_items(chunks)
        model = _MockModel(dim=4)
        embeddings = encode_batch(items, model)
        # Truncate embeddings to test mismatch handling
        docs = attach_embeddings(chunks, embeddings[:1])  # only 1 embedding, 2 items
        assert len(docs) == 1  # truncated to min

    def test_embedding_as_list_of_floats(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="Test."),
        ]
        items = build_encode_items(chunks)
        model = _MockModel(dim=4)
        embeddings = encode_batch(items, model)
        docs = attach_embeddings(chunks, embeddings)
        emb = docs[0]["embedding"]
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
        assert len(emb) == 4

    def test_json_export(self, tmp_path):
        """attach_embeddings writes a JSON file when output_dir is provided."""
        img = tmp_path / "test.png"
        img.write_text("fake")
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="A"),
            _make_chunk(1, embedding_type="text", chunk_text="B"),
            _make_chunk(
                2,
                embedding_type="image",
                chunk_text="Cap",
                image_uri=[str(img)],
                image_type=["picture"],
            ),
        ]
        items = build_encode_items(chunks)
        model = _MockModel(dim=4)
        embeddings = encode_batch(items, model)

        out = tmp_path / "exports"
        docs = attach_embeddings(
            chunks,
            embeddings,
            document_name="mydoc.pdf",
            output_dir=out,
        )
        export_path = out / "mydoc_embeddings.json"
        assert export_path.exists()

        import json

        with export_path.open() as f:
            data = json.load(f)
        assert len(data) == 3
        assert all("embedding" in d for d in data)

    def test_no_export_when_output_dir_none(self, tmp_path):
        chunks = [_make_chunk(0, embedding_type="text", chunk_text="X")]
        items = build_encode_items(chunks)
        model = _MockModel(dim=4)
        embeddings = encode_batch(items, model)
        docs = attach_embeddings(chunks, embeddings)
        assert len(docs) == 1  # just returns, doesn't crash


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_end_to_end(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake")
        chunks = [
            _make_chunk(0, embedding_type="text", chunk_text="First chunk."),
            _make_chunk(
                1,
                embedding_type="image",
                chunk_text="Image caption",
                image_uri=[str(img)],
                image_type=["picture"],
            ),
            _make_chunk(
                2,
                embedding_type="textual_description",
                chunk_text="Detailed description.",
            ),
            _make_chunk(3, embedding_type="text", chunk_text="Another text chunk."),
        ]
        model = _MockModel(dim=16)

        # Step 1
        items = build_encode_items(chunks)
        assert len(items) == 4

        # Step 2
        embeddings = encode_batch(items, model, batch_size=2)
        assert len(embeddings) == 4

        # Step 3
        docs = attach_embeddings(chunks, embeddings)
        assert len(docs) == 4
        for doc in docs:
            assert "chunk_id" in doc
            assert "embedding" in doc
            assert len(doc["embedding"]) == 16
            assert isinstance(doc["metadata"]["token_count"], int)
