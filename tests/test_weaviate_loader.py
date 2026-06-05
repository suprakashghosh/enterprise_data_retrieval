"""Comprehensive unit tests for ``src.loading.weaviate_loader``.

Covers ``create_schema``, ``ingest_chunks``, and ``run_weaviate_ingestion``
with mocked Weaviate client and patched connection.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import weaviate.classes.config as wc

from src.loading.weaviate_loader import (
    _PROPERTY_DEFS,
    create_schema,
    ingest_chunks,
    run_weaviate_ingestion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks_with_embeddings() -> List[Dict[str, Any]]:
    """Return two representative document-chunk dictionaries (as produced by
    ``attach_embeddings``) with UUID strings, embedding vectors, and
    full metadata."""
    return [
        {
            "chunk_id": "11111111-1111-1111-1111-111111111111",
            "embedding": [0.1, 0.2, 0.3],
            "embedding_type": "text",
            "chunk_text": "Sample chunk text",
            "metadata": {
                "document_name": "test.pdf",
                "document_hash": 12345678,
                "document_type": "pdf",
                "chunk_types": ["text"],
                "section_path": "Section 1 > Subsection A",
                "section_headings": ["Section 1", "Subsection A"],
                "page_numbers": [1],
                "sequence_number": 0,
                "image_type": [],
                "image_uri": [],
                "caption_text": [],
                "caption_number": [],
                "element_self_refs": [],
                "token_count": 50,
                "chunk_text": "Sample chunk text",
                "embedding_type": "text",
                "refers_to": [],
                "relates_to": [],
            },
        },
        {
            "chunk_id": "22222222-2222-2222-2222-222222222222",
            "embedding": [0.4, 0.5, 0.6],
            "embedding_type": "image",
            "chunk_text": "Figure 1: Architecture diagram",
            "metadata": {
                "document_name": "test.pdf",
                "document_hash": 12345678,
                "document_type": "pdf",
                "chunk_types": ["picture"],
                "section_path": "Section 2",
                "section_headings": ["Section 2"],
                "page_numbers": [3],
                "sequence_number": 1,
                "image_type": ["picture"],
                "image_uri": ["/path/to/image.png"],
                "caption_text": ["Figure 1: Architecture diagram"],
                "caption_number": ["figure 1"],
                "element_self_refs": ["ref_pic_1"],
                "token_count": 25,
                "chunk_text": "Figure 1: Architecture diagram",
                "embedding_type": "image",
                "refers_to": [],
                "relates_to": [],
            },
        },
    ]


@pytest.fixture
def mock_weaviate_client() -> MagicMock:
    """Return a ``MagicMock`` configured as a ``weaviate.WeaviateClient``.

    The mock includes:
    - ``collections.exists`` — returns ``False`` by default
    - ``collections.delete`` — no-op
    - ``collections.create`` — returns a mock collection object
    - ``collections.get`` — returns a mock collection object whose
      ``data.insert_many`` returns a ``BatchObjectReturn``-like mock with
      ``has_errors = False`` and empty ``errors`` dict.
    """
    client = MagicMock()

    # --- collections methods ---
    client.collections.exists.return_value = False
    client.collections.delete.return_value = None

    mock_collection = MagicMock()
    mock_insert_result = MagicMock()
    mock_insert_result.has_errors = False
    mock_insert_result.errors = {}
    mock_collection.data.insert_many.return_value = mock_insert_result

    client.collections.create.return_value = mock_collection
    client.collections.get.return_value = mock_collection

    return client


@pytest.fixture
def many_chunks() -> List[Dict[str, Any]]:
    """Generate 250 chunk dicts for batching tests."""
    chunks: List[Dict[str, Any]] = []
    for i in range(250):
        chunks.append(
            {
                "chunk_id": f"{i:08d}-0000-0000-0000-000000000000",
                "embedding": [float(i) * 0.01, float(i) * 0.02, float(i) * 0.03],
                "embedding_type": "text",
                "chunk_text": f"Chunk {i}",
                "metadata": {
                    "document_name": "batch_test.pdf",
                    "document_hash": 99999999,
                    "document_type": "pdf",
                    "chunk_types": ["text"],
                    "section_path": "Section",
                    "section_headings": ["Section"],
                    "page_numbers": [1],
                    "sequence_number": i,
                    "image_type": [],
                    "image_uri": [],
                    "caption_text": [],
                    "caption_number": [],
                    "element_self_refs": [],
                    "token_count": 10,
                    "chunk_text": f"Chunk {i}",
                    "embedding_type": "text",
                    "refers_to": [],
                    "relates_to": [],
                },
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# create_schema tests
# ---------------------------------------------------------------------------


class TestCreateSchema:
    """Tests for ``create_schema`` — collection creation, reuse, and drop."""

    def test_create_schema_creates_new_collection(
        self, mock_weaviate_client: MagicMock
    ) -> None:
        """When the collection does *not* exist, ``create_schema`` should
        call ``client.collections.create`` with the expected arguments and
        *not* call ``delete``."""
        client = mock_weaviate_client
        client.collections.exists.return_value = False

        result = create_schema(client)

        # Should NOT have tried to delete
        client.collections.delete.assert_not_called()

        # Should have created the collection with the expected parameters
        client.collections.create.assert_called_once()
        call_kwargs = client.collections.create.call_args.kwargs
        assert call_kwargs["name"] == "DocumentChunks"
        assert call_kwargs["vectorizer_config"] == wc.Configure.Vectorizer.none()
        assert call_kwargs["vector_index_config"] == wc.Configure.VectorIndex.hnsw(
            distance_metric=wc.VectorDistances.COSINE,
        )
        # Verify 18 properties were passed
        properties = call_kwargs["properties"]
        assert len(properties) == 18

        # Verify each property is a wc.Property with correct name/dataType
        for prop_def, prop in zip(_PROPERTY_DEFS, properties):
            assert prop.name == prop_def["name"]
            assert prop.dataType == prop_def["data_type"]

        # Should return the created collection
        assert result is not None

    def test_create_schema_reuses_existing_collection(
        self, mock_weaviate_client: MagicMock
    ) -> None:
        """When the collection already exists and ``drop_if_exists=False``
        (the default), ``create_schema`` should call ``collections.get`` and
        *not* call ``create``."""
        client = mock_weaviate_client
        client.collections.exists.return_value = True

        result = create_schema(client)

        client.collections.exists.assert_called_with("DocumentChunks")
        client.collections.get.assert_called_once_with("DocumentChunks")
        client.collections.create.assert_not_called()
        client.collections.delete.assert_not_called()
        assert result is not None

    def test_create_schema_drops_and_recreates(
        self, mock_weaviate_client: MagicMock
    ) -> None:
        """When the collection exists and ``drop_if_exists=True``,
        ``create_schema`` should first delete the existing collection and
        then create a fresh one.

        Because ``create_schema`` checks ``exists`` again *after* the
        delete, we use a ``side_effect`` so the first check returns
        ``True`` (triggering the delete) and the second returns ``False``
        (so it proceeds to create).
        """
        client = mock_weaviate_client
        client.collections.exists.side_effect = [True, False]

        result = create_schema(client, drop_if_exists=True)

        client.collections.delete.assert_called_once_with("DocumentChunks")
        client.collections.create.assert_called_once()
        assert result is not None

    def test_create_schema_custom_collection_name(
        self, mock_weaviate_client: MagicMock
    ) -> None:
        """The ``collection_name`` parameter should be forwarded to
        all Weaviate API calls."""
        client = mock_weaviate_client
        client.collections.exists.return_value = False

        create_schema(client, collection_name="CustomCollection")

        call_kwargs = client.collections.create.call_args.kwargs
        assert call_kwargs["name"] == "CustomCollection"


# ---------------------------------------------------------------------------
# ingest_chunks tests
# ---------------------------------------------------------------------------


class TestIngestChunks:
    """Tests for ``ingest_chunks`` — successful insert, partial errors,
    empty input, batching, and single-chunk edge cases."""

    def test_ingest_chunks_successful(
        self,
        mock_weaviate_client: MagicMock,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """All chunks inserted without errors — the summary should report
        ``total=2``, ``successful=2``, ``failed=0``, ``errors=[]``.
        Each chunk should be wrapped in a ``DataObject`` with the correct
        UUID, vector, and properties.

        Because ``DataObject`` is a real Weaviate dataclass (not mocked),
        we can inspect the objects passed to ``insert_many`` for their
        actual attribute values.
        """
        client = mock_weaviate_client
        mock_collection = client.collections.get.return_value
        mock_result = mock_collection.data.insert_many.return_value
        mock_result.has_errors = False
        mock_result.errors = {}

        summary = ingest_chunks(client, sample_chunks_with_embeddings)

        assert summary["total"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["errors"] == []

        # Verify the objects passed to insert_many are real DataObject
        # instances with correct values
        mock_collection.data.insert_many.assert_called_once()
        insert_args = mock_collection.data.insert_many.call_args[0][0]
        assert len(insert_args) == 2

        obj0 = insert_args[0]
        assert obj0.uuid == "11111111-1111-1111-1111-111111111111"
        assert obj0.vector == [0.1, 0.2, 0.3]
        assert obj0.properties["document_name"] == "test.pdf"
        assert obj0.properties["document_hash"] == 12345678
        assert obj0.properties["chunk_types"] == ["text"]
        assert obj0.properties["token_count"] == 50

        obj1 = insert_args[1]
        assert obj1.uuid == "22222222-2222-2222-2222-222222222222"
        assert obj1.vector == [0.4, 0.5, 0.6]
        assert obj1.properties["document_name"] == "test.pdf"
        assert obj1.properties["embedding_type"] == "image"

    def test_ingest_chunks_with_errors(
        self,
        mock_weaviate_client: MagicMock,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """When some objects fail, the summary should reflect partial
        success and include the error messages."""
        client = mock_weaviate_client
        mock_collection = client.collections.get.return_value
        mock_result = mock_collection.data.insert_many.return_value
        mock_result.has_errors = True
        # Simulate the first object (index 0) failed
        error_obj = MagicMock()
        error_obj.message = "insert failed"
        mock_result.errors = {0: error_obj}

        summary = ingest_chunks(client, sample_chunks_with_embeddings)

        assert summary["total"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["errors"]) == 1
        assert "insert failed" in summary["errors"][0]

    def test_ingest_chunks_empty_list(self, mock_weaviate_client: MagicMock) -> None:
        """An empty chunks list should return a zeroed summary and
        never call ``insert_many``."""
        client = mock_weaviate_client
        mock_collection = client.collections.get.return_value

        summary = ingest_chunks(client, [])

        assert summary == {"total": 0, "successful": 0, "failed": 0, "errors": []}
        mock_collection.data.insert_many.assert_not_called()

    def test_ingest_chunks_batching(
        self,
        mock_weaviate_client: MagicMock,
        many_chunks: List[Dict[str, Any]],
    ) -> None:
        """With 250 chunks and ``batch_size=100``, ``insert_many`` should
        be called exactly 3 times (batches of 100, 100, 50)."""
        client = mock_weaviate_client
        mock_collection = client.collections.get.return_value

        summary = ingest_chunks(client, many_chunks, batch_size=100)

        assert summary["total"] == 250
        assert summary["successful"] == 250
        assert mock_collection.data.insert_many.call_count == 3

        # Verify batch sizes
        calls = mock_collection.data.insert_many.call_args_list
        assert len(calls[0][0][0]) == 100  # first batch: 100
        assert len(calls[1][0][0]) == 100  # second batch: 100
        assert len(calls[2][0][0]) == 50  # third batch: 50

    def test_ingest_chunks_single_chunk(
        self,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """A single chunk should result in ``total=1``, ``successful=1``,
        and exactly one ``insert_many`` call."""
        client = mock_weaviate_client
        mock_collection = client.collections.get.return_value

        chunks = [
            {
                "chunk_id": "33333333-3333-3333-3333-333333333333",
                "embedding": [0.7, 0.8, 0.9],
                "embedding_type": "text",
                "chunk_text": "Single chunk",
                "metadata": {
                    "document_name": "single.pdf",
                    "document_hash": 11111111,
                    "document_type": "txt",
                    "chunk_types": ["text"],
                    "section_path": "",
                    "section_headings": [],
                    "page_numbers": [],
                    "sequence_number": 0,
                    "image_type": [],
                    "image_uri": [],
                    "caption_text": [],
                    "caption_number": [],
                    "element_self_refs": [],
                    "token_count": 5,
                    "chunk_text": "Single chunk",
                    "embedding_type": "text",
                    "refers_to": [],
                    "relates_to": [],
                },
            }
        ]

        summary = ingest_chunks(client, chunks)

        assert summary["total"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0
        assert summary["errors"] == []
        mock_collection.data.insert_many.assert_called_once()

    def test_ingest_chunks_custom_collection(
        self,
        mock_weaviate_client: MagicMock,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """The ``collection_name`` parameter should be forwarded to
        ``client.collections.get``."""
        client = mock_weaviate_client

        ingest_chunks(client, sample_chunks_with_embeddings, collection_name="MyColl")

        client.collections.get.assert_called_once_with("MyColl")


# ---------------------------------------------------------------------------
# run_weaviate_ingestion tests
# ---------------------------------------------------------------------------


class TestRunWeaviateIngestion:
    """Tests for the convenience wrapper ``run_weaviate_ingestion`` —
    client lifecycle, error handling, and forwarding of flags."""

    def test_run_weaviate_ingestion_with_provided_client(
        self,
        mock_weaviate_client: MagicMock,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """When an existing client is passed in, the function should use it
        directly, *not* create a new connection, and *not* close it."""
        client = mock_weaviate_client

        with patch("src.loading.weaviate_loader.wv.connect_to_local") as mock_connect:
            summary = run_weaviate_ingestion(
                sample_chunks_with_embeddings, client=client
            )

        # Should NOT have created its own connection
        mock_connect.assert_not_called()

        # Should have used the provided client's collections
        client.collections.exists.assert_called()
        client.collections.create.assert_called()

        # Should NOT have closed the client
        client.close.assert_not_called()

        assert summary["total"] == 2
        assert summary["successful"] == 2

    def test_run_weaviate_ingestion_creates_own_client(
        self,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """When *no* client is provided, the function should create one via
        ``wv.connect_to_local``, use it, and then close it."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_insert_result = MagicMock()
        mock_insert_result.has_errors = False
        mock_insert_result.errors = {}
        mock_collection.data.insert_many.return_value = mock_insert_result
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        with patch(
            "src.loading.weaviate_loader.wv.connect_to_local",
            return_value=mock_client,
        ) as mock_connect:
            summary = run_weaviate_ingestion(sample_chunks_with_embeddings)

        # Should have called connect_to_local with default params
        mock_connect.assert_called_once_with(
            host="localhost", port=8080, grpc_port=50051
        )

        # Should have closed the client after use
        mock_client.close.assert_called_once()

        assert summary["total"] == 2
        assert summary["successful"] == 2

    def test_run_weaviate_ingestion_custom_host_port(
        self,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """Custom ``host``, ``port``, and ``grpc_port`` should be forwarded
        to ``connect_to_local``."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.has_errors = False
        mock_result.errors = {}
        mock_collection.data.insert_many.return_value = mock_result
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        with patch(
            "src.loading.weaviate_loader.wv.connect_to_local",
            return_value=mock_client,
        ) as mock_connect:
            run_weaviate_ingestion(
                sample_chunks_with_embeddings,
                host="weaviate.example.com",
                port=8081,
                grpc_port=50052,
            )

        mock_connect.assert_called_once_with(
            host="weaviate.example.com", port=8081, grpc_port=50052
        )

    def test_run_weaviate_ingestion_connection_error(
        self,
    ) -> None:
        """If ``wv.connect_to_local`` raises an exception, the function
        must raise a ``ConnectionError`` with a helpful message."""
        with patch(
            "src.loading.weaviate_loader.wv.connect_to_local",
            side_effect=Exception("connection refused"),
        ):
            with pytest.raises(ConnectionError) as exc_info:
                run_weaviate_ingestion([])

        assert "Cannot connect to Weaviate" in str(exc_info.value)

    def test_run_weaviate_ingestion_drop_if_exists(
        self,
        mock_weaviate_client: MagicMock,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """With ``drop_if_exists=True``, the function should pass the flag
        through to ``create_schema`` so that an existing collection is
        deleted before recreation.

        We use a ``side_effect`` on ``exists`` (True → False) so that
        after the delete the code proceeds to create the collection.
        """
        client = mock_weaviate_client
        client.collections.exists.side_effect = [True, False]

        run_weaviate_ingestion(
            sample_chunks_with_embeddings,
            client=client,
            drop_if_exists=True,
        )

        # Because exists is True and drop_if_exists is True, delete is called
        client.collections.delete.assert_called_once_with("DocumentChunks")
        # After delete, create is called
        client.collections.create.assert_called_once()

    def test_run_weaviate_ingestion_custom_batch_size(
        self,
        mock_weaviate_client: MagicMock,
        many_chunks: List[Dict[str, Any]],
    ) -> None:
        """The ``batch_size`` parameter should be forwarded to
        ``ingest_chunks``, verified by the number of ``insert_many`` calls."""
        client = mock_weaviate_client
        mock_collection = client.collections.get.return_value

        run_weaviate_ingestion(many_chunks, client=client, batch_size=200)

        # 250 chunks / 200 batch = 2 calls (200 + 50)
        assert mock_collection.data.insert_many.call_count == 2

    def test_run_weaviate_ingestion_client_error_does_not_close_own_client(
        self,
        sample_chunks_with_embeddings: List[Dict[str, Any]],
    ) -> None:
        """If *no* client was provided and an error occurs *after* the
        schema is created (e.g., during ingest), the owned client should
        still be closed in the ``finally`` block.

        We simulate an ingest error by making ``insert_many`` raise.
        """
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.data.insert_many.side_effect = RuntimeError("ingest failed")
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        with patch(
            "src.loading.weaviate_loader.wv.connect_to_local",
            return_value=mock_client,
        ):
            with pytest.raises(RuntimeError, match="ingest failed"):
                run_weaviate_ingestion(sample_chunks_with_embeddings)

        # Client should still be closed in finally
        mock_client.close.assert_called_once()
