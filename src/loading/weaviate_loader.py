"""
Weaviate schema creation and batch ingestion for document chunks.

Consumes ``List[Dict[str, Any]]`` from ``attach_embeddings`` (see
``embedding_pipeline.py``) and writes them into a Weaviate collection
with BYO vectors.

Three public functions:

1. ``create_schema`` — idempotent collection creation with filterable properties.
2. ``ingest_chunks`` — batch insert with configurable batch size and error
   collection.
3. ``run_weaviate_ingestion`` — convenience wrapper that connects, creates
   schema, ingests, and returns a summary dict.

Typical usage::

    from src.loading.weaviate_loader import run_weaviate_ingestion

    summary = run_weaviate_ingestion(chunks_with_embeddings)
    print(summary["successful"], "/", summary["total"], "inserted")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import weaviate as wv
import weaviate.classes.config as wc
from weaviate.collections.classes.data import DataObject

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata → Weaviate DataType mapping
# ---------------------------------------------------------------------------

_PROPERTY_DEFS: List[Dict[str, Any]] = [
    {"name": "document_name", "data_type": wc.DataType.TEXT},
    {"name": "document_hash", "data_type": wc.DataType.INT},
    {"name": "document_type", "data_type": wc.DataType.TEXT},
    {"name": "chunk_types", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "section_path", "data_type": wc.DataType.TEXT},
    {"name": "section_headings", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "page_numbers", "data_type": wc.DataType.INT_ARRAY},
    {"name": "sequence_number", "data_type": wc.DataType.INT},
    {"name": "image_type", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "image_uri", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "caption_text", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "caption_number", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "element_self_refs", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "token_count", "data_type": wc.DataType.INT},
    {"name": "chunk_text", "data_type": wc.DataType.TEXT},
    {"name": "embedding_type", "data_type": wc.DataType.TEXT},
    {"name": "refers_to", "data_type": wc.DataType.TEXT_ARRAY},
    {"name": "relates_to", "data_type": wc.DataType.TEXT_ARRAY},
]


# ---------------------------------------------------------------------------
# Function 1 — Create schema
# ---------------------------------------------------------------------------


def create_schema(
    client: wv.WeaviateClient,
    collection_name: str = "DocumentChunks",
    *,
    drop_if_exists: bool = False,
) -> wv.collections.Collection:
    """Create (or recreate) a Weaviate collection for document chunks.

    The collection uses BYO vectors (``Vectorizer.none()``) with an HNSW
    index and COSINE distance metric.  All filterable metadata fields are
    declared as properties with appropriate Weaviate data types.

    Args:
        client: Connected Weaviate client.
        collection_name: Name of the collection to create.
        drop_if_exists: If ``True`` and the collection already exists,
            delete it first before recreating.

    Returns:
        The newly created (or existing) collection object.
    """
    if drop_if_exists and client.collections.exists(collection_name):
        _log.info("Dropping existing collection '%s' ...", collection_name)
        client.collections.delete(collection_name)

    if client.collections.exists(collection_name):
        _log.info("Collection '%s' already exists — reusing.", collection_name)
        return client.collections.get(collection_name)

    properties = [
        wc.Property(name=p["name"], data_type=p["data_type"]) for p in _PROPERTY_DEFS
    ]

    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=wc.Configure.Vectorizer.none(),
        vector_index_config=wc.Configure.VectorIndex.hnsw(
            distance_metric=wc.VectorDistances.COSINE,
        ),
        properties=properties,
    )

    _log.info(
        "Created collection '%s' with %d properties.", collection_name, len(properties)
    )
    return collection


# ---------------------------------------------------------------------------
# Function 2 — Ingest chunks
# ---------------------------------------------------------------------------


def ingest_chunks(
    client: wv.WeaviateClient,
    chunks_with_embeddings: List[Dict[str, Any]],
    collection_name: str = "DocumentChunks",
    *,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """Batch-ingest document chunks with embeddings into a Weaviate collection.

    Args:
        client: Connected Weaviate client.
        chunks_with_embeddings: List of dicts from ``attach_embeddings``.
            Each dict must contain ``chunk_id`` (UUID string), ``embedding``
            (``list[float]``), and ``metadata`` (dict of filterable fields).
        collection_name: Target collection name.
        batch_size: Number of objects per ``insert_many`` call.

    Returns:
        Summary dict with keys ``total``, ``successful``, ``failed``, ``errors``.
    """
    collection = client.collections.get(collection_name)

    total = len(chunks_with_embeddings)
    successful = 0
    failed = 0
    all_errors: List[str] = []

    # Build DataObject list
    objects: List[DataObject] = []
    for chunk in chunks_with_embeddings:
        objects.append(
            DataObject(
                properties=chunk["metadata"],
                uuid=chunk["chunk_id"],
                vector=chunk["embedding"],
            )
        )

    # Insert in sub-batches
    for start in range(0, total, batch_size):
        batch = objects[start : start + batch_size]
        result = collection.data.insert_many(batch)

        n_batch = len(batch)
        if result.has_errors:
            # errors is a dict[int, ErrorObject]
            for idx, error_obj in result.errors.items():
                all_errors.append(f"[{idx}] {error_obj.message}")
            n_errors = len(result.errors)
            failed += n_errors
            successful += n_batch - n_errors
            _log.warning(
                "%d/%d objects in batch [%d:%d] had errors.",
                n_errors,
                n_batch,
                start,
                start + n_batch,
            )
        else:
            successful += n_batch

    if all_errors:
        _log.warning("Ingest finished with %d/%d failures.", failed, total)

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "errors": all_errors,
    }


# ---------------------------------------------------------------------------
# Function 3 — Run full ingestion
# ---------------------------------------------------------------------------


def run_weaviate_ingestion(
    chunks_with_embeddings: List[Dict[str, Any]],
    *,
    client: Optional[wv.WeaviateClient] = None,
    collection_name: str = "DocumentChunks",
    drop_if_exists: bool = False,
    batch_size: int = 100,
    host: str = "localhost",
    port: int = 8080,
    grpc_port: int = 50051,
) -> Dict[str, Any]:
    """Convenience: connect to Weaviate, create schema, ingest chunks.

    Args:
        chunks_with_embeddings: Output from ``attach_embeddings``.
        client: An already-connected Weaviate client.  If ``None`` a new
            client is created for ``(host, port, grpc_port)`` and closed
            after ingestion.
        collection_name: Weaviate collection name.
        drop_if_exists: Passed to ``create_schema``.
        batch_size: Batch size for ``ingest_chunks``.
        host: Weaviate host (used when *client* is not provided).
        port: Weaviate HTTP port.
        grpc_port: Weaviate gRPC port.

    Returns:
        Ingestion summary dict (see ``ingest_chunks``).
    """
    own_client = client is None

    if own_client:
        try:
            _log.info("Connecting to Weaviate at %s:%d ...", host, port)
            client = wv.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Weaviate at {host}:{port} — {exc}"
            ) from exc

    try:
        # Step 1 — create schema
        create_schema(client, collection_name, drop_if_exists=drop_if_exists)
        _log.info("Schema created for collection '%s'.", collection_name)

        # Step 2 — ingest
        n = len(chunks_with_embeddings)
        _log.info("Ingesting %d chunks in batches of %d ...", n, batch_size)
        summary = ingest_chunks(
            client,
            chunks_with_embeddings,
            collection_name,
            batch_size=batch_size,
        )
        _log.info(
            "Ingestion complete: %d/%d successful.",
            summary["successful"],
            summary["total"],
        )
        return summary
    finally:
        if own_client and client is not None:
            client.close()
