"""
``src.loading`` — embedding, vector DB, and graph DB pipeline.

Retrieval-layer modules that consume ``ChunkMetadata`` instances from
the chunking stage and prepare them for Weaviate / Neo4j storage.
"""

from src.loading.embedding_pipeline import (
    attach_embeddings,
    build_encode_items,
    encode_batch,
)
from src.loading.similarity import (
    compute_cosine_similarity_matrix,
    populate_relates_to,
)
from src.loading.weaviate_loader import (
    create_schema,
    ingest_chunks,
    run_weaviate_ingestion,
)

__all__ = [
    "attach_embeddings",
    "build_encode_items",
    "encode_batch",
    "compute_cosine_similarity_matrix",
    "populate_relates_to",
    "create_schema",
    "ingest_chunks",
    "run_weaviate_ingestion",
]
