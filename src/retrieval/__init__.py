"""
``src.retrieval`` — embedding, vector DB, and graph DB pipeline.

Retrieval-layer modules that consume ``ChunkMetadata`` instances from
the chunking stage and prepare them for Weaviate / Neo4j storage.
"""

from src.retrieval.embedding_pipeline import (
    attach_embeddings,
    build_encode_items,
    encode_batch,
)
from src.retrieval.similarity import (
    compute_cosine_similarity_matrix,
    populate_relates_to,
)

__all__ = [
    "attach_embeddings",
    "build_encode_items",
    "encode_batch",
    "compute_cosine_similarity_matrix",
    "populate_relates_to",
]
