"""
Relates-to population via top-k cosine similarity on embedding vectors.

For each chunk, computes the top-*k* most similar chunks (excluding self
and siblings), applies a minimum similarity threshold, and populates
``relates_to`` with the resulting chunk IDs.

Pure numpy — no HDBSCAN, no sklearn clustering.

Typical usage::

    sim_matrix = compute_cosine_similarity_matrix(embeddings)
    enriched = populate_relates_to(chunks, embeddings, top_k=3)
    # enriched[i].relates_to now contains up to 3 chunk IDs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.chunking.models import ChunkMetadata

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity matrix
# ---------------------------------------------------------------------------


def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the ``n × n`` cosine similarity matrix from embedding vectors.

    Vectors are re-normalized to unit length before the dot product, so
    the result is in [−1, 1] regardless of whether inputs were already
    normalized.

    Args:
        embeddings: ``(n, d)`` array where row *i* is the embedding for
            ``chunks[i]``.

    Returns:
        ``(n, n)`` float64 array where ``M[i, j]`` is the cosine
        similarity between chunk *i* and chunk *j*.

    Raises:
        ValueError: If *embeddings* is empty (n == 0).
    """
    if embeddings.size == 0:
        raise ValueError("Cannot compute similarity on empty embeddings array.")

    # Ensure float64 and re-normalize
    vecs = embeddings.astype(np.float64, copy=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Guard against zero-norm rows (already guarded in encode_batch, but safe)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = vecs / norms

    # n × n dot product matrix — cache-friendly via einsum
    # Equivalent to vecs @ vecs.T but explicit about the op
    sim = np.einsum("ij,kj->ik", vecs, vecs, optimize=True)
    return sim


# ---------------------------------------------------------------------------
# Sibling detection
# ---------------------------------------------------------------------------


def _build_sibling_mask(
    chunks: List[ChunkMetadata],
) -> np.ndarray:
    """Build a boolean mask where ``mask[i, j]`` is True if chunks *i* and *j*
    share any ``element_self_ref`` (they are sibling representations of
    the same content — e.g. image ↔ textual_description).
    """
    n = len(chunks)
    mask = np.zeros((n, n), dtype=bool)

    # Pre-compute frozenset of refs per chunk for fast intersection
    ref_sets = [frozenset(cm.element_self_refs) for cm in chunks]

    for i in range(n):
        si = ref_sets[i]
        if not si:  # no refs → no siblings
            continue
        for j in range(i + 1, n):
            if si & ref_sets[j]:
                mask[i, j] = True
                mask[j, i] = True

    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def populate_relates_to(
    chunks: List[ChunkMetadata],
    embeddings: np.ndarray,
    *,
    top_k: int = 3,
    min_similarity: float = 0.75,
    document_name: str = "",
    output_dir: Optional[Path] = None,
) -> List[ChunkMetadata]:
    """Populate ``relates_to`` for each chunk via top-*k* cosine similarity.

    For each chunk *i*:
    1. Compute top-*k* most similar chunks by cosine similarity.
    2. Exclude self (index *i*).
    3. Exclude siblings (chunks sharing ``element_self_refs``).
    4. Exclude candidates below ``min_similarity``.
    5. Sort remaining candidates by descending similarity and assign
       their ``chunk_id`` values to ``relates_to``.

    Original chunks are never mutated — new ``ChunkMetadata`` instances
    are returned via ``model_copy``.

    Args:
        chunks: Chunk metadata list (from chunking → embedding stages).
        embeddings: ``(n, d)`` embedding array where row *i*
            corresponds to ``chunks[i]``.
        top_k: Maximum number of related chunks to return per chunk.
        min_similarity: Minimum cosine similarity to include a candidate
            (range [−1, 1]).  Default 0.75.
        document_name: Source filename (e.g. ``"2502.04644v1.pdf"``).
            Only used when *output_dir* is provided.
        output_dir: If provided, the enriched chunk metadata is exported as
            ``{doc_stem}_chunks_metadata_with_relates_to.json``.

    Returns:
        New list of ``ChunkMetadata`` with ``relates_to`` populated.
    """
    n = len(chunks)
    if n <= 1:
        # 0 or 1 chunk — nothing to relate
        _log.info("Only %d chunk(s) — skipping relates_to computation.", n)
        return list(chunks)

    if embeddings.shape[0] != n:
        raise ValueError(
            f"Embeddings shape {embeddings.shape} does not match "
            f"chunk count {n}. Expected ({n}, d)."
        )

    _log.info("Computing cosine similarity matrix for %d chunks...", n)
    sim = compute_cosine_similarity_matrix(embeddings)

    _log.info("Building sibling mask...")
    sibling_mask = _build_sibling_mask(chunks)

    # For each chunk *i*, find top-*(k+1+max_siblings)* most similar
    # to leave enough candidates after filtering self and siblings.
    # Using argpartition avoids a full O(n log n) sort per row.
    max_siblings = int(sibling_mask.sum(axis=1).max())
    k_search = min(top_k + 1 + max_siblings, n)

    # Use argpartition to find top k_search indices per row
    # negate sim so highest similarity = smallest negative = partitioned first
    # Actually argpartition on -sim puts largest values first (partition on kth)
    neg_sim = -sim
    partitioned_indices = np.argpartition(neg_sim, kth=min(k_search, n - 1), axis=1)

    _log.info(
        "Populating relates_to (top_k=%d, min_similarity=%.2f)...",
        top_k,
        min_similarity,
    )

    result: List[ChunkMetadata] = []
    for i in range(n):
        # Get top k_search indices for chunk i (first k_search are the
        # most similar after argpartition with kth=k_search-1)
        candidates = partitioned_indices[i, :k_search]

        related: List[str] = []
        for j in candidates:
            if j == i:  # exclude self
                continue
            if sibling_mask[i, j]:  # exclude siblings
                continue
            score = sim[i, j].item()
            if score < min_similarity:
                continue
            related.append((score, chunks[j].chunk_id))

        # Sort by descending similarity and take top_k
        related.sort(key=lambda x: x[0], reverse=True)
        top_ids = [cid for _, cid in related[:top_k]]

        if top_ids:
            result.append(chunks[i].model_copy(update={"relates_to": top_ids}))
        else:
            result.append(chunks[i])  # no change needed

    populated = sum(1 for cm in result if cm.relates_to)
    _log.info(
        "relates_to populated: %d/%d chunks have non-empty relates_to.",
        populated,
        n,
    )

    # Optional export
    if output_dir is not None and document_name:
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_stem = Path(document_name).stem
        meta_path = output_dir / f"{doc_stem}_chunks_metadata_with_relates_to.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in result], f, indent=2)
        _log.info("Exported relates_to-enriched chunk metadata to %s", meta_path)

    return result
