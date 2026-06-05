"""
Batch multimodal embedding pipeline.

Consumes ``ChunkMetadata`` instances from the chunking stage, dispatches
them by ``embedding_type``, and produces unit-normalized embedding vectors.

Three-step pipeline:

1. ``build_encode_items(chunks)`` — creates a flat list of encode items.
2. ``encode_batch(items, model)`` — batches by type (text vs image) and
   runs the model in sub-batches to avoid OOM.
3. ``attach_embeddings(chunks, embeddings)`` — zips chunk metadata and
   embedding vectors into dicts ready for Weaviate ingestion.

Typical usage::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("clip-ViT-B-32")
    items = build_encode_items(chunk_metadatas)
    embeddings = encode_batch(items, model, batch_size=32)
    docs = attach_embeddings(chunk_metadatas, embeddings)
    # docs is List[dict] with chunk_id, embedding (list[float]), and metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from src.chunking.models import ChunkMetadata

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model protocol (for type-checking — accepts any SentenceTransformer-like)
# ---------------------------------------------------------------------------


class EmbeddingModel(Protocol):
    """Minimal protocol for an embedding model.

    Compatible with ``sentence_transformers.SentenceTransformer`` instances
    and any mock that exposes ``encode()`` with the same signature.
    """

    def encode(
        self,
        sentences: Any,
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Step 1 — Build encode items
# ---------------------------------------------------------------------------


def build_encode_items(
    chunks: List[ChunkMetadata],
) -> List[Dict[str, Any]]:
    """Build a flat list of items to encode, dispatched by ``embedding_type``.

    For each chunk:

    * ``embedding_type = "text"`` or ``"textual_description"``:
      ``content`` is ``chunk.chunk_text``.
    * ``embedding_type = "image"``:
      ``content`` is ``chunk.image_uri[0]`` (the first URI in the list).
      Visual enrichment creates one chunk per visual element, so image
      chunks always have exactly one URI.

    Args:
        chunks: Chunk metadata list from the chunking stage.

    Returns:
        List of dicts with keys ``chunk_id``, ``embedding_type``, ``content``,
        and ``chunk_index`` (position in the original ``chunks`` list).
    """
    items: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if chunk.embedding_type in ("text", "textual_description"):
            items.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "embedding_type": chunk.embedding_type,
                    "content": chunk.chunk_text,
                    "chunk_index": i,
                }
            )
        elif chunk.embedding_type == "image":
            uri = chunk.image_uri[0] if chunk.image_uri else None
            if uri:
                items.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "embedding_type": chunk.embedding_type,
                        "content": uri,
                        "chunk_index": i,
                    }
                )
            else:
                _log.warning(
                    "Skipping image chunk %s — image_uri missing or not found: %s",
                    chunk.chunk_id,
                    uri,
                )
        else:
            _log.warning(
                "Unknown embedding_type '%s' for chunk %s — skipping.",
                chunk.embedding_type,
                chunk.chunk_id,
            )
    _log.info("Built %d encode items from %d chunks.", len(items), len(chunks))
    return items


# ---------------------------------------------------------------------------
# Step 2 — Encode in batches
# ---------------------------------------------------------------------------


def _split_items_by_modality(
    items: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split items into text and image groups.

    Unknown ``embedding_type`` values are logged as a warning and skipped.

    Returns:
        ``(text_items, image_items)``.
    """
    text_items: List[Dict[str, Any]] = []
    image_items: List[Dict[str, Any]] = []

    for item in items:
        etype = item["embedding_type"]
        if etype in ("text", "textual_description"):
            text_items.append(item)
        elif etype == "image":
            image_items.append(item)
        else:
            _log.warning(
                "Unknown embedding_type '%s' for chunk %s — skipping.",
                etype,
                item["chunk_id"],
            )

    return text_items, image_items


def _infer_model_dim(model: Any) -> int:
    """Return the embedding dimension of *model*, or 0 if unknown."""
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension() or 0
    return 0


def encode_batch(
    items: List[Dict[str, Any]],
    model: Any,
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> List[np.ndarray]:
    """Encode *items* using *model*, dispatching by modality.

    Text items are encoded in sub-batches.  Image items are encoded one at
    a time (most image models don't support true batching for images, and
    pre-loading all images into memory risks OOM).

    Embeddings are returned in the **same order** as *items*.

    Args:
        items: Output of ``build_encode_items``.
        model: A ``SentenceTransformer``-compatible model with an
            ``encode()`` method.
        batch_size: Sub-batch size for text encoding.
        show_progress_bar: Passed through to ``model.encode()``.

    Returns:
        List of unit-normalized numpy arrays, one per item.
    """
    if not items:
        _log.info("No items to encode.")
        return []

    text_items, image_items = _split_items_by_modality(items)
    model_dim = _infer_model_dim(model)

    # --- Text encoding ---
    text_embeddings: List[np.ndarray] = []
    if text_items:
        texts = [it["content"] for it in text_items]
        _log.info("Encoding %d text items...", len(texts))
        raw = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,  # we normalize ourselves
        )
        if model_dim == 0 and len(raw) > 0:
            model_dim = raw.shape[-1]
        # Normalize to unit vectors (L2)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        raw = raw / norms
        text_embeddings = [raw[i] for i in range(len(raw))]

    # --- Image encoding ---
    image_embeddings: List[np.ndarray] = []
    if image_items:
        _log.info("Encoding %d image items...", len(image_items))
        for item in image_items:
            uri = item["content"]
            try:
                vec = model.encode(
                    uri,
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )
                if model_dim == 0:
                    model_dim = vec.shape[-1] if vec.ndim > 0 else vec.shape[0]
                # model.encode for a single image may return (d,) or (1,d)
                if vec.ndim == 2:
                    vec = vec[0]
                # Normalize
                norm = np.linalg.norm(vec)
                if norm == 0:
                    norm = 1.0
                vec = vec / norm
                image_embeddings.append(vec)
            except Exception:
                _log.warning(
                    "Failed to encode image %s — using zero vector.", uri, exc_info=True
                )
                image_embeddings.append(
                    np.zeros(model_dim) if model_dim > 0 else np.zeros(1)
                )

    zero_vec = np.zeros(model_dim) if model_dim > 0 else np.zeros(1)

    # --- Merge in original order ---
    result: List[np.ndarray] = []
    t_idx = 0
    i_idx = 0

    for item in items:
        etype = item["embedding_type"]
        if etype in ("text", "textual_description"):
            if t_idx < len(text_embeddings):
                result.append(text_embeddings[t_idx])
                t_idx += 1
            else:
                result.append(zero_vec.copy())
        elif etype == "image":
            if i_idx < len(image_embeddings):
                result.append(image_embeddings[i_idx])
                i_idx += 1
            else:
                result.append(zero_vec.copy())
        else:
            # Unknown types already skipped in _split_items_by_modality
            result.append(zero_vec.copy())

    _log.info(
        "Encoded %d items (%d text, %d image).",
        len(result),
        len(text_embeddings),
        len(image_embeddings),
    )
    return result


# ---------------------------------------------------------------------------
# Step 3 — Attach embeddings
# ---------------------------------------------------------------------------


def attach_embeddings(
    chunks: List[ChunkMetadata],
    embeddings: List[np.ndarray],
    *,
    document_name: str = "",
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Combine chunk metadata with embedding vectors.

    Args:
        chunks: Original chunk metadata list.
        embeddings: Embedding vectors (same length as number of encoded items,
            as returned by ``build_encode_items`` -> ``encode_batch``).
        document_name: Source filename (e.g. ``"2502.04644v1.pdf"``).
            Only used when *output_dir* is provided.
        output_dir: If provided, the assembled docs are exported as a JSON
            file ``{doc_stem}_embeddings.json`` under this directory.

    Returns:
        List of dicts with keys ``chunk_id``, ``embedding`` (``list[float]``),
        ``embedding_type``, ``chunk_text``, ``metadata`` (dict of filterable
        fields for Weaviate).  Chunks that were skipped during encoding
        (e.g. missing image URI) are omitted.
    """
    items = build_encode_items(chunks)

    if len(items) != len(embeddings):
        _log.warning(
            "Mismatch: %d items but %d embeddings. Truncating to minimum.",
            len(items),
            len(embeddings),
        )
        n = min(len(items), len(embeddings))
        items = items[:n]
        embeddings = embeddings[:n]

    docs: List[Dict[str, Any]] = []
    for item, vec in zip(items, embeddings):
        chunk_idx = item["chunk_index"]
        cm = chunks[chunk_idx]

        docs.append(
            {
                "chunk_id": cm.chunk_id,
                "embedding": vec.tolist(),
                "embedding_type": cm.embedding_type,
                "chunk_text": cm.chunk_text,
                "metadata": {
                    "document_name": cm.document_name,
                    "document_hash": cm.document_hash,
                    "document_type": cm.document_type,
                    "chunk_types": cm.chunk_types,
                    "section_path": cm.section_path,
                    "section_headings": cm.section_headings,
                    "page_numbers": cm.page_numbers,
                    "sequence_number": cm.sequence_number,
                    "image_type": cm.image_type,
                    "image_uri": cm.image_uri,
                    "caption_text": cm.caption_text,
                    "caption_number": cm.caption_number,
                    "element_self_refs": cm.element_self_refs,
                    "token_count": cm.token_count,
                    "refers_to": cm.refers_to,
                    "relates_to": cm.relates_to,
                },
            }
        )
    _log.info("Attached embeddings to %d documents.", len(docs))

    # Optional export
    if output_dir is not None and document_name:
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_stem = Path(document_name).stem
        meta_path = output_dir / f"{doc_stem}_embeddings.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable = [{**doc, "embedding": doc["embedding"]} for doc in docs]
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        _log.info("Exported embeddings to %s", meta_path)

    return docs
