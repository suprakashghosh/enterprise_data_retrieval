"""
Cross-reference resolution.

Scans every chunk's text for caption references (e.g. "see Figure 3",
"Table IV") and populates ``refers_to`` with the chunk IDs of the
referenced visual elements.

Typical usage::

    resolved = resolve_cross_references(
        chunk_metadatas, pic_table_lookup, text_lookup
    )
    # resolved[i].refers_to now contains target chunk IDs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.chunking._caption_utils import get_caption_for_item
from src.chunking.models import ChunkMetadata
from src.utils.caption_extractor import extract_caption_label, find_all_caption_refs

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption index builder
# ---------------------------------------------------------------------------


def _build_caption_index(
    pic_table_lookup: Dict[str, Any],
    text_lookup: Dict[str, str],
    chunk_metadatas: List[ChunkMetadata],
) -> Dict[str, str]:
    """Build a lookup from normalised caption label to chunk ID.

    Iterates every picture/table item in ``pic_table_lookup``, extracts
    its caption text, normalises it via ``extract_caption_label``, and
    finds which chunk contains that visual element.

    Returns:
        ``Dict[label -> chunk_id]`` where label is e.g. ``"figure 1"``.
    """
    # Build reverse lookup: element_self_ref -> chunk_id
    ref_to_chunk: Dict[str, str] = {}
    for cm in chunk_metadatas:
        for ref in cm.element_self_refs:
            ref_to_chunk[ref] = cm.chunk_id

    index: Dict[str, str] = {}
    for ref, item in pic_table_lookup.items():
        caption_text = get_caption_for_item(item, text_lookup)
        if not caption_text:
            continue

        label = extract_caption_label(caption_text)
        if label is None:
            continue

        chunk_id = ref_to_chunk.get(ref)
        if chunk_id is None:
            continue

        # Later items in the same chunk with the same label will overwrite
        # (e.g. "Figure 1" appearing in two different pic items in the same chunk).
        if label in index and index[label] != chunk_id:
            _log.debug(
                "Caption label '%s' overwritten: %s -> %s",
                label,
                index[label],
                chunk_id,
            )
        index[label] = chunk_id

    _log.info("Caption index built: %d entries.", len(index))
    return index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_cross_references(
    chunk_metadatas: List[ChunkMetadata],
    pic_table_lookup: Dict[str, Any],
    text_lookup: Dict[str, str],
    document_name: str,
    output_dir:Optional[Path] = None,
) -> List[ChunkMetadata]:
    """Resolve caption references in chunk text and populate ``refers_to``.

    For each chunk, scans its ``chunk_text`` for caption references using
    ``find_all_caption_refs``, looks up each reference in a caption index
    built from ``pic_table_lookup``, and returns new ``ChunkMetadata``
    instances with ``refers_to`` populated.

    Self-references (a chunk referencing its own visual element) are
    filtered out.  References to captions not in the index are silently
    skipped.

    Args:
        chunk_metadatas: Existing chunk list (from
            ``build_chunk_metadata_list`` + ``enrich_visual_chunks``).
        pic_table_lookup: ``self_ref -> Docling item`` for all
            pictures and tables.
        text_lookup: ``self_ref -> text string`` mapping.

    Returns:
        New list of ``ChunkMetadata`` with ``refers_to`` populated.
        Original chunks are never mutated (they are frozen).
    """
    if not pic_table_lookup:
        _log.info("No visual elements — skipping cross-reference resolution.")
        return chunk_metadatas

    index = _build_caption_index(pic_table_lookup, text_lookup, chunk_metadatas)

    resolved: List[ChunkMetadata] = []
    for cm in chunk_metadatas:
        refs = find_all_caption_refs(cm.chunk_text)
        if not refs:
            resolved.append(cm)
            continue

        target_ids: set[str] = set()
        for ref_label in refs:
            target_id = index.get(ref_label)
            if target_id is not None and target_id != cm.chunk_id:
                target_ids.add(target_id)

        if target_ids:
            new_cm = cm.model_copy(update={"refers_to": sorted(target_ids)})
            resolved.append(new_cm)
        else:
            resolved.append(cm)

    ref_count = sum(1 for cm in resolved if cm.refers_to)
    _log.info(
        "Cross-reference resolution: %d/%d chunks have non-empty refers_to.",
        ref_count,
        len(resolved),
    )

    # Optional export
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_stem = Path(document_name).stem

        meta_path = output_dir / f"{doc_stem}_chunks_metadata_with_refers_to.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in resolved], f, indent=2)

        _log.info("Exported chunk files to %s", output_dir)
    return resolved
