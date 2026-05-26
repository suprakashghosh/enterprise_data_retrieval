"""
Visual element enrichment.

For each picture or table in a Docling document, this module:
1. extracts the caption text,
2. generates an LLM-powered textual description via the instructor API,
3. creates two additional ``ChunkMetadata`` entries:

   * ``embedding_type = "image"`` — the image at ``image_uri`` is embedded by
     the multimodal encoder; ``chunk_text`` holds the caption for reference.
   * ``embedding_type = "textual_description"`` — the LLM description is
     embedded; ``chunk_text`` holds the description.

Unlike the original ``test.py`` prototype, LLM calls are made concurrently
via ``asyncio.gather`` + a semaphore — not with ``asyncio.run()`` per item.

Typical usage::

    all_chunks = enrich_visual_chunks(
        chunk_metadatas=build_chunk_metadata_list(conv_result),
        pic_table_lookup=pic_table_lookup,
        text_lookup=text_lookup,
        pic_table_section_lookup=pic_table_section_lookup,
        document_name=document_name,
        document_type=document_type,
        document_hash=document_hash,
    )
    # all_chunks now includes text + image + textual_description chunks
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.chunking._caption_utils import get_caption_for_item
from src.chunking.models import ChunkMetadata, make_chunk_id
from src.utils.caption_extractor import extract_caption_label
from src.utils.instructor_api_response import get_llm_response_from_instructor

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — label normalisation
# ---------------------------------------------------------------------------


def _label_value(item: Any) -> str:
    """Extract the DocItemLabel as a plain lowercase string."""
    label = getattr(item, "label", None)
    if label is None:
        return ""
    raw = label.value if hasattr(label, "value") else str(label)
    return raw.lower()


# ---------------------------------------------------------------------------
# Async LLM description
# ---------------------------------------------------------------------------


class _ImageDescription(BaseModel):
    """Pydantic model for structured image description output."""

    description: str = Field(
        ...,
        description="A detailed description of the image capturing all main features and facets.",
    )
    keywords: List[str] = Field(
        ...,
        description="Keywords to identify the image for BM25 retrieval.",
    )


async def _describe_image_async(
    image_url: str,
    image_caption: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate a description + keywords for a single image via the LLM."""
    system_prompt = (
        "You are an experienced image annotator. "
        "You will be provided with an image from a document. "
        "The image can be a picture, chart, graph, table, formula, etc. "
        "You must accurately provide the details."
    )
    user_input = (
        f"Please find the image attached. The image caption is: {image_caption}."
    )

    async with semaphore:
        response = await get_llm_response_from_instructor(
            user_input=user_input,
            system_prompt=system_prompt,
            response_format=_ImageDescription,
            image_url=image_url,
        )

    return (
        f"Image Caption: {image_caption}\n"
        f"Image Description: {response.description}\n"
        f"Keywords: {response.keywords}"
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def generate_all_image_descriptions(
    pic_table_lookup: Dict[str, Any],
    text_lookup: Dict[str, str],
    max_concurrent: int = 5,
) -> Dict[str, str]:
    """Generate LLM descriptions for all pictures and tables concurrently.

    Args:
        pic_table_lookup: self_ref → Docling picture/table item mapping.
        text_lookup: self_ref → text mapping (for caption resolution).
        max_concurrent: Maximum number of concurrent LLM API calls.

    Returns:
        Dict mapping self_ref → generated description text.  Failed items
        produce an empty string (errors are logged but do not halt processing).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    coros = []
    for ref, item in pic_table_lookup.items():
        caption = get_caption_for_item(item, text_lookup)
        uri = str(getattr(getattr(item, "image", None), "uri", None))
        if uri and caption:
            coros.append(_describe_image_async(uri, caption, semaphore))
        else:
            coros.append(asyncio.sleep(0))  # no-op placeholder

    results = await asyncio.gather(*coros, return_exceptions=True)

    descriptions: Dict[str, str] = {}
    for (ref, _), result in zip(pic_table_lookup.items(), results):
        if isinstance(result, Exception):
            _log.warning("Image description failed for %s: %s", ref, result)
            descriptions[ref] = ""
        else:
            descriptions[ref] = result or ""

    return descriptions


# ---------------------------------------------------------------------------
# Public sync API — enriches an existing chunk list with visual chunks
# ---------------------------------------------------------------------------


def enrich_visual_chunks(
    chunk_metadatas: List[ChunkMetadata],
    *,
    pic_table_lookup: Dict[str, Any],
    text_lookup: Dict[str, str],
    pic_table_section_lookup: Dict[str, List[str]],
    document_name: str,
    document_type: str,
    document_hash: str,
    max_concurrent: int = 5,
) -> List[ChunkMetadata]:
    """Create additional ``ChunkMetadata`` entries for every picture / table.

    For each visual element, two new chunks are appended:
    1. ``embedding_type="image"`` — ``image_uri`` is embedded, ``chunk_text``
       holds the caption for reference / display only.
    2. ``embedding_type="textual_description"`` — the LLM-generated description
       is embedded.

    Sequence numbers continue from the last text chunk.
    LLM calls are batched and issued concurrently.

    Args:
        chunk_metadatas: Existing list of text chunks (from
            ``build_chunk_metadata_list``).
        pic_table_lookup: self_ref → Docling item for all pictures/tables.
        text_lookup: self_ref → text string mapping.
        pic_table_section_lookup: Accumulator of section headings per visual
            element ref (populated by ``build_chunk_metadata_list``).
        document_name: Source filename.
        document_type: MIME type.
        document_hash: Docling binary hash.
        max_concurrent: Max concurrent LLM API calls.

    Returns:
        Combined list: original text chunks + newly created visual chunks.
    """
    if not pic_table_lookup:
        _log.info("No visual elements found — skipping visual enrichment.")
        return chunk_metadatas

    _log.info(
        "Generating descriptions for %d visual elements (max %d concurrent)...",
        len(pic_table_lookup),
        max_concurrent,
    )
    descriptions = asyncio.run(
        generate_all_image_descriptions(pic_table_lookup, text_lookup, max_concurrent)
    )
    desc_count = sum(1 for v in descriptions.values() if v)
    _log.info(
        "Generated %d descriptions out of %d elements.", desc_count, len(descriptions)
    )

    sequence_number = len(chunk_metadatas)
    visual_chunks: List[ChunkMetadata] = []

    for ref, item in pic_table_lookup.items():
        label = _label_value(item)
        # Canonicalize DocItemLabel values to our Literal
        if label == "picture":
            image_type_label = "picture"  # type: Literal["picture", "table"]
        elif label == "table":
            image_type_label = "table"
        else:
            continue  # unknown visual type — skip

        image_uri = str(getattr(getattr(item, "image", None), "uri", None))
        caption_text = get_caption_for_item(item, text_lookup)

        # Caption number extraction
        caption_number: Optional[str] = None
        if caption_text:
            result = extract_caption_label(caption_text)
            if result:
                caption_number = result

        # Section info (deduplicated, order preserved)
        raw_headings: List[str] = pic_table_section_lookup.get(ref, [])
        section_headings = list(dict.fromkeys(raw_headings))
        section_path = " > ".join(section_headings)

        # Page numbers from the item's provenance
        page_numbers = sorted(
            {
                getattr(p, "page_no", 0)
                for p in (getattr(item, "prov", None) or [])
                if getattr(p, "page_no", None) is not None
            }
        )

        element_self_refs: List[str] = [ref]
        chunk_types: List[str] = [image_type_label]

        # ── Chunk type A: Image embedding ──
        sequence_number += 1
        visual_chunks.append(
            ChunkMetadata(
                chunk_id=make_chunk_id(document_hash, sequence_number, chunk_types),
                document_name=document_name,
                document_type=document_type,
                document_hash=document_hash,
                embedding_type="image",
                chunk_text=caption_text or "",
                chunk_types=chunk_types,
                section_path=section_path,
                section_headings=section_headings,
                page_numbers=page_numbers,
                sequence_number=sequence_number,
                image_type=[image_type_label],
                image_uri=[image_uri],
                caption_text=[caption_text or ""],
                caption_number=[caption_number or ""],
                element_self_refs=element_self_refs,
            )
        )

        # ── Chunk type B: Textual description embedding ──
        description_text = descriptions.get(ref, "")
        sequence_number += 1
        visual_chunks.append(
            ChunkMetadata(
                chunk_id=make_chunk_id(document_hash, sequence_number, chunk_types),
                document_name=document_name,
                document_type=document_type,
                document_hash=document_hash,
                embedding_type="textual_description",
                chunk_text=description_text,
                chunk_types=chunk_types,
                section_path=section_path,
                section_headings=section_headings,
                page_numbers=page_numbers,
                sequence_number=sequence_number,
                image_type=[image_type_label],
                image_uri=[image_uri],
                caption_text=[caption_text or ""],
                caption_number=[caption_number or ""],
                element_self_refs=element_self_refs,
            )
        )

    _log.info("Created %d additional visual chunks.", len(visual_chunks))
    return chunk_metadatas + visual_chunks
