"""
Docling HybridChunker wrapper.

Wraps Docling's built-in ``HybridChunker`` to produce a list of
``ChunkMetadata`` objects.  Extracts all metadata fields (page numbers,
section paths, captions, element references) directly from Docling's
``DocChunk`` model — no custom element registry or normalization step
required.

Typical usage::

    chunk_metadatas = build_chunk_metadata_list(
        conv_result,
        max_tokens=300,
        output_dir=Path("outputs/doc_abc/"),
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from src.chunking.models import ChunkMetadata, make_chunk_id
from src.chunking.visual_enricher import enrich_visual_chunks
from src.utils.caption_extractor import extract_caption_label

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_hybrid_chunker(
    tokenizer_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 300,
    merge_peers: bool = True,
) -> HybridChunker:
    """Create a configured ``HybridChunker`` with a HuggingFace tokenizer.

    Args:
        tokenizer_model_id: HF model ID used for token counting.
        max_tokens: Target maximum token count per chunk.
        merge_peers: Whether to merge peer chunks (default ``True``).

    Returns:
        A ready-to-use ``HybridChunker``.
    """
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=merge_peers)


# ---------------------------------------------------------------------------
# Lookup builders
# ---------------------------------------------------------------------------


def _build_text_lookup(doc: Any) -> Dict[str, str]:
    """Build O(1) lookup from self_ref to text for all text items in the document."""
    texts = getattr(doc, "texts", [])
    return {
        getattr(t, "self_ref"): getattr(t, "text", "")
        for t in texts
        if getattr(t, "self_ref", None)
    }


def _build_picture_table_lookup(doc: Any) -> Dict[str, Any]:
    """Build O(1) lookup from self_ref to Docling item for all pictures and tables."""
    pictures = getattr(doc, "pictures", [])
    tables = getattr(doc, "tables", [])
    lookup: Dict[str, Any] = {}
    for item in pictures + tables:
        ref = getattr(item, "self_ref", None)
        if ref:
            lookup[ref] = item
    return lookup


def _build_picture_table_section_lookup(
    pic_table_lookup: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Initialise a section-name accumulator per picture/table ref."""
    return {ref: [] for ref in pic_table_lookup}


def find_n_th_chunk_with_label(
    iter: Iterable[BaseChunk], n: int, label: DocItemLabel
) -> Optional[DocChunk]:
    num_found = -1
    for i, chunk in enumerate(iter):
        doc_chunk = DocChunk.model_validate(chunk)
        for it in doc_chunk.meta.doc_items:
            if it.label == label:
                num_found += 1
                if num_found == n:
                    return i, chunk
    return None, None


def print_chunk(chunks, chunk_pos):
    chunk = chunks[chunk_pos]
    ctx_text = chunker.contextualize(chunk=chunk)
    num_tokens = tokenizer.count_tokens(text=ctx_text)
    doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
    title = f"{chunk_pos=} {num_tokens=} {doc_items_refs=}"
    # console.print(Panel(ctx_text, title=title))


# ---------------------------------------------------------------------------
# Caption extraction
# ---------------------------------------------------------------------------


def _extract_caption_text(item: Any, text_lookup: Dict[str, str]) -> Optional[str]:
    """Extract the caption string for a picture or table Docling item.

    Tries ``captions[0].cref`` first, then ``children[0].cref`` as fallback.
    """
    captions = getattr(item, "captions", None)
    children = getattr(item, "children", None)

    cref = None
    if captions:
        cref = getattr(captions[0], "cref", None)
    elif children:
        cref = getattr(children[0], "cref", None)

    if cref:
        return text_lookup.get(cref, "")
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_document(dl_doc: Any, chunker: HybridChunker) -> List[Any]:
    """Run the HybridChunker on a Docling document.

    Returns a materialised list of chunk objects (each a ``DocChunk``).
    """
    return list(chunker.chunk(dl_doc=dl_doc))


# ---------------------------------------------------------------------------
# Metadata extraction from a single chunk
# ---------------------------------------------------------------------------


def _label_str(item: Any) -> str:
    """Extract the label value as a plain string from a DocItem."""
    label = getattr(item, "label", None)
    if label is None:
        return "unknown"
    return label.value if hasattr(label, "value") else str(label)


def extract_chunk_metadata(
    chunk: Any,
    *,
    sequence_number: int,
    text_lookup: Dict[str, str],
    pic_table_lookup: Dict[str, Any],
    pic_table_section_lookup: Dict[str, List[str]],
    document_name: str,
    document_type: str,
    document_hash: str,
    tokenizer: Any,
    chunker: Any,
) -> ChunkMetadata:
    """Extract a full ``ChunkMetadata`` from a single Docling chunk.

    Args:
        chunk: A raw chunk from ``HybridChunker.chunk()``.
        sequence_number: Global chunk index (0-based).
        text_lookup: self_ref → text mapping.
        pic_table_lookup: self_ref → Docling picture/table item mapping.
        pic_table_section_lookup: Accumulator mapping ref → section headings.
        document_name: Source filename.
        document_type: MIME type.
        document_hash: Docling binary hash of the source document.
        tokenizer: ``HuggingFaceTokenizer`` for counting tokens.

    Returns:
        A frozen ``ChunkMetadata`` instance.
    """
    doc_chunk = DocChunk.model_validate(chunk)
    meta = doc_chunk.meta
    doc_items = meta.doc_items

    chunk_types: List[str] = []
    element_self_refs: List[str] = []
    page_numbers: set[int] = set()

    image_types: List[Literal["picture", "table"]] = []
    image_uris: List[str] = []
    caption_texts: List[str] = []
    caption_numbers: List[str] = []

    for item in doc_items:
        label = _label_str(item)
        chunk_types.append(label)
        element_self_refs.append(item.self_ref)

        # Page provenance
        for prov in getattr(item, "prov", []) or []:
            pn = getattr(prov, "page_no", None)
            if pn is not None:
                page_numbers.add(pn)

        # Visual element handling
        if label in ("picture", "table"):
            ref = item.self_ref
            pic_item = pic_table_lookup.get(ref)
            if pic_item is not None:
                image_types.append(label)  # type: ignore[arg-type]
                img = getattr(pic_item, "image", None)
                image_uris.append(str(getattr(img, "uri", None)) if img else "")
                cap_text = _extract_caption_text(pic_item, text_lookup) or ""
                caption_texts.append(cap_text)
                cap_num = ""
                if cap_text:
                    result = extract_caption_label(cap_text)
                    if result:
                        cap_num = result
                caption_numbers.append(cap_num)

    # Derived fields
    section_headings = list(meta.headings or [])
    section_path = " > ".join(section_headings)
    chunk_text = chunker.contextualize(chunk=chunk)  # type: ignore[call-arg]
    token_count_raw = tokenizer.count_tokens(text=chunk_text)

    # Accumulate section info per picture/table (for visual_enricher later)
    for item in doc_items:
        label = _label_str(item)
        if label in ("picture", "table"):
            ref = item.self_ref
            if ref in pic_table_section_lookup:
                pic_table_section_lookup[ref].extend(section_headings)

    return ChunkMetadata(
        chunk_id=make_chunk_id(document_hash, sequence_number, chunk_types),
        document_name=document_name,
        document_type=document_type,
        document_hash=document_hash,
        embedding_type="text",
        chunk_text=chunk_text,
        chunk_types=chunk_types,
        section_path=section_path,
        section_headings=section_headings,
        page_numbers=sorted(page_numbers),
        sequence_number=sequence_number,
        image_type=image_types,  # type: ignore[arg-type]
        image_uri=image_uris,
        caption_text=caption_texts,
        caption_number=caption_numbers,
        element_self_refs=element_self_refs,
        token_count=token_count_raw,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_chunk_metadata_list(
    conv_result: Any,
    *,
    tokenizer_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 300,
    merge_peers: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[List[ChunkMetadata], Dict[str, str], Dict[str, Any]]:
    """Chunk a Docling document and return metadata for every chunk.

    This is the primary entry point for the chunking stage. It produces text
    chunks via the HybridChunker and then enriches them with metadata for
    visual elements (pictures/tables) to create a complete list of chunks.

    Args:
        conv_result: A Docling ``ConversionResult`` (has ``.document``).
        tokenizer_model_id: HF tokenizer for token counting.
        max_tokens: Target max tokens per chunk.
        merge_peers: ``HybridChunker`` merge_peers setting.
        output_dir: If provided, raw chunk JSON and metadata JSON are written here
            as ``{stem}_chunks.json`` and ``{stem}_chunks_metadata.json``.

    Returns:
        List of ``ChunkMetadata`` for all chunks (text and visual) produced
        by the chunking process.
        The text lookup dictionary {text_ref: text}
        The picture object lookup dictionary {picture_ref: picture_object}

    """
    dl_doc = conv_result.document
    document_name = dl_doc.origin.filename
    document_type = dl_doc.origin.mimetype
    document_hash = getattr(dl_doc.origin, "binary_hash", "")

    chunker = create_hybrid_chunker(
        tokenizer_model_id=tokenizer_model_id,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
    )

    text_lookup = _build_text_lookup(dl_doc)
    pic_table_lookup = _build_picture_table_lookup(dl_doc)
    pic_table_section_lookup = _build_picture_table_section_lookup(pic_table_lookup)

    chunks = chunk_document(dl_doc, chunker)
    _log.info("HybridChunker produced %d chunks for %s", len(chunks), document_name)

    chunk_metadatas: List[ChunkMetadata] = []

    for i, chunk in enumerate(chunks):
        metadata = extract_chunk_metadata(
            chunk=chunk,
            sequence_number=i,
            text_lookup=text_lookup,
            pic_table_lookup=pic_table_lookup,
            pic_table_section_lookup=pic_table_section_lookup,
            document_name=document_name,
            document_type=document_type,
            document_hash=document_hash,
            tokenizer=chunker.tokenizer,
            chunker=chunker,
        )
        chunk_metadatas.append(metadata)

    # Update chunk_metadatas with the metadata for the visual chunks
    chunk_metadatas = enrich_visual_chunks(
        chunk_metadatas=chunk_metadatas,
        text_lookup=text_lookup,
        pic_table_lookup=pic_table_lookup,
        pic_table_section_lookup=pic_table_section_lookup,
        document_name=document_name,
        document_type=document_type,
        document_hash=document_hash,
    )

    # Optional export
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_stem = Path(document_name).stem

        raw_chunks = [chunk.export_json_dict() for chunk in chunks]
        json_path = output_dir / f"{doc_stem}_chunks.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(raw_chunks, f, indent=2)

        meta_path = output_dir / f"{doc_stem}_chunks_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in chunk_metadatas], f, indent=2)

        _log.info("Exported chunk files to %s", output_dir)

    return chunk_metadatas, text_lookup, pic_table_lookup
