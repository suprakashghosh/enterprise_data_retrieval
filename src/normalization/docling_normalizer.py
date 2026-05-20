"""
Normalise raw Docling output into the project's internal ``ElementSchema``
objects and build an in-memory element registry.

Public API
----------
::

    from src.normalization import (
        DOCLING_TYPE_TO_INTERNAL_TYPE,
        ElementRegistry,
        normalize_document,
        preserve_proximity,
    )

The module is **fully defensive** — it supports Docling objects, exported
dicts, and fake objects.  No real ``docling`` package is required at
import time or runtime.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from src.schemas import (
    BoundingBox,
    CaptionSchema,
    ChartSchema,
    DocumentSchema,
    ElementMetadata,
    ElementSchema,
    FooterSchema,
    FootnoteSchema,
    FormulaSchema,
    GraphSchema,
    HeaderSchema,
    ImageSchema,
    ListBlockSchema,
    PageSchema,
    RelationshipSchema,
    SectionHeaderSchema,
    Size,
    TableSchema,
    TextBlockSchema,
    make_element_id,
    make_relationship_id,
)

logger = logging.getLogger(__name__)

# ===================================================================
#  Public constants
# ===================================================================

DOCLING_TYPE_TO_INTERNAL_TYPE: Dict[str, str] = {
    "text": "text_block",
    "paragraph": "text_block",
    "textblock": "text_block",
    "table": "table",
    "picture": "image",
    "image": "image",
    "figure": "image",
    "chart": "chart",
    "graph": "graph",
    "diagram": "graph",
    "formula": "formula",
    "equation": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "header": "header",
    "page_header": "header",
    "footer": "footer",
    "page_footer": "footer",
    "list": "list_block",
    "list_item": "list_block",
    "title": "section_header",
    "section_header": "section_header",
    "heading": "section_header",
}
"""Mapping from Docling type/label strings to internal ``element_type`` values."""

# Schema classes keyed by internal element type.
_INTERNAL_TYPE_TO_SCHEMA: Dict[str, Any] = {
    "text_block": TextBlockSchema,
    "table": TableSchema,
    "image": ImageSchema,
    "chart": ChartSchema,
    "graph": GraphSchema,
    "formula": FormulaSchema,
    "caption": CaptionSchema,
    "footnote": FootnoteSchema,
    "header": HeaderSchema,
    "footer": FooterSchema,
    "list_block": ListBlockSchema,
    "section_header": SectionHeaderSchema,
}

# Visual parent types — elements that can own a caption.
_CAPTION_PARENT_TYPES = frozenset({"table", "image", "chart", "graph", "formula"})

# Fields to try when looking for a value on an object or dict.
_PAGE_NUM_FIELDS = ("page_no", "page_num", "page")
_TYPE_FIELDS = ("type", "label", "kind")
_ORDER_FIELDS = ("reading_order", "order", "index")

# ===================================================================
#  Internal helpers  (defensive attribute / dict access)
# ===================================================================


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    """Get an attribute or dict key from *obj* by trying *names* in order.

    Returns the first non-``None`` value found, or *default* if none match.
    """
    for name in names:
        if obj is None:
            continue
        if isinstance(obj, dict):
            try:
                val = obj[name]
                if val is not None:
                    return val
            except (KeyError, TypeError, IndexError):
                continue
        else:
            try:
                val = getattr(obj, name)
                if val is not None:
                    return val
            except AttributeError:
                continue
    return default


# ===================================================================
#  Bounding-box helpers
# ===================================================================


def _normalize_bbox_tuple(
    bbox: Any,
) -> Optional[Tuple[float, float, float, float]]:
    """Normalise a bounding box to ``(left, top, right, bottom)``.

    Supports:
    - ``BoundingBox`` model / object with ``left/top/right/bottom``
    - Object or dict with ``l/t/r/b``
    - Dict with ``left/top/right/bottom``
    - Tuple or list ``[left, top, right, bottom]``
    - Nested tuple/list ``[(left, top, right, bottom)]``
    """
    if bbox is None:
        return None

    # tuple/list [left, top, right, bottom]
    if isinstance(bbox, (tuple, list)):
        if len(bbox) == 4:
            try:
                return (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except (TypeError, ValueError):
                return None
        # Possibly a nested structure — try the first element
        if len(bbox) > 0 and isinstance(bbox[0], (tuple, list)):
            return _normalize_bbox_tuple(bbox[0])
        return None

    # dict
    if isinstance(bbox, dict):
        # left/top/right/bottom
        if all(k in bbox for k in ("left", "top", "right", "bottom")):
            try:
                return (
                    float(bbox["left"]),
                    float(bbox["top"]),
                    float(bbox["right"]),
                    float(bbox["bottom"]),
                )
            except (TypeError, ValueError):
                return None
        # l/t/r/b
        if all(k in bbox for k in ("l", "t", "r", "b")):
            try:
                return (
                    float(bbox["l"]),
                    float(bbox["t"]),
                    float(bbox["r"]),
                    float(bbox["b"]),
                )
            except (TypeError, ValueError):
                return None
        return None

    # object with left/top/right/bottom
    if all(hasattr(bbox, k) for k in ("left", "top", "right", "bottom")):
        try:
            return (
                float(bbox.left),
                float(bbox.top),
                float(bbox.right),
                float(bbox.bottom),
            )
        except (TypeError, ValueError):
            return None

    # object with l/t/r/b
    if all(hasattr(bbox, k) for k in ("l", "t", "r", "b")):
        try:
            return (
                float(bbox.l),
                float(bbox.t),
                float(bbox.r),
                float(bbox.b),
            )
        except (TypeError, ValueError):
            return None

    return None


def _extract_bbox(item: Any) -> Optional[Tuple[float, float, float, float]]:
    """Extract a normalised bounding-box tuple from an item.

    Tries ``item.bbox`` first, then ``item.rect`` / ``item.bounds``,
    then falls back to the first provenance entry's bbox.
    """
    bbox_raw = _get_field(item, "bbox")
    if bbox_raw is not None:
        result = _normalize_bbox_tuple(bbox_raw)
        if result is not None:
            return result

    bbox_raw = _get_field(item, "rect", "bounds")
    if bbox_raw is not None:
        result = _normalize_bbox_tuple(bbox_raw)
        if result is not None:
            return result

    # Try provenance
    prov = _get_field(item, "prov")
    if prov is not None and isinstance(prov, (list, tuple)) and len(prov) > 0:
        for prov_entry in prov:
            bbox_raw = _get_field(prov_entry, "bbox", "bbox_normalized")
            if bbox_raw is not None:
                result = _normalize_bbox_tuple(bbox_raw)
                if result is not None:
                    return result

    return None


def _to_normalized_bbox(
    bbox_tuple: Optional[Tuple[float, float, float, float]],
    page_dim: Optional[Tuple[float, float]],
) -> BoundingBox:
    """Convert a raw bbox tuple into a ``BoundingBox`` with normalized coordinates.

    When *page_dim* is available and positive the coordinates are divided
    by page width/height.  When the raw values already appear to be in
    [0, 1] they are kept as-is.  Otherwise a zero bbox is returned as a
    safe fallback.
    """
    if bbox_tuple is None:
        return BoundingBox(
            left=0.0,
            top=0.0,
            right=0.0,
            bottom=0.0,
            coord_system="normalized",
        )

    left, top, right, bottom = bbox_tuple

    # Normalise using page dimensions when available.
    if page_dim is not None:
        pw, ph = page_dim
        if pw is not None and ph is not None and pw > 0 and ph > 0:
            return BoundingBox(
                left=left / pw,
                top=top / ph,
                right=right / pw,
                bottom=bottom / ph,
                coord_system="normalized",
            )

    # If values are already in 0-1 range, treat as normalized.
    if all(0.0 <= v <= 1.0 for v in (left, top, right, bottom)):
        return BoundingBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            coord_system="normalized",
        )

    # Safe fallback — return a zero bbox (normalized).
    return BoundingBox(
        left=0.0,
        top=0.0,
        right=0.0,
        bottom=0.0,
        coord_system="normalized",
    )


# ===================================================================
#  Page dimension helpers
# ===================================================================


def _get_page_dimensions(page_obj: Any) -> Optional[Tuple[float, float]]:
    """Try to extract ``(width, height)`` from a page object/dict."""
    if page_obj is None:
        return None

    size = _get_field(page_obj, "size")
    if size is not None:
        w = _get_field(size, "width")
        h = _get_field(size, "height")
        if w is not None and h is not None:
            try:
                return (float(w), float(h))
            except (TypeError, ValueError):
                pass

    w = _get_field(page_obj, "width")
    h = _get_field(page_obj, "height")
    if w is not None and h is not None:
        try:
            return (float(w), float(h))
        except (TypeError, ValueError):
            pass

    return None


def _build_page_dim_map(
    dl_doc: Any,
) -> Dict[int, Optional[Tuple[float, float]]]:
    """Build a mapping of page number → ``(width, height)``."""
    dims: Dict[int, Optional[Tuple[float, float]]] = {}
    pages = _get_field(dl_doc, "pages")
    if pages is None:
        return dims

    if isinstance(pages, dict):
        for pnum, pobj in pages.items():
            pnum_i = int(pnum) if isinstance(pnum, (int, float)) else 0
            dims[pnum_i] = _get_page_dimensions(pobj)
    elif isinstance(pages, (list, tuple)):
        for idx, pobj in enumerate(pages, start=1):
            dims[idx] = _get_page_dimensions(pobj)

    return dims


# ===================================================================
#  Item type / text helpers
# ===================================================================


def _get_item_type(item: Any) -> Optional[str]:
    """Return the normalised type/label of an item, or ``None``."""
    for field in _TYPE_FIELDS:
        val = _get_field(item, field)
        if val is not None:
            if isinstance(val, str):
                return val.lower()
    return None


def _get_item_text(item: Any) -> str:
    """Return the text content of an item, or empty string."""
    text = _get_field(item, "text", "orig", "content")
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, (list, tuple)):
        return " ".join(str(t) for t in text if t)
    return str(text)


# ===================================================================
#  Reading-order helpers
# ===================================================================


def _get_item_order(item: Any) -> Optional[int]:
    """Return an explicit reading-order value from *item*, or ``None``.

    Tries ``reading_order``, ``order``, ``index`` fields, then falls
    back to the first provenance entry's order field.
    """
    order_val = _get_field(item, *_ORDER_FIELDS)
    if order_val is not None:
        try:
            return int(order_val)
        except (TypeError, ValueError):
            pass

    prov = _get_field(item, "prov")
    if isinstance(prov, (list, tuple)) and len(prov) > 0:
        prov_order = _get_field(prov[0], *_ORDER_FIELDS)
        if prov_order is not None:
            try:
                return int(prov_order)
            except (TypeError, ValueError):
                pass

    return None


def _spatial_sort_key(
    bbox_tuple: Optional[Tuple[float, float, float, float]],
) -> Tuple[int, float, float]:
    """Column-aware spatial sort key.

    Groups items into vertical columns (every ~0.3 of page width),
    then sorts top-to-bottom, left-to-right within each column.
    """
    if bbox_tuple is None:
        return (0, 0.0, 0.0)
    left, top, right, _bottom = bbox_tuple
    x_center = (left + right) / 2.0
    y_center = top  # Use top edge for vertical ordering
    # Column grouping: every 0.3 of the (normalized) page width.
    # If coordinates look absolute (> 1) use a different grouping.
    col_width = 0.3 if max(bbox_tuple) <= 1.0 else 200.0
    col = int(x_center / col_width) if col_width > 0 else 0
    return (col, y_center, x_center)


# ===================================================================
#  Item iteration  (supports objects, dicts, fake data)
# ===================================================================


def _iter_items(dl_doc: Any) -> List[Tuple[Any, int]]:
    """Iterate all content items in a Docling-like document.

    Returns a list of ``(item, page_num)`` tuples.
    """
    items: List[Tuple[Any, int]] = []
    seen_ids: set[int] = set()

    pages = _get_field(dl_doc, "pages")
    if pages is None:
        _collect_top_level_items(dl_doc, items, seen_ids)
        return items

    if isinstance(pages, dict):
        page_entries: Sequence = sorted(pages.items(), key=lambda kv: kv[0])
    elif isinstance(pages, (list, tuple)):
        page_entries = list(enumerate(pages, start=1))
    else:
        page_entries = []

    for page_key, page_obj in page_entries:
        if page_obj is None:
            continue

        page_num = _get_field(page_obj, *_PAGE_NUM_FIELDS)
        if page_num is None:
            page_num = int(page_key) if isinstance(page_key, (int, float)) else 0
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = 0

        _collect_page_items(page_obj, page_num, items, seen_ids)

    # Also collect top-level items not nested under pages
    _collect_top_level_items(dl_doc, items, seen_ids)

    return items


def _collect_page_items(
    page_obj: Any,
    page_num: int,
    items: List[Tuple[Any, int]],
    seen_ids: set[int],
) -> None:
    """Collect items from a single page object."""
    for container_field in ("items", "texts", "tables", "pictures", "figures"):
        container = _get_field(page_obj, container_field)
        if container is None:
            continue
        if isinstance(container, (list, tuple)):
            for item in container:
                if item is not None and id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    item_page = _get_field(item, *_PAGE_NUM_FIELDS)
                    effective_page = (
                        int(item_page) if item_page is not None else page_num
                    )
                    items.append((item, effective_page))

                    # Also collect children nested within this item
                    _collect_item_children(item, effective_page, items, seen_ids)


def _collect_item_children(
    item: Any,
    page_num: int,
    items: List[Tuple[Any, int]],
    seen_ids: set[int],
) -> None:
    """Collect child items nested inside a parent item."""
    for container_field in ("children", "items"):
        container = _get_field(item, container_field)
        if container is None:
            continue
        if isinstance(container, (list, tuple)):
            for child in container:
                if child is not None and id(child) not in seen_ids:
                    seen_ids.add(id(child))
                    child_page = _get_field(child, *_PAGE_NUM_FIELDS)
                    effective_page = (
                        int(child_page) if child_page is not None else page_num
                    )
                    items.append((child, effective_page))


def _collect_top_level_items(
    dl_doc: Any,
    items: List[Tuple[Any, int]],
    seen_ids: set[int],
) -> None:
    """Collect items from top-level containers."""
    for container_field in ("texts", "tables", "pictures", "figures", "items"):
        container = _get_field(dl_doc, container_field)
        if container is None:
            continue
        if isinstance(container, (list, tuple)):
            for item in container:
                if item is not None and id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    item_page = _get_field(item, *_PAGE_NUM_FIELDS)
                    page_num = int(item_page) if item_page is not None else 0
                    items.append((item, page_num))


# ===================================================================
#  Reading-order assignment per page
# ===================================================================


def _assign_reading_orders(
    page_items: Dict[
        int, List[Tuple[Any, int, Optional[Tuple[float, float, float, float]]]]
    ],
) -> Dict[int, List[Tuple[int, Any, int, Optional[Tuple[float, float, float, float]]]]]:
    """Assign sequential reading orders to items grouped by page.

    For each page, items with explicit ordering fields are used as-is;
    items without explicit order are sorted spatially (column-aware)
    and appended after the explicit items.  Then all items on the page
    are renumbered sequentially starting at 0.

    Returns ``{page_num: [(reading_order, item, page_num, bbox), ...]}``.
    """
    result: Dict[
        int, List[Tuple[int, Any, int, Optional[Tuple[float, float, float, float]]]]
    ] = {}

    for page_num in sorted(page_items):
        entries = page_items[page_num]

        # Separate items with and without explicit order
        with_order: List[Tuple[int, Any, int, Optional[Tuple]]] = []
        without_order: List[Tuple[Any, int, Optional[Tuple]]] = []

        for item, pn, bbox_t in entries:
            order = _get_item_order(item)
            if order is not None:
                with_order.append((order, item, pn, bbox_t))
            else:
                without_order.append((item, pn, bbox_t))

        # Sort each group
        with_order.sort(key=lambda x: x[0])
        without_order.sort(key=lambda x: _spatial_sort_key(x[2]))

        # If there are no explicit orders, use pure spatial sort
        if not with_order:
            # Re-sort everything spatially
            all_spatial: List[Tuple[int, Any, int, Optional[Tuple]]] = []
            for item, pn, bbox_t in without_order:
                all_spatial.append((0, item, pn, bbox_t))
            all_spatial.sort(key=lambda x: _spatial_sort_key(x[3]))
            result[page_num] = [
                (idx, item, pn, bbox_t)
                for idx, (_, item, pn, bbox_t) in enumerate(all_spatial)
            ]
        else:
            # Merge: explicit items first (by order), then spatial items
            merged: List[Tuple[Any, int, Optional[Tuple]]] = [
                (item, pn, bbox_t) for _, item, pn, bbox_t in with_order
            ]
            merged.extend(without_order)
            result[page_num] = [
                (idx, item, pn, bbox_t) for idx, (item, pn, bbox_t) in enumerate(merged)
            ]

    return result


# ===================================================================
#  Element creation
# ===================================================================


def _create_element(
    doc_id: uuid.UUID,
    item: Any,
    page_num: int,
    reading_order: int,
    internal_type: str,
    dl_type: Optional[str],
    bbox: BoundingBox,
) -> ElementSchema:
    """Create a frozen ``ElementSchema`` (or subclass) from a raw item.

    This is a factory method that dispatches to the correct schema class
    based on *internal_type* and populates type-specific fields where
    available.
    """
    content = _get_item_text(item)

    # Build metadata
    meta_custom: Dict[str, Any] = {}
    if dl_type is not None:
        meta_custom["original_type"] = dl_type

    # Capture confidence
    confidence = _get_field(item, "confidence", "confidence_score")
    if confidence is not None:
        try:
            meta_custom["confidence"] = float(confidence)
        except (TypeError, ValueError):
            pass

    # Capture font info
    font_info: Optional[Dict[str, Any]] = None
    font_family = _get_field(item, "font", "font_family", "font_name")
    font_size = _get_field(item, "font_size")
    if font_family is not None or font_size is not None:
        font_info = {}
        if font_family is not None:
            font_info["family"] = str(font_family)
        if font_size is not None:
            try:
                font_info["size"] = float(font_size)
            except (TypeError, ValueError):
                font_info["size"] = font_size

    # Capture source metadata
    source_metadata = _get_field(item, "source", "source_metadata")
    if source_metadata is not None:
        meta_custom["source_metadata"] = str(source_metadata)

    metadata = ElementMetadata(
        confidence_score=float(meta_custom.get("confidence", 0))
        if "confidence" in meta_custom
        else None,
        font_info=font_info,
        custom=meta_custom,
    )

    # Get the schema class
    schema_cls = _INTERNAL_TYPE_TO_SCHEMA.get(internal_type, TextBlockSchema)

    # Common kwargs for all element types
    common_kwargs: Dict[str, Any] = dict(
        element_id=make_element_id(doc_id, page_num, reading_order, internal_type),
        doc_id=doc_id,
        page_num=page_num,
        bbox=bbox,
        reading_order=reading_order,
        element_type=internal_type,
        content=content,
        metadata=metadata,
    )

    # Type-specific fields
    if internal_type == "table":
        # Extract table data
        markdown = _get_field(item, "markdown", "md", default="")
        html = _get_field(item, "html", default="")
        json_data_raw = _get_field(item, "json", "json_data", "data")
        json_data: Dict[str, Any] = {}
        if isinstance(json_data_raw, dict):
            json_data = json_data_raw
        elif isinstance(json_data_raw, (list, tuple)):
            json_data = {"rows": json_data_raw}

        rows = _get_field(item, "rows", "num_rows", "row_count")
        cols = _get_field(item, "cols", "num_cols", "col_count")
        try:
            row_count = int(rows) if rows is not None else 0
        except (TypeError, ValueError):
            row_count = 0
        try:
            col_count = int(cols) if cols is not None else 0
        except (TypeError, ValueError):
            col_count = 0

        headers_raw = _get_field(item, "headers", default=None)
        headers: List[str] = []
        if isinstance(headers_raw, (list, tuple)):
            headers = [str(h) for h in headers_raw]

        caption = _get_field(item, "caption", "caption_text", default="")

        return TableSchema(
            **common_kwargs,
            markdown=str(markdown) if markdown else "",
            html=str(html) if html else "",
            json_data=json_data,
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            summary=str(caption) if caption else "",
        )

    if internal_type == "image":
        asset_path = _get_field(item, "asset_path", "image_path", default=None)
        caption_text = _get_field(item, "caption", "caption_text", default=None)
        return ImageSchema(
            **common_kwargs,
            asset_path=str(asset_path) if asset_path else None,
            caption=str(caption_text) if caption_text else None,
        )

    if internal_type == "chart":
        asset_path = _get_field(item, "asset_path", default=None)
        caption_text = _get_field(item, "caption", "caption_text", default=None)
        return ChartSchema(
            **common_kwargs,
            asset_path=str(asset_path) if asset_path else None,
            caption=str(caption_text) if caption_text else None,
        )

    if internal_type == "graph":
        asset_path = _get_field(item, "asset_path", default=None)
        caption_text = _get_field(item, "caption", "caption_text", default=None)
        return GraphSchema(
            **common_kwargs,
            asset_path=str(asset_path) if asset_path else None,
            caption=str(caption_text) if caption_text else None,
        )

    if internal_type == "formula":
        latex = _get_field(item, "latex", "text", default="")
        text_approx = _get_field(item, "text_approximation", default="")
        formula_type = _get_field(item, "formula_type", default="display")
        if formula_type not in ("inline", "display"):
            formula_type = "display"
        return FormulaSchema(
            **common_kwargs,
            latex=str(latex) if isinstance(latex, str) else str(latex),
            text_approximation=str(text_approx) if text_approx else content,
            formula_type=formula_type,
        )

    if internal_type == "caption":
        return CaptionSchema(**common_kwargs)

    if internal_type == "footnote":
        footnote_id = _get_field(item, "footnote_id", default=None)
        return FootnoteSchema(
            **common_kwargs,
            footnote_id=str(footnote_id) if footnote_id else None,
        )

    if internal_type == "header":
        return HeaderSchema(**common_kwargs)

    if internal_type == "footer":
        return FooterSchema(**common_kwargs)

    if internal_type == "list_block":
        items_raw = _get_field(item, "items", default=None)
        list_items: List[str] = []
        if isinstance(items_raw, (list, tuple)):
            list_items = [str(it) for it in items_raw]
        ordered = _get_field(item, "ordered", default=False)
        if isinstance(ordered, str):
            ordered = ordered.lower() in ("true", "1", "yes")
        if not list_items and content:
            list_items = [content]
        return ListBlockSchema(
            **common_kwargs,
            items=list_items,
            ordered=bool(ordered),
        )

    if internal_type == "section_header":
        level = _get_field(item, "level", default=1)
        try:
            level = int(level) if level is not None else 1
        except (TypeError, ValueError):
            level = 1
        section_number = _get_field(item, "section_number", default=None)
        return SectionHeaderSchema(
            **common_kwargs,
            level=max(1, level),
            section_number=str(section_number) if section_number else None,
        )

    # Default: text_block
    if internal_type == "text_block":
        language = _get_field(item, "language", default=None)
        kwargs = dict(**common_kwargs)
        if language is not None:
            kwargs["language"] = str(language)
        return TextBlockSchema(**kwargs)

    # Fallback: generic element (shouldn't normally reach here)
    return schema_cls(**common_kwargs)


# ===================================================================
#  Caption linking
# ===================================================================


def _find_caption_parent(
    caption_elem: CaptionSchema,
    registry: ElementRegistry,
    threshold: float = 0.3,
) -> Optional[ElementSchema]:
    """Find the most likely parent element for a caption.

    Uses spatial heuristic: among visual elements on the same page,
    pick the nearest one whose bbox is above the caption (captions
    typically appear below their parent).  Returns ``None`` when no
    suitable parent is found within *threshold* normalized distance.
    """
    candidates = registry.get_by_page(caption_elem.page_num)
    caption_center_y = (caption_elem.bbox.top + caption_elem.bbox.bottom) / 2.0

    best: Optional[ElementSchema] = None
    best_dist = float("inf")

    for elem in candidates:
        if elem.element_id == caption_elem.element_id:
            continue
        if elem.element_type not in _CAPTION_PARENT_TYPES:
            continue

        elem_center_y = (elem.bbox.top + elem.bbox.bottom) / 2.0
        # Prefer parents above the caption
        if elem_center_y > caption_center_y:
            continue  # Parent should be above caption

        dist = abs(elem_center_y - caption_center_y)
        if dist < best_dist:
            best_dist = dist
            best = elem

    if best is not None and best_dist <= threshold:
        return best
    return None


def _link_captions(
    elements: Dict[str, ElementSchema],
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Link caption elements to their parent visual elements.

    Returns a list of ``has_caption`` and ``describes`` relationships.
    """
    relationships: List[RelationshipSchema] = []
    seen_pairs: set[Tuple[uuid.UUID, uuid.UUID]] = set()

    for elem in elements.values():
        if not isinstance(elem, CaptionSchema):
            continue

        # Try explicit parent_element_id
        parent_id: Optional[uuid.UUID] = None
        if elem.parent_element_id is not None:
            parent_id = elem.parent_element_id

        # Try explicit reference fields on the original item
        # These are stored in metadata.custom by _create_element
        custom = elem.metadata.custom if elem.metadata else {}
        for ref_field in ("caption_of", "parent_ref", "parent", "ref"):
            ref_val = custom.get(ref_field)
            if ref_val is not None:
                try:
                    if isinstance(ref_val, uuid.UUID):
                        candidate_id = ref_val
                    elif isinstance(ref_val, str):
                        candidate_id = uuid.UUID(ref_val)
                    else:
                        continue
                    candidate = registry.get(candidate_id)
                    if (
                        candidate is not None
                        and candidate.element_type in _CAPTION_PARENT_TYPES
                    ):
                        parent_id = candidate_id
                        break
                except (ValueError, AttributeError):
                    continue

        # Spatial heuristic fallback
        if parent_id is None:
            parent = _find_caption_parent(elem, registry)
            if parent is not None:
                parent_id = parent.element_id

        if parent_id is not None:
            pair = (parent_id, elem.element_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                # has_caption (parent → caption)
                relationships.append(
                    RelationshipSchema(
                        relationship_id=make_relationship_id(
                            parent_id, elem.element_id, "has_caption"
                        ),
                        source_id=parent_id,
                        target_id=elem.element_id,
                        relationship_type="has_caption",
                        metadata={"page_num": elem.page_num},
                        weight=1.0,
                    )
                )
                # describes (caption → parent)
                relationships.append(
                    RelationshipSchema(
                        relationship_id=make_relationship_id(
                            elem.element_id, parent_id, "describes"
                        ),
                        source_id=elem.element_id,
                        target_id=parent_id,
                        relationship_type="describes",
                        metadata={"page_num": elem.page_num},
                        weight=1.0,
                    )
                )

    return relationships


# ===================================================================
#  ElementRegistry
# ===================================================================


class ElementRegistry:
    """In-memory registry for looking up elements by ID, page, or type.

    Provides O(1) lookup by element ID and stable sorted iteration in
    reading order (by ``page_num``, ``reading_order``, ``bbox.top``,
    ``bbox.left``).
    """

    def __init__(self) -> None:
        self._by_id: Dict[uuid.UUID, ElementSchema] = {}
        self._by_page: Dict[int, List[ElementSchema]] = {}
        self._by_type: Dict[str, List[ElementSchema]] = {}

    def add(self, element: ElementSchema) -> None:
        """Register *element* in the registry."""
        self._by_id[element.element_id] = element
        self._by_page.setdefault(element.page_num, []).append(element)
        self._by_type.setdefault(element.element_type, []).append(element)

    def get(self, element_id: uuid.UUID) -> Optional[ElementSchema]:
        """Look up an element by its UUID.  Returns ``None`` if not found."""
        return self._by_id.get(element_id)

    def get_by_page(self, page_num: int) -> List[ElementSchema]:
        """Return all elements on *page_num* (copy)."""
        return list(self._by_page.get(page_num, []))

    def get_by_type(self, element_type: str) -> List[ElementSchema]:
        """Return all elements of *element_type* (copy)."""
        return list(self._by_type.get(element_type, []))

    def iter_in_reading_order(self) -> Iterator[ElementSchema]:
        """Yield elements sorted by ``(page_num, reading_order, bbox.top, bbox.left)``."""
        sorted_elems = sorted(
            self._by_id.values(),
            key=lambda e: (e.page_num, e.reading_order, e.bbox.top, e.bbox.left),
        )
        return iter(sorted_elems)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, element_id: uuid.UUID) -> bool:
        return element_id in self._by_id


# ===================================================================
#  Proximity computation
# ===================================================================


def _bbox_center_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Euclidean distance between centres of two normalized bounding boxes."""
    c1_x = (bbox1.left + bbox1.right) / 2.0
    c1_y = (bbox1.top + bbox1.bottom) / 2.0
    c2_x = (bbox2.left + bbox2.right) / 2.0
    c2_y = (bbox2.top + bbox2.bottom) / 2.0
    return math.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)


def preserve_proximity(
    elements: List[ElementSchema],
    threshold: float = 0.08,
) -> List[RelationshipSchema]:
    """Compute ``nearby`` relationships for elements on the same page.

    Two elements on the same page are considered *nearby* when the
    Euclidean distance between their bounding-box centres is ≤
    *threshold* (in normalized coordinates).

    Args:
        elements: All elements to consider.
        threshold: Maximum normalized centre-to-centre distance.

    Returns:
        A list of ``nearby`` :class:`RelationshipSchema` objects.
        Does **not** mutate any input.
    """
    # Group by page
    page_groups: Dict[int, List[ElementSchema]] = {}
    for elem in elements:
        page_groups.setdefault(elem.page_num, []).append(elem)

    relationships: List[RelationshipSchema] = []
    seen_pairs: set[Tuple[uuid.UUID, uuid.UUID]] = set()

    for page_num in sorted(page_groups):
        page_elems = page_groups[page_num]
        for i, a in enumerate(page_elems):
            for b in page_elems[i + 1 :]:
                if a.element_id == b.element_id:
                    continue
                dist = _bbox_center_distance(a.bbox, b.bbox)
                if dist <= threshold:
                    pair_key = (a.element_id, b.element_id)
                    rev_key = (b.element_id, a.element_id)
                    if pair_key not in seen_pairs and rev_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        relationships.append(
                            RelationshipSchema(
                                relationship_id=make_relationship_id(
                                    a.element_id, b.element_id, "nearby"
                                ),
                                source_id=a.element_id,
                                target_id=b.element_id,
                                relationship_type="nearby",
                                metadata={
                                    "distance": round(dist, 6),
                                    "page_num": page_num,
                                },
                                weight=round(1.0 - dist, 6),
                            )
                        )

    return relationships


# ===================================================================
#  Main normalization function
# ===================================================================


def normalize_document(
    doc: DocumentSchema,
    dl_doc: Any,
    proximity_threshold: float = 0.08,
) -> DocumentSchema:
    """Normalize a Docling document into the internal ``DocumentSchema``.

    Discovers pages and elements from *dl_doc* (which may be a Docling
    object, an exported dict, or a fake/test double), maps each item to
    the correct ``ElementSchema`` subclass, preserves bounding boxes
    (normalised to 0-1), reading orders, captions, and proximity
    relationships.

    Args:
        doc: The ``DocumentSchema`` from ingestion (must have ``doc_id``
            and other metadata populated).
        dl_doc: A Docling-like document (object or exported dict).
        proximity_threshold: Max normalised centre-to-centre distance
            for ``nearby`` relationships.

    Returns:
        A new ``DocumentSchema`` with populated ``pages``, ``elements``,
        and ``relationships`` fields, preserving all existing document
        metadata.
    """
    doc_id = doc.doc_id

    # ------------------------------------------------------------------
    #  1. Build page dimension map
    # ------------------------------------------------------------------
    page_dim_map = _build_page_dim_map(dl_doc)

    # ------------------------------------------------------------------
    #  2. Iterate all items
    # ------------------------------------------------------------------
    raw_items = _iter_items(dl_doc)

    # Group items by page with their bboxes
    page_items: Dict[
        int, List[Tuple[Any, int, Optional[Tuple[float, float, float, float]]]]
    ] = {}
    for item, page_num in raw_items:
        page_items.setdefault(page_num, []).append(
            (item, page_num, _extract_bbox(item))
        )

    # ------------------------------------------------------------------
    #  3. Assign reading orders per page
    # ------------------------------------------------------------------
    ordered_items = _assign_reading_orders(page_items)

    # ------------------------------------------------------------------
    #  4. Create elements
    # ------------------------------------------------------------------
    elements: Dict[str, ElementSchema] = {}
    page_elem_ids: Dict[int, List[uuid.UUID]] = {}

    for page_num in sorted(ordered_items):
        page_dim = page_dim_map.get(page_num)
        for reading_order, item, pn, bbox_tuple in ordered_items[page_num]:
            # Determine type
            dl_type = _get_item_type(item)
            internal_type = DOCLING_TYPE_TO_INTERNAL_TYPE.get(
                dl_type or "", "text_block"
            )

            # Normalize bbox
            bbox = _to_normalized_bbox(bbox_tuple, page_dim)

            # Create element
            try:
                element = _create_element(
                    doc_id=doc_id,
                    item=item,
                    page_num=pn,
                    reading_order=reading_order,
                    internal_type=internal_type,
                    dl_type=dl_type,
                    bbox=bbox,
                )
            except Exception:
                logger.warning(
                    "Failed to create element on page %s, type=%s: skipping",
                    pn,
                    internal_type,
                    exc_info=True,
                )
                continue

            elem_key = str(element.element_id)
            elements[elem_key] = element
            page_elem_ids.setdefault(pn, []).append(element.element_id)

    # Attach caption reference hints from items to caption metadata
    _attach_caption_references(raw_items, elements)

    # ------------------------------------------------------------------
    #  5. Create PageSchema objects
    # ------------------------------------------------------------------
    pages: List[PageSchema] = []
    all_page_nums = set(page_elem_ids.keys())
    # Also include pages from the dimension map that have no elements
    all_page_nums.update(page_dim_map.keys())

    for pnum in sorted(all_page_nums):
        dim = page_dim_map.get(pnum)
        size: Optional[Size] = None
        if dim is not None and dim[0] > 0 and dim[1] > 0:
            size = Size(width=dim[0], height=dim[1])
        pages.append(
            PageSchema(
                page_num=pnum,
                size=size,
                element_ids=page_elem_ids.get(pnum, []),
            )
        )

    # ------------------------------------------------------------------
    #  6. Build registry
    # ------------------------------------------------------------------
    registry = ElementRegistry()
    for elem in elements.values():
        registry.add(elem)

    # ------------------------------------------------------------------
    #  7. Link captions
    # ------------------------------------------------------------------
    caption_relationships = _link_captions(elements, registry)

    # ------------------------------------------------------------------
    #  8. Compute proximity
    # ------------------------------------------------------------------
    all_element_list = list(elements.values())
    proximity_relationships = preserve_proximity(all_element_list, proximity_threshold)

    # ------------------------------------------------------------------
    #  9. Return updated DocumentSchema
    # ------------------------------------------------------------------
    all_relationships: List[RelationshipSchema] = []
    seen_rel_ids: set[uuid.UUID] = set()

    for rel in (
        list(doc.relationships) + caption_relationships + proximity_relationships
    ):
        if rel.relationship_id not in seen_rel_ids:
            seen_rel_ids.add(rel.relationship_id)
            all_relationships.append(rel)

    return doc.model_copy(
        update={
            "pages": pages,
            "elements": elements,
            "relationships": all_relationships,
        }
    )


def _attach_caption_references(
    raw_items: List[Tuple[Any, int]],
    elements: Dict[str, ElementSchema],
) -> None:
    """Scan raw items for caption-of references and attach them to element metadata.

    Caption elements get their ``parent_element_id`` set, and parent
    elements (tables, figures, etc.) get caption references stored in
    their metadata.custom.
    """
    # Build a mapping from item id -> element_id for caption elements
    item_to_elem: Dict[int, uuid.UUID] = {}
    for item, _page_num in raw_items:
        # We need to match elements to their raw items.
        # This is done by looking at the element's original_type in metadata.
        pass

    # For each raw item, check if it has a "caption_of" reference
    for item, page_num in raw_items:
        caption_of = _get_field(item, "caption_of", "ref", "parent_ref", "parent")
        if caption_of is None:
            continue

        # Try to resolve the referenced element
        try:
            if isinstance(caption_of, uuid.UUID):
                ref_id = caption_of
            elif isinstance(caption_of, str):
                ref_id = uuid.UUID(caption_of)
            else:
                ref_id = getattr(caption_of, "element_id", None) or getattr(
                    caption_of, "uuid", None
                )
                if ref_id is not None and not isinstance(ref_id, uuid.UUID):
                    ref_id = uuid.UUID(str(ref_id))
        except (ValueError, TypeError, AttributeError):
            continue

        if ref_id is None:
            continue

        # Check if this item itself became a caption element
        item_elem_id = None
        for elem_key, elem in elements.items():
            if elem.page_num == page_num and isinstance(elem, CaptionSchema):
                # Rough match: same page, same type, similar content
                item_text = _get_item_text(item)
                if item_text and item_text.strip() == elem.content.strip():
                    item_elem_id = elem.element_id
                    break

        if item_elem_id is not None:
            # Update the caption element's parent_element_id
            # Since models are frozen, we need to create a new one
            old_elem = elements[str(item_elem_id)]
            if isinstance(old_elem, CaptionSchema):
                new_elem = old_elem.model_copy(update={"parent_element_id": ref_id})
                elements[str(item_elem_id)] = new_elem

        # Store the reference in the parent's metadata custom
        parent_elem = next(
            (e for e in elements.values() if e.element_id == ref_id),
            None,
        )
        if parent_elem is not None:
            # Update parent metadata with caption reference
            new_custom = dict(parent_elem.metadata.custom)
            new_custom["has_caption_id"] = str(item_elem_id or "")
            new_meta = parent_elem.metadata.model_copy(update={"custom": new_custom})
            new_parent = parent_elem.model_copy(update={"metadata": new_meta})
            elements[str(ref_id)] = new_parent
