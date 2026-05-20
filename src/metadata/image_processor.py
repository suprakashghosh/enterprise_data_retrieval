"""
Process images, charts, and graphs — asset saving, visual type
classification, metadata preparation, and relationship generation.

Public API
----------
::

    from src.metadata import (
        save_visual_asset,
        classify_visual_type,
        prepare_visual_metadata,
        process_images,
    )

The module uses duck typing throughout — no real Docling, PIL, fitz, or
pdf2image dependency is required at import time.  Image objects must
provide ``.save()`` and optionally ``.thumbnail()``, ``.size``, ``.mode``.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from src.schemas import (
    BoundingBox,
    CaptionSchema,
    ChartSchema,
    DocumentSchema,
    ElementSchema,
    GraphSchema,
    ImageSchema,
    RelationshipSchema,
    make_relationship_id,
)
from src.utils.config import PipelineSettings
from src.utils.file_io import ensure_dir

if TYPE_CHECKING:
    from src.normalization import ElementRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THUMBNAIL_MAX_SIZE: int = 256
"""Maximum width/height for thumbnail images."""

_VISUAL_ELEMENT_TYPES = frozenset({"image", "chart", "graph"})
"""Element types that this processor handles."""

_VISUAL_SCHEMA_TYPES = (ImageSchema, ChartSchema, GraphSchema)
"""Schema classes that this processor handles."""

_CAPTION_PROXIMITY_THRESHOLD: float = 0.15
"""Max normalised centre-to-centre distance for caption proximity."""

_DESCRIBES_PROXIMITY_THRESHOLD: float = 0.2
"""Max normalised centre-to-centre distance for 'describes' text."""

_NEARBY_THRESHOLD: float = 0.15
"""Max normalised centre-to-centre distance for 'nearby' relationship."""

_VISUAL_MENTION_PATTERN = re.compile(
    r"\b(?:figure|fig\.|chart|graph|diagram|illustration|image|plot|screenshot)\b",
    re.IGNORECASE,
)
"""Pattern to detect text that likely describes a visual element."""


# ---------------------------------------------------------------------------
# Internal helpers — image source discovery (duck-typed)
# ---------------------------------------------------------------------------


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    """Get an attribute or dict key from *obj* by trying *names* in order."""
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


def _has_save(img: Any) -> bool:
    """Check if *img* looks like a savable image (duck typing)."""
    return img is not None and hasattr(img, "save") and callable(img.save)


def _has_thumbnail(img: Any) -> bool:
    """Check if *img* has a ``thumbnail`` method (PIL-like)."""
    return hasattr(img, "thumbnail") and callable(img.thumbnail)


def _has_size(img: Any) -> bool:
    """Check if *img* has a ``size`` attribute (PIL-like ``(width, height)``)."""
    return hasattr(img, "size") and img.size is not None


def _has_mode(img: Any) -> bool:
    """Check if *img* has a ``mode`` attribute (PIL-like ``"RGB"``, etc.)."""
    return hasattr(img, "mode") and img.mode is not None


def _try_get_image_from_element(element: Any) -> Any:
    """Try to obtain an image object directly from *element*.

    Priority:
    a. ``element.image`` (direct PIL/Image object)
    b. ``element.pil_image``
    c. ``element.get_image(dl_doc)`` — but we need dl_doc for that.

    Returns ``None`` if none found.
    """
    # Path A: direct .image attribute
    img = _get_field(element, "image")
    if _has_save(img):
        return img

    # Path B: .pil_image attribute
    img = _get_field(element, "pil_image")
    if _has_save(img):
        return img

    return None


def _try_get_image_from_dl_doc(dl_doc: Any, element: ElementSchema) -> Any:
    """Try to obtain an image for *element* from *dl_doc*.

    Strategies:
    a. ``element.get_image(dl_doc)``
    b. Search ``dl_doc.pages[*].pictures`` for a matching item that has
       a ``.get_image(dl_doc)``, ``.image``, or ``.pil_image``.

    Returns ``None`` if none found.
    """
    if dl_doc is None:
        return None

    # Path A: element-level get_image method
    get_image = _get_field(element, "get_image")
    if callable(get_image):
        try:
            img = get_image(dl_doc)
            if _has_save(img):
                return img
        except (AttributeError, TypeError, ImportError):
            pass

    # Path B: search page pictures for a matching visual item
    pages = _get_field(dl_doc, "pages")
    if pages is None:
        return None

    page_obj = None
    if isinstance(pages, dict):
        page_obj = pages.get(element.page_num)
    elif isinstance(pages, (list, tuple)):
        idx = element.page_num - 1
        if 0 <= idx < len(pages):
            page_obj = pages[idx]

    if page_obj is None:
        return None

    # Collect pictures from the page
    picture_candidates: List[Any] = []
    for container_field in ("pictures", "figures", "images"):
        container = _get_field(page_obj, container_field)
        if isinstance(container, (list, tuple)):
            picture_candidates.extend(container)
        elif container is not None:
            picture_candidates.append(container)

    # Also try top-level containers
    if not picture_candidates:
        for container_field in ("pictures", "figures", "images"):
            container = _get_field(dl_doc, container_field)
            if isinstance(container, (list, tuple)):
                # Try to filter by page number
                for item in container:
                    item_page = _get_field(item, "page_num", "page", default=None)
                    if item_page == element.page_num or item_page is None:
                        picture_candidates.append(item)

    if not picture_candidates:
        return None

    # Try each candidate's get_image method
    for pic in picture_candidates:
        # Path B1: pic.get_image(dl_doc)
        pic_get_image = _get_field(pic, "get_image")
        if callable(pic_get_image):
            try:
                img = pic_get_image(dl_doc)
                if _has_save(img):
                    return img
            except (AttributeError, TypeError, ImportError):
                pass

        # Path B2: pic.image
        img = _get_field(pic, "image")
        if _has_save(img):
            return img

        # Path B3: pic.pil_image
        img = _get_field(pic, "pil_image")
        if _has_save(img):
            return img

    return None


def _obtain_image(
    element: ElementSchema,
    dl_doc: Any = None,
) -> Any:
    """Obtain an image bitmap for *element* through any available channel.

    Priority order:
    1. Direct image object on the element itself (``.image``, ``.pil_image``).
    2. ``element.get_image(dl_doc)`` or matching Docling picture item.
    3. Graceful return ``None`` if no source is available.

    Returns a duck-typed image object (needs ``.save()``), or ``None``.
    """
    # Priority 1: direct element image attributes
    img = _try_get_image_from_element(element)
    if img is not None:
        return img

    # Priority 2: dl_doc-based lookup
    if dl_doc is not None:
        img = _try_get_image_from_dl_doc(dl_doc, element)
        if img is not None:
            return img

    return None


# ---------------------------------------------------------------------------
# Internal helpers — image saving
# ---------------------------------------------------------------------------


def _save_image(
    img: Any,
    asset_path: Path,
    thumbnail_path: Path,
) -> None:
    """Save *img* as a PNG asset and a thumbnail.

    Both directories are created as needed.  If *img* does not support
    ``.thumbnail()`` (e.g. a simple fake object), the full-size image is
    saved again as the thumbnail.
    """
    ensure_dir(asset_path.parent)
    ensure_dir(thumbnail_path.parent)

    # Save full-size PNG
    img.save(str(asset_path))

    # Save thumbnail
    if _has_thumbnail(img):
        # PIL Image.thumbnail() modifies in-place, so we need a copy
        try:
            thumb = img.copy()
        except AttributeError:
            thumb = img
        try:
            thumb.thumbnail((_THUMBNAIL_MAX_SIZE, _THUMBNAIL_MAX_SIZE))
            thumb.save(str(thumbnail_path))
        except (AttributeError, OSError, TypeError):
            # Fallback: save the full image as the thumbnail
            img.save(str(thumbnail_path))
    else:
        # No thumbnail support — save the same image as thumbnail
        img.save(str(thumbnail_path))


# ---------------------------------------------------------------------------
#  Internal helpers — image properties extraction
# ---------------------------------------------------------------------------


def _compute_image_properties(
    img: Any, file_size: Optional[int] = None
) -> Dict[str, Any]:
    """Compute basic image properties from a duck-typed image object.

    Returns a dict with keys: ``width``, ``height``, ``aspect_ratio``,
    ``file_size`` (bytes), ``color_mode``.  Missing values are set to
    ``None``.
    """
    props: Dict[str, Any] = {
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "file_size": file_size,
        "color_mode": None,
    }

    if _has_size(img):
        try:
            w, h = img.size
            props["width"] = int(w)
            props["height"] = int(h)
            if w and h:
                props["aspect_ratio"] = round(w / h, 4)
        except (TypeError, ValueError):
            pass

    if _has_mode(img):
        try:
            props["color_mode"] = str(img.mode)
        except (TypeError, ValueError):
            pass

    return props


# ---------------------------------------------------------------------------
#  Internal helpers — visual type classification
# ---------------------------------------------------------------------------


def _classify_chart(element: ChartSchema, image_data: Any = None) -> str:
    """Classify a chart-type element."""
    # Default for explicit ChartSchema
    return "chart"


def _classify_graph(element: GraphSchema, image_data: Any = None) -> str:
    """Classify a graph-type element."""
    return "graph"


def _classify_image(
    element: ImageSchema,
    image_data: Any = None,
) -> str:
    """Classify an image element using deterministic heuristics.

    Heuristics (in priority order):
    - Caption/content mentions help decide the type.
    - Aspect ratio (from image_data if available) can distinguish
      logos/screenshots from photographs.
    """
    text = (element.caption or "") + " " + (element.content or "")
    text_lower = text.lower()

    # --- Explicit mention in caption/content ---
    if re.search(r"\bchart\b", text_lower):
        return "chart"
    if re.search(r"\bgraph\b", text_lower):
        return "graph"
    if re.search(r"\bdiagram\b", text_lower):
        return "diagram"
    if re.search(r"\bflowchart\b", text_lower):
        return "diagram"
    if re.search(r"\bphotograph\b", text_lower) or re.search(r"\bphoto\b", text_lower):
        return "photograph"
    if re.search(r"\billustration\b", text_lower):
        return "illustration"
    if re.search(r"\blogo\b", text_lower):
        return "logo"
    if re.search(r"\bscreenshot\b", text_lower):
        return "screenshot"

    # --- Aspect-ratio heuristics ---
    aspect = _get_aspect_ratio(element, image_data)
    if aspect is not None:
        # Very wide or tall: likely decorative / banner
        if aspect > 4.0 or aspect < 0.25:
            return "decorative"
        # Nearly square: likely logo or icon
        if 0.8 <= aspect <= 1.2:
            # Check size: if very small area, it's probably a logo
            area = _get_area(element, image_data)
            if area is not None and area < 10000:
                return "logo"
            return "illustration"

    # --- Size heuristics ---
    area = _get_area(element, image_data)
    if area is not None:
        # Tiny images are often logos or decorative
        if area < 5000:
            return "logo"
        # Very large images are often photographs
        if area > 500000:
            return "photograph"

    return "unclassified"


def _get_aspect_ratio(element: ElementSchema, image_data: Any) -> Optional[float]:
    """Get aspect ratio from image data or element bounding box."""
    if image_data is not None and _has_size(image_data):
        try:
            w, h = image_data.size
            if w and h:
                return w / h
        except (TypeError, ValueError):
            pass

    # Fall back to bounding box aspect ratio
    bbox = element.bbox
    if bbox is not None:
        w = bbox.right - bbox.left
        h = bbox.bottom - bbox.top
        if w and h:
            return w / h

    return None


def _get_area(element: ElementSchema, image_data: Any) -> Optional[float]:
    """Get pixel area from image data or bounding box."""
    if image_data is not None and _has_size(image_data):
        try:
            w, h = image_data.size
            if w and h:
                return w * h
        except (TypeError, ValueError):
            pass

    # Fallback using bounding box in normalized coords
    bbox = element.bbox
    if bbox is not None:
        w = bbox.right - bbox.left
        h = bbox.bottom - bbox.top
        if w and h:
            return w * h * 1_000_000  # Rough pixel estimate

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_visual_asset(
    element: Union[ImageSchema, ChartSchema, GraphSchema],
    dl_doc: Any = None,
    doc: DocumentSchema = None,
    settings: Optional[PipelineSettings] = None,
) -> Union[ImageSchema, ChartSchema, GraphSchema]:
    """Save a visual element's image as a PNG asset and thumbnail.

    Tries to obtain an image bitmap via the priority order described in
    :func:`_obtain_image`.  When no image source is found the element is
    returned unchanged.

    The PNG is saved to ``{settings.raw_dir}/{doc.doc_id}/assets/{element.element_id}.png``
    and the thumbnail to ``.../thumb_{element.element_id}.png``.

    Args:
        element: The visual element (``ImageSchema``, ``ChartSchema``, or
            ``GraphSchema``).
        dl_doc: Optional Docling document or dict for picture lookup.
        doc: The parent document (needed for the output path).
        settings: Pipeline configuration.  Defaults when ``None``.

    Returns:
        An updated copy of *element* with ``asset_path`` and
        ``thumbnail_path`` populated, or the original element unchanged
        when no image source is available.
    """
    settings = settings or PipelineSettings()
    img = _obtain_image(element, dl_doc=dl_doc)

    if img is None:
        logger.debug(
            "No image source found for element %s (type=%s) on page %s",
            element.element_id,
            element.element_type,
            element.page_num,
        )
        return element

    # Build paths
    doc_id_str = str(doc.doc_id) if doc is not None else str(element.doc_id)
    assets_dir = settings.raw_dir / doc_id_str / "assets"
    asset_filename = f"{element.element_id}.png"
    thumb_filename = f"thumb_{element.element_id}.png"

    asset_path = assets_dir / asset_filename
    thumb_path = assets_dir / thumb_filename

    # Save
    _save_image(img, asset_path, thumb_path)

    logger.debug(
        "Saved visual asset for %s: %s (thumb: %s)",
        element.element_id,
        asset_path,
        thumb_path,
    )

    return element.model_copy(
        update={
            "asset_path": str(asset_path),
            "thumbnail_path": str(thumb_path),
        }
    )


def classify_visual_type(
    element: ElementSchema,
    image_data: Any = None,
) -> str:
    """Classify a visual element's type using deterministic heuristics.

    Supported output values:
    ``chart``, ``graph``, ``diagram``, ``photograph``, ``illustration``,
    ``logo``, ``screenshot``, ``decorative``, ``other``, ``unclassified``.

    ``ChartSchema`` elements default to ``"chart"`` and ``GraphSchema``
    elements default to ``"graph"``.  ``ImageSchema`` elements use
    caption/content text, aspect ratio, and size heuristics.

    Args:
        element: The visual element to classify.
        image_data: Optional duck-typed image object for size/ratio-based
            heuristics.

    Returns:
        A classification string.
    """
    if isinstance(element, ChartSchema):
        return _classify_chart(element, image_data)

    if isinstance(element, GraphSchema):
        return _classify_graph(element, image_data)

    if isinstance(element, ImageSchema):
        return _classify_image(element, image_data)

    # Not a recognized visual type
    return "unclassified"


def _bbox_center_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Euclidean distance between centres of two normalized bounding boxes."""
    c1_x = (bbox1.left + bbox1.right) / 2.0
    c1_y = (bbox1.top + bbox1.bottom) / 2.0
    c2_x = (bbox2.left + bbox2.right) / 2.0
    c2_y = (bbox2.top + bbox2.bottom) / 2.0
    return ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5


def _extract_caption_text(element: ElementSchema) -> Optional[str]:
    """Extract caption text from a visual element.

    Checks ``element.caption`` for explicit caption text.
    """
    caption = _get_field(element, "caption", "caption_text", default=None)
    if caption and isinstance(caption, str) and caption.strip():
        return caption.strip()
    return None


def _text_mentions_visual(text: str) -> bool:
    """Check if *text* mentions a visual element pattern."""
    return bool(_VISUAL_MENTION_PATTERN.search(text))


def prepare_visual_metadata(
    element: Union[ImageSchema, ChartSchema, GraphSchema],
    doc: DocumentSchema,
    registry: Optional[Any] = None,
    settings: Optional[PipelineSettings] = None,
) -> Tuple[Union[ImageSchema, ChartSchema, GraphSchema], List[RelationshipSchema]]:
    """Compute metadata and relationships for a visual element.

    Steps:
    1. Compute image properties (width, height, aspect ratio, file size,
       color mode) when the asset file exists and image properties can be
       determined.
    2. Set ``vision_description`` to ``None`` if not already set.
    3. Populate ``visual_type`` via :func:`classify_visual_type` if empty
       or ``"unclassified"``.
    4. Generate relationships:
       - ``has_caption`` for nearby caption elements.
       - ``describes`` for nearby text blocks that discuss the visual.
       - ``nearby`` for spatially close elements.
    5. Returns the updated element and new relationships.

    Args:
        element: The visual element to enrich.
        doc: The parent document (for element lookup).
        registry: Optional ``ElementRegistry`` for element lookups.  When
            ``None``, relationship generation is skipped.
        settings: Pipeline configuration.  Defaults when ``None``.

    Returns:
        A tuple ``(updated_element, new_relationships)``.  The element is
        an updated frozen copy with metadata populated.  Relationships are
        new ``RelationshipSchema`` objects (not yet appended to the doc).
    """
    settings = settings or PipelineSettings()
    new_relationships: List[RelationshipSchema] = []
    seen_pairs: set[Tuple[uuid.UUID, uuid.UUID, str]] = set()
    updates: Dict[str, Any] = {}

    # --- 1. Image properties ---
    asset_path = element.asset_path
    if asset_path:
        asset_file = Path(asset_path)
        file_size: Optional[int] = None
        try:
            if asset_file.is_file():
                file_size = asset_file.stat().st_size
        except OSError:
            file_size = None

        # Try to open the saved image for property extraction (lazy PIL)
        props: Optional[Dict[str, Any]] = None
        if asset_file.is_file():
            try:
                from PIL import Image as PILImage

                with PILImage.open(str(asset_file)) as pil_img:
                    props = _compute_image_properties(pil_img, file_size=file_size)
            except (ImportError, AttributeError, OSError, TypeError):
                props = _compute_image_properties(None, file_size=file_size)
        else:
            props = _compute_image_properties(None, file_size=file_size)

        updates["image_properties"] = props

    # --- 2. vision_description placeholder ---
    if element.vision_description is None:
        updates["vision_description"] = None

    # --- 3. Visual type classification ---
    if not element.visual_type or element.visual_type == "unclassified":
        # Try to obtain image data for heuristics
        image_data = _obtain_image(element)
        updates["visual_type"] = classify_visual_type(element, image_data=image_data)

    # Create the updated element copy
    if updates:
        updated_element = element.model_copy(update=updates)
    else:
        updated_element = element

    # --- 4. Relationship generation (requires registry) ---
    if registry is not None and len(registry) > 0:
        new_relationships = _generate_visual_relationships(
            updated_element, registry, seen_pairs
        )

    return updated_element, new_relationships


def _generate_visual_relationships(
    element: Union[ImageSchema, ChartSchema, GraphSchema],
    registry: Any,
    seen_pairs: set,
) -> List[RelationshipSchema]:
    """Generate relationships linking a visual element to relevant elements.

    Uses the registry to find candidate elements on the same page.
    Returns new ``RelationshipSchema`` objects and updates *seen_pairs*
    to avoid duplicates.
    """
    relationships: List[RelationshipSchema] = []
    same_page = registry.get_by_page(element.page_num)

    caption_text = _extract_caption_text(element)

    for candidate in same_page:
        if candidate.element_id == element.element_id:
            continue  # No self-references

        pair_key = (element.element_id, candidate.element_id)

        # --- has_caption: caption elements ---
        if isinstance(candidate, CaptionSchema):
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            is_nearby = dist <= _CAPTION_PROXIMITY_THRESHOLD
            mentions_visual = _text_mentions_visual(candidate.content)

            # Also match if the caption text on the visual matches
            # the candidate's content.
            caption_match = (
                caption_text is not None
                and caption_text.lower() == candidate.content.lower().strip()
            )

            if is_nearby or mentions_visual or caption_match:
                _add_rel(
                    relationships,
                    seen_pairs,
                    source_id=element.element_id,
                    target_id=candidate.element_id,
                    rtype="has_caption",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6) if is_nearby else None,
                    },
                )

        # --- describes: nearby text blocks that discuss the visual ---
        if candidate.element_type in ("text_block", "list_block"):
            content_lower = candidate.content.lower()
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            mentions_visual = _text_mentions_visual(content_lower)
            is_nearby = dist <= _DESCRIBES_PROXIMITY_THRESHOLD

            if mentions_visual and is_nearby:
                _add_rel(
                    relationships,
                    seen_pairs,
                    source_id=candidate.element_id,
                    target_id=element.element_id,
                    rtype="describes",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6),
                    },
                )

        # --- nearby: spatially close elements (not already captured) ---
        # Only for non-caption, non-text_block types or when useful
        if candidate.element_type not in ("caption", "text_block", "list_block"):
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            if dist <= _NEARBY_THRESHOLD:
                _add_rel(
                    relationships,
                    seen_pairs,
                    source_id=element.element_id,
                    target_id=candidate.element_id,
                    rtype="nearby",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6),
                    },
                )

    return relationships


def _add_rel(
    relationships: List[RelationshipSchema],
    seen_pairs: set,
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    rtype: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Add a relationship if not already seen (dedup by pair + type)."""
    if source_id == target_id:
        return

    key = (source_id, target_id, rtype)
    if key in seen_pairs:
        return
    rev_key = (target_id, source_id, rtype)
    if rev_key in seen_pairs:
        return

    seen_pairs.add(key)
    relationships.append(
        RelationshipSchema(
            relationship_id=make_relationship_id(source_id, target_id, rtype),
            source_id=source_id,
            target_id=target_id,
            relationship_type=rtype,  # type: ignore[arg-type]
            metadata=metadata or {},
            weight=1.0,
        )
    )


# ---------------------------------------------------------------------------
# Convenience: process all visual elements in a document
# ---------------------------------------------------------------------------


def process_images(
    doc: DocumentSchema,
    registry: Any,
    dl_doc: Any = None,
    settings: Optional[PipelineSettings] = None,
) -> DocumentSchema:
    """Process all visual elements (image, chart, graph) in a document.

    For each visual element:

    1. Calls :func:`save_visual_asset` to persist the image.
    2. Calls :func:`prepare_visual_metadata` to compute image properties,
       visual type, and generate relationships.

    Non-visual elements are preserved unchanged.

    Args:
        doc: The document whose visual elements should be processed.
        registry: ``ElementRegistry`` with all elements (will be updated
            as elements are enriched).
        dl_doc: Optional Docling document / dict for picture extraction.
        settings: Pipeline configuration.  Defaults when ``None``.

    Returns:
        A new ``DocumentSchema`` with enriched visual elements and new
        relationships appended (without duplicates).
    """
    settings = settings or PipelineSettings()
    updated_elements: Dict[str, ElementSchema] = {}
    new_relationships: List[RelationshipSchema] = []
    seen_rel_ids: set[uuid.UUID] = set()

    # Seed with existing relationship IDs
    for rel in doc.relationships:
        seen_rel_ids.add(rel.relationship_id)
    new_relationships.extend(doc.relationships)

    for elem_key, elem in doc.elements.items():
        if isinstance(elem, _VISUAL_SCHEMA_TYPES):
            # 1. Save the visual asset
            saved = save_visual_asset(elem, dl_doc=dl_doc, doc=doc, settings=settings)

            # 2. Compute metadata and relationships
            enriched, rels = prepare_visual_metadata(
                saved, doc=doc, registry=registry, settings=settings
            )
            updated_elements[elem_key] = enriched

            # Update registry for subsequent lookups
            registry.add(enriched)

            # Append non-duplicate relationships
            for rel in rels:
                if rel.relationship_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.relationship_id)
                    new_relationships.append(rel)
        else:
            updated_elements[elem_key] = elem

    return doc.model_copy(
        update={
            "elements": updated_elements,
            "relationships": new_relationships,
        }
    )
