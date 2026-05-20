"""
Validator for Docling extraction outputs.

Provides :class:`DoclingValidator` with individual check methods for
testability, and a convenience :func:`validate_docling_output` function.

The validator is fully defensive: it supports dict/list pages, fake
objects, and exported dict structures.  It does **not** require the real
``docling`` package at import time or runtime.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from src.schemas import DocumentSchema
from src.utils.logging import get_logger
from src.validation.models import ValidationCheck, ValidationReport

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_KNOWN_ITEM_TYPES = frozenset(
    {
        "text",
        "table",
        "figure",
        "formula",
        "caption",
        "heading",
        "list",
        "checkbox",
        "picture",
        "chart",
        "graph",
        "header",
        "footer",
        "footnote",
        "page_number",
        "diagram",
        "shape",
        "line",
        "group",
    }
)

_PAGE_NUM_FIELDS = ("page_no", "page_num", "page")
_BBOX_FIELDS = ("bbox",)
_TYPE_FIELDS = ("type", "label", "kind")
_ORDER_FIELDS = ("order", "reading_order", "index")


# ===================================================================
#  Internal helpers
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


def _normalize_bbox(
    bbox: Any,
) -> Optional[Tuple[float, float, float, float]]:
    """Normalise a bounding box to ``(left, top, right, bottom)``.

    Supports the following input forms:

    - ``BoundingBox`` model / object with ``left/top/right/bottom``
    - Object or dict with ``l/t/r/b``
    - Dict with ``left/top/right/bottom``
    - Tuple or list ``[left, top, right, bottom]``

    Returns ``None`` when the input cannot be parsed.
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
            return _normalize_bbox(bbox[0])
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


def _is_valid_bbox(
    bbox_tuple: Tuple[float, float, float, float],
    page_dim: Optional[Tuple[float, float]] = None,
) -> bool:
    """Return ``True`` if the bounding box is valid.

    A valid bounding box has:
    - Finite coordinates (not NaN, not inf).
    - ``left >= 0``, ``top >= 0``, ``right >= 0``, ``bottom >= 0``.
    - ``left < right`` and ``top < bottom`` (strictly positive dimensions).
    - If *page_dim* is provided and positive, coordinates are within page
      bounds.
    """
    left, top, right, bottom = bbox_tuple

    # Check finite
    for v in bbox_tuple:
        if math.isnan(v) or math.isinf(v):
            return False

    # Non-negative
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        return False

    # Positive area (strict ordering)
    if left >= right or top >= bottom:
        return False

    # Within page dimensions
    if page_dim is not None:
        pw, ph = page_dim
        if pw is not None and ph is not None and pw > 0 and ph > 0:
            if right > pw or bottom > ph:
                return False

    return True


def _get_page_dimensions(page_obj: Any) -> Optional[Tuple[float, float]]:
    """Try to extract ``(width, height)`` from a page object.

    Recognises:
    - ``page.size`` where ``size`` is a ``Size`` model, a dict, or an
      object with ``width`` / ``height``.
    - ``page.width`` / ``page.height`` directly.
    """
    if page_obj is None:
        return None

    # Try .size (Size model or object/dict with width/height)
    size = _get_field(page_obj, "size")
    if size is not None:
        w = _get_field(size, "width")
        h = _get_field(size, "height")
        if w is not None and h is not None:
            try:
                return (float(w), float(h))
            except (TypeError, ValueError):
                pass

    # Direct width/height fields on the page object
    w = _get_field(page_obj, "width")
    h = _get_field(page_obj, "height")
    if w is not None and h is not None:
        try:
            return (float(w), float(h))
        except (TypeError, ValueError):
            pass

    return None


def _iter_items(dl_doc: Any) -> List[Tuple[Any, int]]:
    """Iterate over all content items in a Docling-like document.

    Supports:
    - ``dl_doc.pages`` as a ``dict[int, Page]`` (Docling's native format).
    - ``dl_doc.pages`` as a ``list[Page]``.
    - Top-level ``dl_doc.texts`` and ``dl_doc.tables`` lists.
    - Items contained in per-page ``page.items``, ``page.texts``, or
      ``page.tables``.

    Returns a list of ``(item, page_num)`` tuples.  Page number is 0 for
    items whose page could not be determined.
    """
    items: List[Tuple[Any, int]] = []
    seen_ids: set[int] = set()

    pages = _get_field(dl_doc, "pages")
    if pages is None:
        # No page structure — try top-level texts / tables
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
    """Collect items from a single page object.

    Each item is associated with its own ``page_num`` field if present,
    falling back to the container page's number.
    """
    for container_field in ("items", "texts", "tables", "pictures", "figures"):
        container = _get_field(page_obj, container_field)
        if container is None:
            continue
        if isinstance(container, (list, tuple)):
            for item in container:
                if item is not None and id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    # Prefer item's own page_num over the container page
                    item_page = _get_field(item, *_PAGE_NUM_FIELDS)
                    effective_page = (
                        int(item_page) if item_page is not None else page_num
                    )
                    items.append((item, effective_page))


def _collect_top_level_items(
    dl_doc: Any,
    items: List[Tuple[Any, int]],
    seen_ids: set[int],
) -> None:
    """Collect items from top-level containers (texts, tables, etc.)."""
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


def _extract_bbox_from_item(
    item: Any,
) -> Optional[Tuple[float, float, float, float]]:
    """Extract a normalised bounding box from a single item.

    Tries ``item.bbox`` first, then falls back to the first provenance
    entry's bounding box.
    """
    bbox_raw = _get_field(item, *_BBOX_FIELDS)
    if bbox_raw is not None:
        bbox = _normalize_bbox(bbox_raw)
        if bbox is not None:
            return bbox

    # Try provenance
    prov = _get_field(item, "prov")
    if prov is not None and isinstance(prov, (list, tuple)) and len(prov) > 0:
        prov_entry = prov[0]
        bbox_raw = _get_field(prov_entry, "bbox", "bbox_normalized")
        if bbox_raw is not None:
            return _normalize_bbox(bbox_raw)

    return None


def _build_page_dim_map(
    dl_doc: Any,
) -> Dict[int, Optional[Tuple[float, float]]]:
    """Build a mapping of page number → (width, height) from *dl_doc*."""
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
#  DoclingValidator
# ===================================================================


class DoclingValidator:
    """Validates Docling extraction outputs against a ``DocumentSchema``.

    Each validation concern is implemented as a separate public method so
    that individual checks can be tested in isolation.

    The :meth:`validate` method runs all checks and returns a
    :class:`ValidationReport`.  No check ever raises an exception —
    unexpected errors are caught and converted into a failed
    ``ValidationCheck``.
    """

    def __init__(self) -> None:
        self._logger = get_logger(f"{__name__}.DoclingValidator")

    # ------------------------------------------------------------------
    #  Orchestration
    # ------------------------------------------------------------------

    def validate(self, dl_doc: Any, doc: DocumentSchema) -> ValidationReport:
        """Run all validation checks and return a report.

        Parameters
        ----------
        dl_doc:
            A Docling-like document object or exported dict.
        doc:
            The :class:`DocumentSchema` to validate against.

        Returns
        -------
        ValidationReport
            Never raises; returns a report even for malformed input.
        """
        report = ValidationReport(doc_id=doc.doc_id)

        # (name, method, is_critical)
        check_specs: List[Tuple[str, Any, bool]] = [
            ("page_count", self.check_page_count, True),
            ("not_empty", self.check_not_empty, True),
            ("all_bboxes_valid", self.check_all_bboxes_valid, True),
            ("page_number_valid", self.check_page_number_valid, False),
            ("item_types", self.check_item_types, False),
            ("individual_bboxes", self.check_individual_bboxes, False),
            ("reading_order", self.check_reading_order, False),
            ("tables", self.check_tables, False),
            ("captions", self.check_captions, False),
        ]

        for name, method, is_critical in check_specs:
            try:
                check = method(dl_doc, doc)
            except Exception as exc:
                check = ValidationCheck(
                    check_name=name,
                    passed=False,
                    severity="critical" if is_critical else "error",
                    message=f"Check '{name}' raised an exception: {exc}",
                    details=str(exc),
                )
                self._logger.warning("Validation check '%s' raised: %s", name, exc)

            report.checks.append(check)

            if not check.passed:
                if check.severity in ("error", "critical"):
                    report.errors.append(check.message)
                elif check.severity == "warning":
                    report.warnings.append(check.message)

                if is_critical:
                    report.is_valid = False

        # Build summary
        passed = sum(1 for c in report.checks if c.passed)
        total = len(report.checks)
        report.summary = (
            f"Validation {'PASSED' if report.is_valid else 'FAILED'}: "
            f"{passed}/{total} checks passed, "
            f"{len(report.errors)} errors, {len(report.warnings)} warnings"
        )

        return report

    # ------------------------------------------------------------------
    #  Individual check methods
    # ------------------------------------------------------------------

    def check_page_count(self, dl_doc: Any, doc: DocumentSchema) -> ValidationCheck:
        """**Critical.** Docling page count must match ``doc.page_count``.

        When ``doc.page_count <= 0`` the check is skipped (info-level).
        """
        dl_page_count = _get_field(dl_doc, "page_count")
        if dl_page_count is None:
            pages = _get_field(dl_doc, "pages")
            if pages is not None:
                if isinstance(pages, dict) and pages:
                    dl_page_count = len(pages)
                elif isinstance(pages, (list, tuple)) and pages:
                    dl_page_count = len(pages)

        if dl_page_count is None:
            return ValidationCheck(
                check_name="page_count",
                passed=True,
                severity="info",
                message="Page count could not be determined from Docling output",
            )

        try:
            dl_page_count = int(dl_page_count)
        except (TypeError, ValueError):
            return ValidationCheck(
                check_name="page_count",
                passed=True,
                severity="info",
                message=f"Docling page count ({dl_page_count!r}) is not a valid integer",
            )

        doc_page_count = doc.page_count

        if doc_page_count > 0 and dl_page_count != doc_page_count:
            return ValidationCheck(
                check_name="page_count",
                passed=False,
                severity="critical",
                message=(
                    f"Page count mismatch: Docling={dl_page_count}, "
                    f"Schema={doc_page_count}"
                ),
                details=(
                    f"The Docling output reports {dl_page_count} page(s) "
                    f"but the document schema reports {doc_page_count} page(s)."
                ),
            )

        return ValidationCheck(
            check_name="page_count",
            passed=True,
            severity="info",
            message=f"Page count matches: {dl_page_count}",
        )

    def check_not_empty(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Critical.** The Docling output must contain at least one item
        or non-empty markdown exported content.

        An ``export_to_dict`` returning a trivial dict (e.g.
        ``{"type": "document"}``) does **not** count as content.
        """
        items = _iter_items(dl_doc)

        # Check for non-empty markdown export
        has_export = False
        try:
            md_fn = _get_field(dl_doc, "export_to_markdown")
            if callable(md_fn):
                md_content = md_fn()
                has_export = bool(md_content and md_content.strip())
        except Exception:
            pass

        if not items and not has_export:
            return ValidationCheck(
                check_name="not_empty",
                passed=False,
                severity="critical",
                message="Docling output is empty: no items and no exported content",
            )

        return ValidationCheck(
            check_name="not_empty",
            passed=True,
            severity="info",
            message=f"Docling output has {len(items)} item(s)",
        )

        return ValidationCheck(
            check_name="not_empty",
            passed=True,
            severity="info",
            message=f"Docling output has {len(items)} item(s)",
        )

    def check_all_bboxes_valid(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Critical.** Fail if *every* discovered bounding box is invalid.

        When there are no items with bounding boxes the check passes with
        an info-level message (not all items are expected to have bboxes).
        """
        items = _iter_items(dl_doc)
        if not items:
            return ValidationCheck(
                check_name="all_bboxes_valid",
                passed=True,
                severity="info",
                message="No items to check for bounding boxes",
            )

        page_dims = _build_page_dim_map(dl_doc)

        valid_count = 0
        invalid_count = 0
        total_with_bbox = 0

        for item, page_num in items:
            bbox = _extract_bbox_from_item(item)
            if bbox is None:
                continue
            total_with_bbox += 1
            dim = page_dims.get(page_num) or page_dims.get(0)
            if _is_valid_bbox(bbox, dim):
                valid_count += 1
            else:
                invalid_count += 1

        if total_with_bbox == 0:
            return ValidationCheck(
                check_name="all_bboxes_valid",
                passed=True,
                severity="info",
                message="No bounding boxes found to validate",
            )

        if valid_count == 0 and invalid_count > 0:
            return ValidationCheck(
                check_name="all_bboxes_valid",
                passed=False,
                severity="critical",
                message=f"All {invalid_count} bounding box(es) are invalid",
                details=f"Found {total_with_bbox} bbox(es), all invalid",
            )

        return ValidationCheck(
            check_name="all_bboxes_valid",
            passed=True,
            severity="info",
            message=(
                f"Bounding boxes: {valid_count} valid, "
                f"{invalid_count} invalid out of {total_with_bbox}"
            ),
        )

    def check_page_number_valid(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Warning.** Items referencing pages outside the valid range."""
        max_page = _get_field(dl_doc, "page_count")
        if max_page is None:
            pages = _get_field(dl_doc, "pages")
            if pages is not None:
                if isinstance(pages, dict) and pages:
                    max_page = len(pages)
                elif isinstance(pages, (list, tuple)) and pages:
                    max_page = len(pages)

        if max_page is None:
            return ValidationCheck(
                check_name="page_number_valid",
                passed=True,
                severity="info",
                message="Page count unknown, skipping page range check",
            )

        try:
            max_page = int(max_page)
        except (TypeError, ValueError):
            return ValidationCheck(
                check_name="page_number_valid",
                passed=True,
                severity="info",
                message="Page count could not be parsed, skipping check",
            )

        invalid_pages: List[str] = []
        items = _iter_items(dl_doc)
        for _, page_num in items:
            if page_num < 1 or page_num > max_page:
                invalid_pages.append(str(page_num))

        if invalid_pages:
            return ValidationCheck(
                check_name="page_number_valid",
                passed=False,
                severity="warning",
                message=f"{len(invalid_pages)} item(s) reference invalid page numbers",
                details=(
                    f"Invalid page numbers: {', '.join(sorted(set(invalid_pages)))}"
                ),
            )

        return ValidationCheck(
            check_name="page_number_valid",
            passed=True,
            severity="info",
            message="All items reference valid page numbers",
        )

    def check_item_types(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Warning.** Unrecognised item types."""
        unknown: Dict[str, int] = {}
        total = 0
        items = _iter_items(dl_doc)

        for item, _ in items:
            typ = _get_item_type(item)
            total += 1
            if typ is not None and typ not in _KNOWN_ITEM_TYPES:
                unknown[typ] = unknown.get(typ, 0) + 1

        if not total:
            return ValidationCheck(
                check_name="item_types",
                passed=True,
                severity="info",
                message="No items to check",
            )

        if unknown:
            details = "; ".join(f"'{t}': {c}" for t, c in sorted(unknown.items()))
            return ValidationCheck(
                check_name="item_types",
                passed=False,
                severity="warning",
                message=(f"Found {sum(unknown.values())} item(s) with unknown types"),
                details=details,
            )

        return ValidationCheck(
            check_name="item_types",
            passed=True,
            severity="info",
            message=f"All {total} item(s) have known types",
        )

    def check_individual_bboxes(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Warning.** Individual invalid bounding boxes.

        Invalid individual boxes generate warnings.  When *all* boxes are
        invalid this check still warns (the critical
        :meth:`check_all_bboxes_valid` handles the critical failure).
        """
        items = _iter_items(dl_doc)
        if not items:
            return ValidationCheck(
                check_name="individual_bboxes",
                passed=True,
                severity="info",
                message="No items to check",
            )

        page_dims = _build_page_dim_map(dl_doc)
        invalid_items: List[str] = []
        valid_count = 0
        total_with_bbox = 0

        for item, page_num in items:
            bbox = _extract_bbox_from_item(item)
            if bbox is None:
                continue
            total_with_bbox += 1
            dim = page_dims.get(page_num) or page_dims.get(0)
            if _is_valid_bbox(bbox, dim):
                valid_count += 1
            else:
                item_type = _get_item_type(item) or "unknown"
                invalid_items.append(f"page={page_num}, type={item_type}")

        if invalid_items and valid_count >= 0:
            # Show details when there are any invalid boxes
            detail_str = "; ".join(invalid_items[:20])
            if len(invalid_items) > 20:
                detail_str += f" … and {len(invalid_items) - 20} more"

            return ValidationCheck(
                check_name="individual_bboxes",
                passed=False,
                severity="warning",
                message=(f"{len(invalid_items)} item(s) have invalid bounding boxes"),
                details=detail_str,
            )

        return ValidationCheck(
            check_name="individual_bboxes",
            passed=True,
            severity="info",
            message=(
                f"All {total_with_bbox} bounding box(es) are valid"
                if total_with_bbox > 0
                else "No bounding boxes to check"
            ),
        )

    def check_reading_order(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Warning.** Reading-order gaps or non-monotonic order per page.

        Only reports when items carry explicit order fields (``order``,
        ``reading_order``, or ``index``).
        """
        items = _iter_items(dl_doc)
        if not items:
            return ValidationCheck(
                check_name="reading_order",
                passed=True,
                severity="info",
                message="No items to check reading order",
            )

        # Group explicit order values by page
        page_orders: Dict[int, List[int]] = {}
        for item, page_num in items:
            order_val = _get_field(item, *_ORDER_FIELDS)
            if order_val is None:
                continue
            try:
                order_val = int(order_val)
            except (TypeError, ValueError):
                continue
            page_orders.setdefault(page_num, []).append(order_val)

        if not page_orders:
            return ValidationCheck(
                check_name="reading_order",
                passed=True,
                severity="info",
                message="No ordering information found on items",
            )

        gaps: List[str] = []
        non_monotonic: List[str] = []

        for page_num in sorted(page_orders):
            orders = sorted(page_orders[page_num])
            if len(orders) <= 1:
                continue

            # Check for duplicate values (non-monotonic / duplicate orders)
            if len(orders) != len(set(orders)):
                non_monotonic.append(f"page {page_num}")

            # Check gaps
            min_o, max_o = orders[0], orders[-1]
            expected_count = max_o - min_o + 1
            actual_count = len(set(orders))
            missing = expected_count - actual_count
            if missing > 0:
                gaps.append(f"page {page_num}: {missing} missing order value(s)")

        messages: List[str] = []
        if gaps:
            messages.append(f"Reading order gaps: {'; '.join(gaps)}")
        if non_monotonic:
            messages.append(f"Non-monotonic order on: {', '.join(non_monotonic)}")

        if messages:
            return ValidationCheck(
                check_name="reading_order",
                passed=False,
                severity="warning",
                message="; ".join(messages),
            )

        return ValidationCheck(
            check_name="reading_order",
            passed=True,
            severity="info",
            message="Reading order is monotonic with no gaps",
        )

    def check_tables(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Warning.** Tables with zero rows or columns."""
        items = _iter_items(dl_doc)

        table_issues: List[str] = []
        table_count = 0

        for item, page_num in items:
            typ = _get_item_type(item)
            is_table = typ == "table"

            # Also count items that have table-like data containers
            if not is_table:
                data = _get_field(item, "data", "cells", "rows")
                if data is None:
                    continue
                # Could still be a table even if not explicitly labeled
                is_table = True

            if not is_table:
                continue

            table_count += 1

            rows = _get_field(item, "rows", "num_rows", "row_count")
            cols = _get_field(item, "cols", "num_cols", "col_count")

            # Try to infer from .data
            if rows is None:
                data = _get_field(item, "data")
                if data is not None:
                    if isinstance(data, (list, tuple)):
                        rows = len(data)

            if cols is None:
                data = _get_field(item, "data")
                if data is not None:
                    if isinstance(data, (list, tuple)) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, (list, tuple)):
                            cols = len(first)
                        elif isinstance(first, dict):
                            cols = len(first)

            try:
                rows_val = int(rows) if rows is not None else None
            except (TypeError, ValueError):
                rows_val = None
            try:
                cols_val = int(cols) if cols is not None else None
            except (TypeError, ValueError):
                cols_val = None

            if rows_val is not None and rows_val == 0:
                table_issues.append(f"page {page_num}: table {table_count} has 0 rows")
            if cols_val is not None and cols_val == 0:
                table_issues.append(
                    f"page {page_num}: table {table_count} has 0 columns"
                )

        if table_issues:
            return ValidationCheck(
                check_name="tables",
                passed=False,
                severity="warning",
                message=f"Table issue(s) found ({len(table_issues)})",
                details="; ".join(table_issues),
            )

        return ValidationCheck(
            check_name="tables",
            passed=True,
            severity="info",
            message=(
                f"All {table_count} table(s) have valid dimensions"
                if table_count > 0
                else "No tables found"
            ),
        )

    def check_captions(
        self,
        dl_doc: Any,
        doc: DocumentSchema,  # noqa: ARG002
    ) -> ValidationCheck:
        """**Warning.** Visual items (figures, tables, charts) missing
        captions."""
        items = _iter_items(dl_doc)

        needs_caption = 0
        no_caption: List[str] = []

        for item, page_num in items:
            typ = _get_item_type(item)
            if typ not in ("figure", "table", "chart", "graph", "picture"):
                continue
            needs_caption += 1

            # Direct caption field
            caption = _get_field(item, "caption", "caption_text")
            if caption is not None:
                cap_text = caption if isinstance(caption, str) else str(caption)
                if cap_text.strip():
                    continue

            # Check children for caption-type items
            children = _get_field(item, "children", "items")
            if children is not None and isinstance(children, (list, tuple)):
                found = False
                for child in children:
                    child_type = _get_item_type(child)
                    if child_type == "caption":
                        child_text = _get_item_text(child)
                        if child_text.strip():
                            found = True
                            break
                if not found:
                    no_caption.append(f"page {page_num}, type={typ}")
            else:
                no_caption.append(f"page {page_num}, type={typ}")

        if no_caption:
            detail_str = "; ".join(no_caption[:20])
            if len(no_caption) > 20:
                detail_str += f" … and {len(no_caption) - 20} more"
            return ValidationCheck(
                check_name="captions",
                passed=False,
                severity="warning",
                message=f"{len(no_caption)} visual item(s) are missing captions",
                details=detail_str,
            )

        if needs_caption > 0:
            return ValidationCheck(
                check_name="captions",
                passed=True,
                severity="info",
                message=f"All {needs_caption} visual item(s) have captions",
            )

        return ValidationCheck(
            check_name="captions",
            passed=True,
            severity="info",
            message="No visual items found (captions not applicable)",
        )


# ===================================================================
#  Convenience function
# ===================================================================


def validate_docling_output(dl_doc: Any, doc: DocumentSchema) -> ValidationReport:
    """Validate a Docling output document against the schema.

    This is a convenience wrapper that creates a :class:`DoclingValidator`
    and runs :meth:`DoclingValidator.validate`.

    Parameters
    ----------
    dl_doc:
        A Docling-like document object or exported dict.
    doc:
        The :class:`DocumentSchema` to validate against.

    Returns
    -------
    ValidationReport
        Never raises; returns a report even for malformed input.
    """
    validator = DoclingValidator()
    return validator.validate(dl_doc, doc)
