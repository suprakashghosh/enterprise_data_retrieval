"""
Process tables into structured representations (markdown, HTML, JSON,
plain-text summary), extract table metadata, and handle multi-page/spanned
tables.

Public API
----------
::

    from src.normalization import (
        process_table,
        detect_spanning_tables,
        generate_table_relationships,
        process_tables,
    )
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.normalization.docling_normalizer import ElementRegistry
from src.schemas import (
    CaptionSchema,
    DocumentSchema,
    ElementSchema,
    FootnoteSchema,
    RelationshipSchema,
    TableSchema,
    make_relationship_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UUIDv5 namespace for span group IDs (stable, deterministic).
_SPAN_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "docling-project-span")

# Maximum normalised centre-to-centre distance for a caption to be
# considered "nearby" its table.
_CAPTION_PROXIMITY_THRESHOLD: float = 0.15

# Maximum normalised centre-to-centre distance for a text block to be
# considered "describing" a table.
_TEXT_PROXIMITY_THRESHOLD: float = 0.2

# Minimum columns for a potential header row to look "header-like".
_MIN_HEADER_ROWS: int = 1

# ---------------------------------------------------------------------------
# Internal helpers — tabular data normalisation
# ---------------------------------------------------------------------------


def _is_dataframe(obj: Any) -> bool:
    """Check if *obj* looks like a pandas DataFrame (lazy, no import-time
    dependency on pandas)."""
    if obj is None:
        return False
    # Duck-type: look for attributes that pandas DataFrames expose.
    for attr in ("columns", "iloc", "to_dict", "values", "head"):
        if not hasattr(obj, attr):
            return False
    return True


def _normalize_table_data(
    data: Any,
) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Convert various tabular data formats into ``(rows, headers)``.

    *rows* is a list of lists of string cell values.
    *headers* is a list of column name strings (may be empty if none
    could be distinguished).

    Supports (in priority order within this function):

    - pandas DataFrame (detected by duck-typing)
    - List of dicts (``[{"col": val, ...}, ...]``)
    - List of lists (``[["a", "b"], ["c", "d"]]``)
    - Dict of columns (``{"col1": [...], "col2": [...]}``)

    Returns ``None`` when no recognised format is found.
    """
    if data is None:
        return None

    # --- 1. pandas DataFrame (lazy detection) ---
    if _is_dataframe(data):
        try:
            headers = [str(c) for c in data.columns]
            rows: List[List[str]] = []
            for _, row in data.iterrows():
                rows.append([str(v) if v is not None else "" for v in row])
            return (rows, headers)
        except Exception:
            logger.debug("Failed to extract from DataFrame-like object", exc_info=True)

    # --- 2. List of dicts ---
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            headers = list(first.keys())
            rows = []
            for row_dict in data:
                rows.append([str(row_dict.get(h, "")) for h in headers])
            return (rows, headers)

    # --- 3. List of lists ---
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, (list, tuple)):
            rows = [[str(c) for c in row] for row in data]
            return (rows, [])

    # --- 4. Dict of columns ---
    if isinstance(data, dict):
        # Must have at least one key with a list value of length > 0
        for val in data.values():
            if isinstance(val, (list, tuple)) and len(val) > 0:
                headers = list(data.keys())
                length = max(
                    (len(v) for v in data.values() if isinstance(v, (list, tuple))),
                    default=0,
                )
                rows = []
                for i in range(length):
                    row: List[str] = []
                    for h in headers:
                        col_data = data.get(h, [])
                        if isinstance(col_data, (list, tuple)) and i < len(col_data):
                            val = col_data[i]
                            row.append(str(val) if val is not None else "")
                        else:
                            row.append("")
                    rows.append(row)
                return (rows, headers)
            break  # only inspect first value

    return None


def _looks_like_header(row: List[str]) -> bool:
    """Heuristic: a row looks like a header if most cells are short,
    non-numeric strings and are unique."""
    if not row:
        return False
    non_empty = [c for c in row if c.strip()]
    if not non_empty:
        return False
    # Most cells should be text (not purely numeric).
    numeric_count = sum(1 for c in non_empty if _is_numeric(c.strip()))
    if numeric_count > len(non_empty) // 2:
        return False
    # Values should be reasonably short (not giant text blocks).
    avg_len = sum(len(c) for c in non_empty) / len(non_empty)
    if avg_len > 60:
        return False
    return True


def _is_numeric(s: str) -> bool:
    """Check if a string is purely numeric (int or float)."""
    if not s:
        return False
    try:
        float(s.replace(",", "").replace("%", "").strip())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Internal helpers — source-specific extraction
# ---------------------------------------------------------------------------


def _extract_from_json_data(
    json_data: Dict[str, Any],
) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Extract ``(rows, headers)`` from ``json_data``.

    Expects either:
    - ``{"rows": [{"col": val, ...}, ...]}``
    - ``{"data": [[...], ...]}``
    """
    if not json_data:
        return None

    # Try "rows" key (our convention)
    rows_raw = json_data.get("rows")
    if rows_raw is not None:
        return _normalize_table_data(rows_raw)

    # Try "data" key
    data_raw = json_data.get("data")
    if data_raw is not None:
        return _normalize_table_data(data_raw)

    # Maybe json_data itself is dict-of-columns
    return _normalize_table_data(json_data)


def _extract_from_dl_item(
    dl_doc: Any,
    element: TableSchema,
) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Extract ``(rows, headers)`` from a Docling-like item or document.

    *dl_doc* may be:
    - A dict or object with a ``data`` attribute/key containing tabular data.
    - A Docling document object (dict-like with ``pages``) whose table items
      are searched for one matching *element*.

    Returns ``None`` when no compatible data is found.
    """
    if dl_doc is None:
        return None

    # --- Direct table item (has .data or ["data"]) ---
    data_attr = _get_field(dl_doc, "data")
    if data_attr is not None:
        result = _normalize_table_data(data_attr)
        if result is not None:
            return result

    # --- Docling document — search for matching table item ---
    pages = _get_field(dl_doc, "pages")
    if pages is not None:
        table_item = _find_table_item(dl_doc, element)
        if table_item is not None:
            data_attr = _get_field(table_item, "data")
            if data_attr is not None:
                result = _normalize_table_data(data_attr)
                if result is not None:
                    return result
            # Also try markdown/html from the item
            md = _get_field(table_item, "markdown", "md", default="")
            if md and not element.markdown:
                pass  # We'll handle markdown in the main flow
            html = _get_field(table_item, "html", default="")
            if html and not element.html:
                pass

    return None


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


def _find_table_item(dl_doc: Any, element: TableSchema) -> Any:
    """Find a Docling table item in *dl_doc* that corresponds to *element*.

    Matches by page number and reading order proximity.
    """
    page_num = element.page_num
    pages = _get_field(dl_doc, "pages")
    if pages is None:
        return None

    # Get the page object
    page_obj = None
    if isinstance(pages, dict):
        page_obj = pages.get(page_num)
    elif isinstance(pages, (list, tuple)):
        idx = page_num - 1
        if 0 <= idx < len(pages):
            page_obj = pages[idx]

    if page_obj is None:
        return None

    # Collect table items from the page
    table_items: List[Any] = []
    for container_name in ("tables", "table"):
        container = _get_field(page_obj, container_name)
        if isinstance(container, (list, tuple)):
            table_items.extend(container)
        elif container is not None:
            table_items.append(container)

    if not table_items:
        # Try top-level containers
        for container_name in ("tables", "table"):
            container = _get_field(dl_doc, container_name)
            if isinstance(container, (list, tuple)):
                # Filter tables on the same page
                for item in container:
                    item_page = _get_field(item, "page_num", "page", default=None)
                    if item_page == page_num or item_page is None:
                        table_items.append(item)

    if not table_items:
        return None

    # If only one table on this page, return it.
    if len(table_items) == 1:
        return table_items[0]

    # Multiple tables — try to match by reading order.
    target_order = element.reading_order
    best: Optional[Any] = None
    best_diff = float("inf")
    for item in table_items:
        item_order = _get_field(item, "reading_order", "order", "index", default=None)
        if item_order is not None:
            try:
                diff = abs(int(item_order) - target_order)
                if diff < best_diff:
                    best_diff = diff
                    best = item
            except (TypeError, ValueError):
                continue

    if best is not None and best_diff <= 3:
        return best

    # Fallback: return first table on page
    return table_items[0]


# ---------------------------------------------------------------------------
# Internal helpers — HTML / Markdown parsing
# ---------------------------------------------------------------------------


def _extract_from_html(html: str) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Parse a simple HTML table and return ``(rows, headers)``.

    Uses a minimal SAX-like approach (no external parser required).
    Handles ``<table>``, ``<tr>``, ``<th>``, ``<td>`` tags.
    """
    if not html:
        return None

    # Simple HTML table parser
    rows: List[List[str]] = []
    current_row: List[str] = []
    in_cell = False
    cell_text: List[str] = []
    in_table = False
    tag_buffer = ""
    in_tag = False

    for ch in html:
        if ch == "<":
            in_tag = True
            tag_buffer = ""
            continue
        if ch == ">":
            in_tag = False
            tag_lower = tag_buffer.strip().lower().split()[0] if tag_buffer else ""
            # Strip attributes for tag name
            tag_name = tag_lower.split()[0] if tag_lower else ""
            # Remove leading slash for closing tags
            is_closing = tag_name.startswith("/")
            tag_name_clean = tag_name.lstrip("/")

            if tag_name_clean == "table":
                in_table = not is_closing
                if is_closing:
                    # Flush any remaining row
                    if current_row:
                        rows.append(current_row)
                    current_row = []
            elif tag_name_clean in ("tr", "thead", "tbody", "tfoot"):
                if is_closing and tag_name_clean == "tr":
                    if current_row:
                        rows.append(current_row)
                    current_row = []
            elif tag_name_clean in ("td", "th"):
                if is_closing:
                    text = "".join(cell_text).strip()
                    # Collapse whitespace
                    text = " ".join(text.split())
                    current_row.append(text)
                    cell_text = []
                    in_cell = False
                else:
                    in_cell = True
                    cell_text = []
            continue

        if in_tag:
            tag_buffer += ch
            continue

        if in_cell:
            cell_text.append(ch)

    # Determine if first row looks like a header
    if not rows:
        return None

    # Check for <th> usage: we can't detect it easily from our minimal parser,
    # so use heuristic
    if _looks_like_header(rows[0]):
        # Return (data_rows, header_row) where data_rows is List[List[str]]
        # and header_row is List[str].
        return (rows[1:], rows[0]) if len(rows) > 1 else (rows, [])

    return (rows, [])


def _extract_from_markdown(
    md: str,
) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Parse a markdown table and return ``(rows, headers)``.

    Expects the GFM pipe-table format::

        | Header 1 | Header 2 |
        |----------|----------|
        | Cell 1   | Cell 2   |
    """
    if not md:
        return None

    lines = md.strip().split("\n")
    # Find the first table-like section (consecutive lines with pipes)
    table_lines: List[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            # Table ended
            break

    if len(table_lines) < 2:
        return None

    # First line — headers
    header_cells = _split_md_row(table_lines[0])
    if not header_cells:
        return None

    # Second line — separator (skip)
    # Remaining lines — data rows
    data_lines = table_lines[2:] if len(table_lines) > 2 else []
    rows = [_split_md_row(line) for line in data_lines]
    # Filter out empty rows
    rows = [r for r in rows if r and any(c.strip() for c in r)]

    return (rows, header_cells)


def _split_md_row(line: str) -> List[str]:
    """Split a markdown pipe-table row into cells."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = []
    for cell in stripped.split("|"):
        cell = cell.strip()
        # Collapse internal whitespace
        cell = " ".join(cell.split())
        cells.append(cell)
    return cells


def _extract_from_text(text: str) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Fallback: try to parse tabular data from plain text.

    Looks for lines with consistent delimiter spacing (tabs, multiple spaces).
    """
    if not text:
        return None

    lines = text.strip().split("\n")
    if len(lines) < 2:
        return None

    # Try tab-delimited first
    rows: List[List[str]] = []
    for line in lines:
        if "\t" in line:
            cells = [c.strip() for c in line.split("\t")]
            if len(cells) >= 2:
                rows.append(cells)

    if len(rows) >= 2:
        if _looks_like_header(rows[0]):
            return (rows[1:], rows[0]) if len(rows) > 1 else (rows, [])
        return (rows, [])

    # Try space-delimited (multiple spaces as separator)
    rows = []
    for line in lines:
        import re

        cells = re.split(r"\s{2,}", line.strip())
        cells = [c.strip() for c in cells if c.strip()]
        if len(cells) >= 2:
            rows.append(cells)

    if len(rows) >= 2:
        if _looks_like_header(rows[0]):
            return (rows[1:], rows[0]) if len(rows) > 1 else (rows, [])
        return (rows, [])

    return None


# ---------------------------------------------------------------------------
# Output format generators
# ---------------------------------------------------------------------------


def _rows_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Convert ``(headers, rows)`` to a GFM markdown table string."""
    if not headers and not rows:
        return ""

    parts: List[str] = []

    if headers:
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join("---" for _ in headers) + " |")
    elif rows:
        # Use first row as pseudo-header for markdown
        parts.append("| " + " | ".join(rows[0]) + " |")
        parts.append("| " + " | ".join("---" for _ in rows[0]) + " |")
        rows = rows[1:]

    for row in rows:
        parts.append("| " + " | ".join(row) + " |")

    return "\n".join(parts)


def _rows_to_html(headers: List[str], rows: List[List[str]]) -> str:
    """Convert ``(headers, rows)`` to an HTML table string."""
    if not headers and not rows:
        return ""

    parts: List[str] = ["<table>"]

    if headers:
        parts.append("  <thead>")
        parts.append("    <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
        parts.append("  </thead>")

    parts.append("  <tbody>")
    data_rows = rows if headers else (rows[1:] if len(rows) > 1 else rows)
    for row in data_rows:
        parts.append("    <tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    parts.append("  </tbody>")
    parts.append("</table>")

    return "\n".join(parts)


def _generate_summary(
    element: TableSchema,
    headers: List[str],
    row_count: int,
    col_count: int,
) -> str:
    """Generate a plain-text summary for a table."""
    parts: List[str] = []

    # Use existing summary prefix if it has content
    existing = element.summary.strip()
    if existing and not existing.startswith("Table showing"):
        parts.append(existing)

    # Build descriptor
    descriptor = f"Table with {row_count} row(s) and {col_count} column(s)"
    if headers:
        header_list = ", ".join(headers[:5])
        if len(headers) > 5:
            header_list += ", ..."
        descriptor += f", headers: [{header_list}]"

    parts.append(descriptor)
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_table(
    element: TableSchema,
    dl_doc: Any = None,
) -> TableSchema:
    """Enrich a ``TableSchema`` element with structured representations.

    Extracts tabular data from the best available source (in priority
    order):

    1. Existing ``element.json_data`` if already meaningful.
    2. Docling-like table item ``data`` (DataFrame, list-of-dicts,
       list-of-lists, dict-of-columns) when *dl_doc* is provided.
    3. HTML table string from ``element.html``.
    4. Markdown table string from ``element.markdown``.
    5. Fallback from plain ``element.content``.

    Populates / refines:
    - ``markdown``, ``html``, ``json_data``
    - ``row_count``, ``col_count``, ``headers``, ``summary``
    - ``is_spanning`` / ``span_group_id`` (only when detectable from
      the available data — *not* the place for spanning detection;
      use :func:`detect_spanning_tables` for that).

    Args:
        element: The table element to process (frozen — an updated copy
            is returned).
        dl_doc: Optional Docling document or table item with structured
            data for extraction.

    Returns:
        A new ``TableSchema`` with enriched fields.
    """
    # -- 1. Extract cell data from best available source --
    headers: List[str] = list(element.headers)
    rows_data: Optional[List[List[str]]] = None

    # Priority a: existing json_data
    if not element.json_data.get("rows") and not element.json_data.get("data"):
        pass  # json_data is empty, skip
    result = _extract_from_json_data(element.json_data)
    if result is not None:
        extracted_rows, extracted_headers = result
        if extracted_rows:
            rows_data = extracted_rows
            if extracted_headers:
                headers = extracted_headers

    # Priority b: dl_doc
    if rows_data is None and dl_doc is not None:
        result = _extract_from_dl_item(dl_doc, element)
        if result is not None:
            extracted_rows, extracted_headers = result
            if extracted_rows:
                rows_data = extracted_rows
                if extracted_headers:
                    headers = extracted_headers

    # Priority c: HTML
    if rows_data is None and element.html:
        result = _extract_from_html(element.html)
        if result is not None:
            extracted_rows, extracted_headers = result
            if extracted_rows:
                rows_data = extracted_rows
                if extracted_headers:
                    headers = extracted_headers

    # Priority d: Markdown
    if rows_data is None and element.markdown:
        result = _extract_from_markdown(element.markdown)
        if result is not None:
            extracted_rows, extracted_headers = result
            if extracted_rows:
                rows_data = extracted_rows
                if extracted_headers:
                    headers = extracted_headers

    # Priority e: Plain text content
    if rows_data is None and element.content:
        result = _extract_from_text(element.content)
        if result is not None:
            extracted_rows, extracted_headers = result
            if extracted_rows:
                rows_data = extracted_rows
                if extracted_headers:
                    headers = extracted_headers

    # -- 2. If we still have nothing, return unmodified. --
    if not rows_data:
        return element

    # -- 3. Infer headers from first row if not already known --
    if not headers and rows_data:
        first_row = rows_data[0]
        if _looks_like_header(first_row):
            headers = first_row
            rows_data = rows_data[1:] if len(rows_data) > 1 else []
        else:
            # Generate generic column names
            col_count = len(first_row) if first_row else 0
            headers = [f"Column {i + 1}" for i in range(col_count)]

    # -- 4. Compute derived metadata --
    col_count = (
        len(headers)
        if headers
        else (len(rows_data[0]) if rows_data and rows_data[0] else 0)
    )
    row_count = len(rows_data)

    # -- 5. Generate representations --
    # Use refined headers + data rows
    markdown = _rows_to_markdown(headers, rows_data)
    html = _rows_to_html(headers, rows_data)
    row_dicts: List[Dict[str, str]] = []
    if headers and rows_data:
        row_dicts = [dict(zip(headers, row)) for row in rows_data]
    json_data = {"rows": row_dicts} if row_dicts else {}
    summary = _generate_summary(element, headers, row_count, col_count)

    # -- 6. Return updated copy --
    return element.model_copy(
        update={
            "markdown": markdown,
            "html": html,
            "json_data": json_data,
            "row_count": row_count,
            "col_count": col_count,
            "headers": headers,
            "summary": summary,
        }
    )


# ---------------------------------------------------------------------------
# Spanning-table detection
# ---------------------------------------------------------------------------


def _make_span_group_id(
    doc_id: uuid.UUID,
    section_path: str,
    first_page: int,
    col_count: int,
    headers_hash: str,
) -> str:
    """Generate a deterministic span-group ID string.

    Stable across runs (UUIDv5 under ``_SPAN_NAMESPACE``).
    """
    name = f"{doc_id}:{section_path}:{first_page}:{col_count}:{headers_hash}"
    return str(uuid.uuid5(_SPAN_NAMESPACE, name))


def _headers_signature(headers: List[str]) -> str:
    """Normalised hash of column headers for comparison."""
    normalised = [h.strip().lower() for h in headers if h.strip()]
    return hashlib.md5("|".join(normalised).encode()).hexdigest()[:12]


def _table_group_key(table: TableSchema) -> Tuple:
    """Primary grouping key for spanning detection.

    Returns ``(section_path, col_count, headers_signature)``.
    """
    return (
        table.section_path,
        table.col_count,
        _headers_signature(table.headers),
    )


def detect_spanning_tables(
    elements: List[TableSchema],
) -> List[TableSchema]:
    """Detect multi-page spanning tables and mark them.

    Uses heuristics:
    - Same (or similar) section path.
    - Adjacent page numbers (consecutive or within 1 page gap).
    - Same column count.
    - Overlapping / normalised headers (signature match).

    Matched tables receive ``is_spanning=True`` and a deterministic
    ``span_group_id`` that is stable across runs.

    Args:
        elements: List of table elements to analyse.  They are *not*
            mutated — an updated copy list is returned.

    Returns:
        A new list of ``TableSchema`` objects with spanning metadata
        populated where applicable.
    """
    if not elements:
        return list(elements)

    # Group candidates by primary key
    groups: Dict[Tuple, List[int]] = {}  # key -> list of indices
    for idx, tbl in enumerate(elements):
        key = _table_group_key(tbl)
        groups.setdefault(key, []).append(idx)

    updated: Dict[int, TableSchema] = {}

    for key, indices in groups.items():
        if len(indices) < 2:
            continue  # No possibility of spanning

        section_path, col_count, _ = key

        # Sort by page number within this group
        indices.sort(key=lambda i: elements[i].page_num)

        # Find consecutive page runs
        runs: List[List[int]] = []
        current_run: List[int] = [indices[0]]
        for i in range(1, len(indices)):
            prev_page = elements[indices[i - 1]].page_num
            curr_page = elements[indices[i]].page_num
            if curr_page - prev_page <= 2:  # allow 1-page gap
                current_run.append(indices[i])
            else:
                if len(current_run) >= 2:
                    runs.append(current_run)
                current_run = [indices[i]]
        if len(current_run) >= 2:
            runs.append(current_run)

        # Mark tables in each run
        for run in runs:
            first_page = elements[run[0]].page_num
            doc_id = elements[run[0]].doc_id
            span_id = _make_span_group_id(
                doc_id=doc_id,
                section_path=section_path,
                first_page=first_page,
                col_count=col_count,
                headers_hash=_headers_signature(elements[run[0]].headers),
            )
            for idx in run:
                tbl = elements[idx]
                updated[idx] = tbl.model_copy(
                    update={
                        "is_spanning": True,
                        "span_group_id": span_id,
                    }
                )

    # Build result list: copy all elements, replace updated ones
    result: List[TableSchema] = []
    for idx, tbl in enumerate(elements):
        if idx in updated:
            result.append(updated[idx])
        else:
            result.append(tbl)

    return result


# ---------------------------------------------------------------------------
# Table relationship generation
# ---------------------------------------------------------------------------


def _bbox_center_distance(
    bbox1: Any,
    bbox2: Any,
) -> float:
    """Euclidean distance between centres of two bounding boxes."""
    c1_x = (bbox1.left + bbox1.right) / 2.0
    c1_y = (bbox1.top + bbox1.bottom) / 2.0
    c2_x = (bbox2.left + bbox2.right) / 2.0
    c2_y = (bbox2.top + bbox2.bottom) / 2.0
    return ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5


def _mentions_table(text: str) -> bool:
    """Check if *text* mentions a table (case-insensitive)."""
    lower = text.lower().strip()
    # "Table N" or "table N" or "Table N.N" pattern
    import re

    if re.search(r"\btable\s*\d+(?:\.\d+)?", lower):
        return True
    # "see table" / "as shown in table"
    if re.search(r"\b(?:see|as\s+shown\s+in|refer\s+to|from)\s+table\b", lower):
        return True
    # "the following table"
    if re.search(r"\bthe\s+following\s+table\b", lower):
        return True
    # "Table N" (capital T)
    if re.search(r"\bTable\s+\d+", text):
        return True
    return False


def _extract_table_number(text: str) -> Optional[str]:
    """Extract a table number from text (e.g. 'Table 3' -> '3')."""
    import re

    m = re.search(r"\bTable\s+(\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    return None


def generate_table_relationships(
    element: TableSchema,
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate relationships linking a table to relevant elements.

    Produces:
    - ``has_caption`` for caption elements on the same page that are
      spatially close or whose content mentions the table/table number.
    - ``describes`` from nearby text blocks that appear to discuss the
      table (same page, nearby, content mentions table/table number).
    - ``refers_to`` for footnote elements on the same page.

    Args:
        element: The table element to generate relationships for.
        registry: An ``ElementRegistry`` with access to all elements.

    Returns:
        A list of new ``RelationshipSchema`` objects.  Does **not**
        mutate any input.
    """
    relationships: List[RelationshipSchema] = []
    seen_pairs: set[Tuple[uuid.UUID, uuid.UUID, str]] = set()

    def _add_rel(
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        rtype: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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

    # Gather candidate elements on the same page
    same_page = registry.get_by_page(element.page_num)

    table_number = _extract_table_number(element.summary) or _extract_table_number(
        element.content
    )

    for candidate in same_page:
        if candidate.element_id == element.element_id:
            continue  # No self-references

        # --- has_caption: caption elements ---
        if isinstance(candidate, CaptionSchema):
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            content_lower = candidate.content.lower()

            # Check spatial proximity or table mention
            is_nearby = dist <= _CAPTION_PROXIMITY_THRESHOLD
            mentions = _mentions_table(content_lower)

            if is_nearby or mentions:
                _add_rel(
                    element.element_id,
                    candidate.element_id,
                    "has_caption",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6) if is_nearby else None,
                    },
                )

        # --- describes: nearby text blocks that discuss the table ---
        if (
            candidate.element_type == "text_block"
            or candidate.element_type == "list_block"
        ):
            content_lower = candidate.content.lower()
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            mentions = _mentions_table(content_lower)
            is_nearby = dist <= _TEXT_PROXIMITY_THRESHOLD

            if mentions and is_nearby:
                _add_rel(
                    candidate.element_id,
                    element.element_id,
                    "describes",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6),
                    },
                )

        # --- refers_to: footnotes on the same page ---
        if isinstance(candidate, FootnoteSchema):
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            if dist <= _CAPTION_PROXIMITY_THRESHOLD:
                _add_rel(
                    element.element_id,
                    candidate.element_id,
                    "refers_to",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6),
                    },
                )

    return relationships


# ---------------------------------------------------------------------------
# Convenience: process all tables in a document
# ---------------------------------------------------------------------------


def process_tables(
    doc: DocumentSchema,
    registry: ElementRegistry,
    dl_doc: Any = None,
) -> DocumentSchema:
    """Reprocess all table elements in a document.

    For each ``TableSchema`` element in *doc*:

    1. Calls :func:`process_table` to enrich representations.
    2. Calls :func:`generate_table_relationships` to create links.
    3. Collects all generated relationships, deduplicating against
       existing ones.

    Does **not** call :func:`detect_spanning_tables` — that should be
    done separately when the full list of tables is available.

    Args:
        doc: The document whose tables should be processed.
        registry: Element registry with all elements.
        dl_doc: Optional Docling document / item for additional
            structured data.

    Returns:
        A new ``DocumentSchema`` with enriched tables and new
        relationships appended (without duplicates).
    """
    updated_elements: Dict[str, ElementSchema] = {}
    new_relationships: List[RelationshipSchema] = []
    seen_rel_ids: set[uuid.UUID] = set()

    # Seed with existing relationship IDs
    for rel in doc.relationships:
        seen_rel_ids.add(rel.relationship_id)
    new_relationships.extend(doc.relationships)

    for elem_key, elem in doc.elements.items():
        if isinstance(elem, TableSchema):
            # Process table
            processed = process_table(elem, dl_doc=dl_doc)
            updated_elements[elem_key] = processed

            # Generate relationships (uses registry, so registry must
            # reflect the *updated* element for spatial queries).
            # We need to update the registry on the fly:
            registry.add(processed)
            rels = generate_table_relationships(processed, registry)
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
