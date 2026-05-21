"""
Process formulas extracted from Docling output: populate LaTeX, plain-text
approximation, inline/display classification, variable extraction, and
generate formula-specific relationships (explains, has_formula, refers_to).

Public API
----------
::

    from src.normalization import (
        process_formula,
        generate_formula_relationships,
        process_formulas,
    )
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.normalization.docling_normalizer import ElementRegistry
from src.schemas import (
    CaptionSchema,
    DocumentSchema,
    ElementSchema,
    FootnoteSchema,
    FormulaSchema,
    RelationshipSchema,
    TextBlockSchema,
    make_relationship_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum normalised centre-to-centre distance for spatially nearby elements.
_FORMULA_PROXIMITY_THRESHOLD: float = 0.15

# Maximum normalised centre-to-centre distance for a footnote.
_FOOTNOTE_PROXIMITY_THRESHOLD: float = 0.15

# Reading-order window (max difference) for considering a text block as
# explaining a formula.
_EXPLAINS_ORDER_WINDOW: int = 3

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regex patterns for formula delimiters.
# NOTE: _INLINE_DOLLAR_RE must be checked AFTER _DISPLAY_DOLLAR_RE and
# _DISPLAY_BRACKET_RE in all call sites.  The negative lookbehind on the
# closing `$` means `$$x$$` would fail to match the inner `$x$` if this
# regex were tried first, because the inner `$` is preceded by another `$`.
# Both _detect_formula_type and _extract_latex_content already try display
# delimiters first, preserving the required ordering.
_INLINE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

_DISPLAY_DOLLAR_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

_DISPLAY_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

# LaTeX command patterns for variable extraction.
_LATEX_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")
_LATEX_GREEK_RE = re.compile(
    r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|"
    r"nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)"
)

# Standalone ASCII letters often used as variables.
_STANDALONE_VAR_RE = re.compile(r"\b([a-zA-Z])\b(?!\s*\{)")

# Patterns to strip or simplify for text approximation.
_LATEX_ENV_RE = re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL)
_LATEX_MACRO_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathbb|mathcal|mathscr|mathfrak|textsf|texttt|textbf|textit|textrm|text)\{([^}]*)\}"
)
_LATEX_OP_RE = re.compile(
    r"\\(?:cdot|times|div|pm|mp|circ|bullet|ast|star|oplus|otimes|odot|oslash|equiv|approx|sim|simeq|cong|propto|infty|partial|nabla|prime|emptyset|exists|forall|neg|wedge|vee|rightarrow|leftarrow|Rightarrow|Leftarrow|mapsto|longrightarrow|longleftarrow|Longrightarrow|Longleftarrow|uparrow|downarrow|leftrightarrow|rightleftharpoons)"
)
_LATEX_SYMBOL_RE = re.compile(
    r"\\(?:left|right|big|Big|bigg|Bigg|bigl|bigr|bigm|biggl|biggr|biggm)"
)
_LATEX_ACCENT_RE = re.compile(
    r"\\(?:hat|check|tilde|acute|grave|dot|ddot|breve|bar|vec|widehat|widetilde)\{([^}]*)\}"
)
_LATEX_FRAC_RE = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
_LATEX_SUPER_SUB_RE = re.compile(r"[\^_]+\{([^}]*)\}")
_LATEX_SPACES_RE = re.compile(r"\s+")
_LATEX_BRACES_RE = re.compile(r"[\{\}]")


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


def _detect_formula_type(
    latex: str,
    content: str,
    explicit_type: Optional[str],
    bbox: Any,
) -> Optional[str]:
    """Detect whether a formula is inline or display.

    Returns ``"inline"``, ``"display"``, or ``None`` when no heuristic
    can determine the type (the caller should fall back to the
    element-level default).

    Heuristics applied in order:

    1. Explicit *formula_type* / *inline* / *display* flags from dl_doc
       (not the element default).
    2. ``$$...$$`` or ``\\[...\\]`` delimiters in the original content
       or the LaTeX text → ``"display"``.
    3. LaTeX env like ``\\begin{equation}`` → ``"display"``.
    4. Multiline content → ``"display"``.
    5. ``$...$`` delimiters in the original content → ``"inline"``.
    6. Bbox width/height ratio: wide (> 0.4) → ``"display"``,
       narrow (< 0.15) → ``"inline"``.
    7. No heuristic matched → ``None`` (caller decides).
    """
    # Explicit type flag (from dl_doc only).
    if explicit_type is not None:
        if explicit_type in ("inline", "display"):
            return explicit_type
        if explicit_type is True:
            return "inline"
        if explicit_type is False:
            return "display"

    # Check original content AND cleaned latex for delimiters.
    raw_source = content or latex

    # $$...$$ or \[...\] → display
    if _DISPLAY_DOLLAR_RE.search(raw_source) or _DISPLAY_BRACKET_RE.search(raw_source):
        return "display"

    # LaTeX environment → display
    if re.search(
        r"\\begin\{(?:equation|align|gather|multline|split|array|cases|eqnarray)\}",
        raw_source,
    ):
        return "display"

    # Multiline → display
    if "\n" in raw_source.strip():
        return "display"

    # $...$ in the original content → inline
    if _INLINE_DOLLAR_RE.search(raw_source):
        return "inline"

    # Bbox heuristic: if width > 0.4 of page (normalized) it's likely display.
    try:
        width = bbox.right - bbox.left if bbox is not None else None
        if width is not None and width > 0.4:
            return "display"
        if width is not None and width < 0.15:
            return "inline"
    except (AttributeError, TypeError):
        pass

    # No heuristic matched — let the caller fall back.
    return None


def _extract_latex_content(source: str) -> str:
    """Extract a clean LaTeX string by stripping outer delimiters.

    Handles:
    - ``$...$``
    - ``$$...$$``
    - ``\\[...\\]``
    - Raw LaTeX (no delimiters)
    """
    source = source.strip()

    # Try to find a LaTeX expression inside delimiters.
    m = _DISPLAY_DOLLAR_RE.search(source)
    if m:
        return m.group(1).strip()
    m = _DISPLAY_BRACKET_RE.search(source)
    if m:
        return m.group(1).strip()
    m = _INLINE_DOLLAR_RE.search(source)
    if m:
        return m.group(1).strip()

    # If the entire content looks like LaTeX (contains \, _, ^, etc.)
    if "\\" in source or "{" in source or "}" in source:
        return source

    return source


def _make_text_approximation(latex: str) -> str:
    """Convert a LaTeX string into a simple plain-text approximation.

    This is intentionally heuristic and lightweight — no CAS or full LaTeX
    parser is used.
    """
    if not latex:
        return ""

    text = latex

    # Remove LaTeX environments entirely.
    text = _LATEX_ENV_RE.sub("", text)

    # Simplify \frac{a}{b} → a/b
    text = _LATEX_FRAC_RE.sub(r"\1 / \2", text)

    # Simplify \text{...} and similar to just the content.
    text = _LATEX_MACRO_RE.sub(r"\1", text)

    # Remove \left, \right, \big, etc.
    text = _LATEX_SYMBOL_RE.sub("", text)

    # Accent commands: \hat{x} → x̂ (keep the char)
    text = _LATEX_ACCENT_RE.sub(r"\1", text)

    # Operator names → text equivalents
    op_map = {
        "cdot": " * ",
        "times": " * ",
        "div": " / ",
        "pm": " +/- ",
        "mp": " -/+ ",
        "circ": " deg ",
        "bullet": " * ",
        "rightarrow": " -> ",
        "leftarrow": " <- ",
        "Rightarrow": " => ",
        "Leftarrow": " <= ",
        "mapsto": " -> ",
        "longrightarrow": " -> ",
        "longleftarrow": " <- ",
        "Longrightarrow": " => ",
        "Longleftarrow": " <= ",
        "leftrightarrow": " <-> ",
        "infty": "infinity",
        "partial": "partial ",
        "nabla": "nabla ",
        "emptyset": "empty",
        "exists": "exists ",
        "forall": "forall ",
        "neg": "not ",
        "wedge": " and ",
        "vee": " or ",
        "approx": " ~ ",
        "simeq": " ~ ",
        "cong": " ~ ",
        "equiv": " == ",
        "sim": " ~ ",
        "propto": " proportional to ",
        "prime": "'",
    }

    def _replace_op(m: re.Match) -> str:
        name = m.group(1)
        return op_map.get(name, f" {name} ")

    text = _LATEX_OP_RE.sub(_replace_op, text)

    # Remove superscript/subscript braces: ^{...} and _{...}
    text = _LATEX_SUPER_SUB_RE.sub(r" \1", text)

    # Remove stray braces
    text = _LATEX_BRACES_RE.sub(" ", text)

    # Replace common LaTeX commands with their names
    text = _LATEX_COMMAND_RE.sub(r" \1 ", text)

    # Collapse whitespace
    text = _LATEX_SPACES_RE.sub(" ", text).strip()

    return text


def _extract_variables(latex: str, content: str) -> List[str]:
    """Extract variable/symbol names from LaTeX or plain text.

    Uses deterministic regex heuristics — no external dependencies.
    Returns a sorted unique list of variable names.
    """
    source = latex or content
    variables: set[str] = set()

    # Greek letter commands → variable names
    for m in _LATEX_GREEK_RE.finditer(source):
        name = m.group(1)
        variables.add(name)

    # Other named LaTeX commands that look like variables
    for m in _LATEX_COMMAND_RE.finditer(source):
        name = m.group(1)
        if name.lower() not in (
            # Skip structural/formatting commands.
            "text",
            "mathrm",
            "mathbf",
            "mathit",
            "mathsf",
            "mathtt",
            "mathbb",
            "mathcal",
            "mathscr",
            "mathfrak",
            "textsf",
            "texttt",
            "textbf",
            "textit",
            "textrm",
            "frac",
            "left",
            "right",
            "big",
            "bigl",
            "bigr",
            "bigm",
            "biggl",
            "biggr",
            "biggm",
            "quad",
            "qquad",
            "hspace",
            "vspace",
            "mbox",
            "makebox",
            "raisebox",
            "resizebox",
            "scalebox",
            "rotatebox",
            "reflectbox",
            "style",
            "displaystyle",
            "textstyle",
            "scriptstyle",
            "scriptscriptstyle",
            "limits",
            "nolimits",
            "over",
            "atop",
            "choose",
            "brack",
            "brace",
            "label",
            "ref",
            "cite",
            "tag",
            "notag",
            "nonumber",
            "begin",
            "end",
            "caption",
            "footnote",
            "label",
        ):
            variables.add(name)

    # Standalone single-letter variables (a-z, A-Z)
    for m in _STANDALONE_VAR_RE.finditer(source):
        letter = m.group(1)
        # Skip letters that are part of LaTeX commands (preceded by \)
        start = m.start()
        if start > 0 and source[start - 1] == "\\":
            continue
        variables.add(letter)

    # Sort for determinism.
    return sorted(variables)


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


def _mentions_formula(text: str) -> bool:
    """Check if *text* mentions a formula/equation (case-insensitive)."""
    lower = text.lower().strip()
    if not lower:
        return False
    # "equation N" or "Equation N" or "formula N"
    if re.search(r"\b(?:equation|formula|eq\.?)\s*\d+(?:\.\d+)?", lower):
        return True
    # "the following equation" / "the following formula"
    if re.search(r"\b(?:the\s+)?following\s+(?:equation|formula)\b", lower):
        return True
    # "see equation" / "as shown in equation"
    if re.search(
        r"\b(?:see|as\s+shown\s+in|refer\s+to|from|in)\s+(?:equation|formula|eq\.?)\b",
        lower,
    ):
        return True
    # "Eq." or "Eqn."
    if re.search(r"\b(?:eq|eqn)\.\s*\d+", lower):
        return True
    return False


# ===================================================================
# Public API
# ===================================================================


def process_formula(
    element: FormulaSchema,
    dl_doc: Any = None,
) -> FormulaSchema:
    """Enrich a ``FormulaSchema`` element with processed formula data.

    Populates or refines:
    - ``latex`` — cleaned LaTeX expression
    - ``text_approximation`` — plain-text approximation
    - ``formula_type`` — ``"inline"`` or ``"display"``
    - ``variables`` — extracted variable/symbol names

    Input sources (in priority order):

    a. Existing fields on the ``FormulaSchema`` element.
    b. Docling-like item fields discoverable from *dl_doc*.
    c. Fallback to ``element.content``.

    Args:
        element: The formula element to process (frozen — an updated copy
            is returned).
        dl_doc: Optional Docling document or formula item with additional
            fields (``latex``, ``formula_type``, etc.).

    Returns:
        A new ``FormulaSchema`` with enriched fields.
    """
    # --- 1. Extract raw material from best available source ---
    raw_latex: str = element.latex
    raw_text_approx: str = element.text_approximation
    raw_content: str = element.content

    # Explicit formula-type flag (from dl_doc only, not element default).
    explicit_formula_type: Optional[str] = None

    # Priority b: dl_doc fields
    if dl_doc is not None:
        dl_latex = _get_field(dl_doc, "latex", "text", default=None)
        if dl_latex is not None and isinstance(dl_latex, str) and dl_latex.strip():
            raw_latex = dl_latex

        dl_text_approx = _get_field(dl_doc, "text_approximation", default=None)
        if (
            dl_text_approx is not None
            and isinstance(dl_text_approx, str)
            and dl_text_approx.strip()
        ):
            raw_text_approx = dl_text_approx

        dl_formula_type = _get_field(dl_doc, "formula_type", default=None)
        if dl_formula_type is not None:
            if dl_formula_type in ("inline", "display"):
                explicit_formula_type = dl_formula_type
            elif dl_formula_type is True:
                explicit_formula_type = "inline"
            elif dl_formula_type is False:
                explicit_formula_type = "display"
            elif isinstance(dl_formula_type, str):
                explicit_formula_type = dl_formula_type.lower()

        dl_inline = _get_field(dl_doc, "inline", default=None)
        if dl_inline is not None and explicit_formula_type is None:
            if isinstance(dl_inline, bool):
                explicit_formula_type = "inline" if dl_inline else "display"

        dl_display = _get_field(dl_doc, "display", default=None)
        if dl_display is not None and explicit_formula_type is None:
            if isinstance(dl_display, bool):
                explicit_formula_type = "display" if dl_display else "inline"

    # Priority c: fallback to element.content
    if not raw_latex.strip() and raw_content.strip():
        raw_latex = raw_content

    # --- 2. Clean LaTeX ---
    latex = _extract_latex_content(raw_latex)

    # --- 3. Detect formula type ---
    bbox = element.bbox
    formula_type = _detect_formula_type(latex, raw_content, explicit_formula_type, bbox)

    # Fallback: if heuristics could not determine type, use element default.
    if formula_type is None:
        formula_type = element.formula_type or "display"

    # --- 4. Text approximation ---
    if raw_text_approx.strip():
        text_approximation = raw_text_approx
    else:
        text_approximation = _make_text_approximation(latex)

    # --- 5. Extract variables ---
    variables = _extract_variables(latex, raw_content)

    # --- 6. Return updated copy ---
    return element.model_copy(
        update={
            "latex": latex,
            "text_approximation": text_approximation,
            "formula_type": formula_type,
            "variables": variables,
        }
    )


def generate_formula_relationships(
    element: FormulaSchema,
    registry: ElementRegistry,
) -> List[RelationshipSchema]:
    """Generate relationships linking a formula to relevant elements.

    Produces:
    - ``explains`` from nearby text blocks that likely discuss the formula
      (same page, close proximity or reading-order adjacency, content
      mentions formula/equation keywords).
    - ``has_formula`` from caption / table / container elements that
      structurally contain or reference the formula.
    - ``refers_to`` for footnote elements on the same page that are
      spatially close.

    Args:
        element: The formula element to generate relationships for.
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
    formula_order = element.reading_order

    for candidate in same_page:
        if candidate.element_id == element.element_id:
            continue  # No self-references

        # --- explains: text blocks on same page ---
        if isinstance(candidate, TextBlockSchema):
            content_lower = candidate.content.lower()
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            mentions = _mentions_formula(content_lower)
            order_diff = abs(candidate.reading_order - formula_order)

            # A text block explains a formula if it is:
            # - nearby spatially, OR
            # - nearby in reading order AND mentions formula keywords
            is_nearby = dist <= _FORMULA_PROXIMITY_THRESHOLD
            is_adjacent = order_diff <= _EXPLAINS_ORDER_WINDOW

            if is_nearby or (is_adjacent and mentions):
                _add_rel(
                    candidate.element_id,
                    element.element_id,
                    "explains",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6),
                        "reading_order_diff": order_diff,
                    },
                )

        # --- has_formula: container/caption elements that reference the formula ---
        if isinstance(candidate, CaptionSchema):
            content_lower = candidate.content.lower()
            mentions = _mentions_formula(content_lower)
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            is_nearby = dist <= _FORMULA_PROXIMITY_THRESHOLD

            if mentions or is_nearby:
                _add_rel(
                    candidate.element_id,
                    element.element_id,
                    "has_formula",
                    metadata={
                        "page_num": element.page_num,
                        "distance": round(dist, 6) if is_nearby else None,
                    },
                )

        # --- refers_to: footnotes on the same page ---
        if isinstance(candidate, FootnoteSchema):
            dist = _bbox_center_distance(element.bbox, candidate.bbox)
            if dist <= _FOOTNOTE_PROXIMITY_THRESHOLD:
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


def process_formulas(
    doc: DocumentSchema,
    registry: ElementRegistry,
    dl_doc: Any = None,
) -> DocumentSchema:
    """Reprocess all formula elements in a document.

    For each ``FormulaSchema`` element in *doc*:

    1. Calls :func:`process_formula` to enrich formula fields.
    2. Calls :func:`generate_formula_relationships` to create links.
    3. Collects all generated relationships, deduplicating against
       existing ones.

    Preserves non-formula elements unchanged.

    Args:
        doc: The document whose formulas should be processed.
        registry: Element registry with all elements.
        dl_doc: Optional Docling document / item for additional
            structured data.

    Returns:
        A new ``DocumentSchema`` with enriched formulas and new
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
        if isinstance(elem, FormulaSchema):
            # Process formula
            processed = process_formula(elem, dl_doc=dl_doc)
            updated_elements[elem_key] = processed

            # Generate relationships (registry must reflect the updated
            # element for spatial queries).
            registry.add(processed)
            rels = generate_formula_relationships(processed, registry)
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
