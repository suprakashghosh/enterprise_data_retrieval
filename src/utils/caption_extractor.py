"""
Caption label extraction — regex-based identification of figure / table /
chart numbers from document captions.

Supports research papers, company documents, financial reports, and other
document types whose captions follow the common pattern::

    <ImageType> <Number> <Separator> <Explanatory text>
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Curated canonical image types and their normalised forms
# ---------------------------------------------------------------------------
VALID_IMAGE_TYPES: dict[str, str] = {
    # --- Figure family ----------------------------------------------------
    "figure": "figure",
    "fig": "figure",
    "fig.": "figure",
    # --- Table ------------------------------------------------------------
    "table": "table",
    # --- Chart family -----------------------------------------------------
    "chart": "chart",
    # --- Graph ------------------------------------------------------------
    "graph": "graph",
    # --- Image / picture / photo family -----------------------------------
    "image": "image",
    "picture": "image",
    "photo": "image",
    "photograph": "image",
    # --- Diagram ----------------------------------------------------------
    "diagram": "diagram",
    # --- Illustration -----------------------------------------------------
    "illustration": "illustration",
    # --- Schematic / drawing ----------------------------------------------
    "schematic": "schematic",
    "drawing": "schematic",
    # --- Exhibit ----------------------------------------------------------
    "exhibit": "exhibit",
    # --- Appendix ---------------------------------------------------------
    "appendix": "appendix",
    # --- Box / panel -------------------------------------------------------
    "box": "box",
    "panel": "panel",
    # --- Algorithm ---------------------------------------------------------
    "algorithm": "algorithm",
    "algo": "algorithm",
    "alg.": "algorithm",
    # --- Equation / formula ------------------------------------------------
    "equation": "equation",
    "eq": "equation",
    "eq.": "equation",
    "eqn": "equation",
    "formula": "equation",
    # --- Map ---------------------------------------------------------------
    "map": "map",
    # --- Slide -------------------------------------------------------------
    "slide": "slide",
    # --- Plate (academic publishing) --------------------------------------
    "plate": "plate",
    # --- Scheme ------------------------------------------------------------
    "scheme": "scheme",
    # --- Listing / code ---------------------------------------------------
    "listing": "listing",
    "code": "listing",
}

# ---------------------------------------------------------------------------
# Number patterns
# ---------------------------------------------------------------------------

# Lettered suffix: 1a, 2B, 3C  (must follow a digit; avoids treating a
# standalone letter heading like "A. Introduction" as a caption)
_LETTERED = r"\d+[a-zA-Z](?:\.[a-zA-Z])?"

# Hierarchical: 1, 1.1, 1.2.3, 1.1.2.3
_HIER_NUM = r"\d+(?:\.\d+)*"

# Range continuation:  -2,  –3,  —4  (dash followed by number/letter).
# Used after the main number to capture "1-2", "3a–b", etc.
_RANGE_TAIL = r"(?:[-–—]\s*\d+[a-zA-Z]?(?:\.\d+)*|[-–—]\s*[a-zA-Z])\b"

# Roman numerals I .. XXXIX  (tens: X/XX/XXX?  ones: IX/IV/V?I{0,3}).
# The pattern guarantees at least one character is consumed.
_ROMAN = (
    r"(?:X(?:X(?:X)?)?)"
    r"(?:IX|IV|V?I{0,3})?"
    r"|"
    r"(?:IX|IV|V?I{1,3}|I{1,3})"
)

# Standalone uppercase letter: A, B, C (for appendix-style labels).
# Guarded with (?![a-z]) so "a" in "alone" doesn't match.
_LETTER = r"[A-Z](?![a-z])"

# Range tails for non-hierarchical numbers.
_RANGE_TAIL_ROMAN = r"(?:[-–—]\s*[IVXLCDM]+)?"
_RANGE_TAIL_LETTERED = r"(?:[-–—]\s*[a-zA-Z])?"

# ---------------------------------------------------------------------------
# Separators
# ---------------------------------------------------------------------------
_SEPS = r"[:\u202F\u00A0;·•,\u2013\u2014\u2012\-—|)]"

# Full separator regex — matches the separator plus any trailing whitespace.
_SEP_RE = re.compile(
    rf"\s*{_SEPS}+\s*"
    r"|\s{2,}"  # two or more spaces when no explicit separator
    r"|\s+(?=[A-Z])"  # whitespace then capital letter (implicit separator)
    r"|(?<=\d)(?=[A-Z])"  # digit directly followed by capital letter
    r"|$",  # end of string (no explanatory text)
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Separators
# ---------------------------------------------------------------------------
_SEPS = r"[:\u202F\u00A0;·•,\u2013\u2014\u2012\-—|)]"

# Full separator regex — matches the separator plus any trailing whitespace.
_SEP_RE = re.compile(
    rf"\s*{_SEPS}+\s*"
    r"|\s{2,}"  # two or more spaces when no explicit separator
    r"|\s+(?=[A-Z])"  # whitespace then capital letter (implicit separator)
    r"|(?<=\d)(?=[A-Z])"  # digit directly followed by capital letter
    r"|$",  # end of string (no explanatory text)
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Build the type-matching regex (sorted longest-first so "algorithm"
# matches before "algo", "figure" before "fig", etc.)
# ---------------------------------------------------------------------------
_SORTED_TYPES = sorted(VALID_IMAGE_TYPES.keys(), key=len, reverse=True)
_TYPE_ALTERNATION = "|".join(re.escape(t) for t in _SORTED_TYPES)

_CAPTION_RE = re.compile(
    rf"^\s*(?P<type>{_TYPE_ALTERNATION})"  # image type (case-insensitive)
    rf"\s*\.?\s*"  # optional dot after type, then whitespace
    # Number — ordered so longer / more-specific patterns are tried first.
    # Each alternation supports an optional range tail (e.g. "1-2", "XI–XII").
    rf"(?P<number>"
    rf"{_LETTERED}(?:{_RANGE_TAIL_LETTERED})?"  # 1a, 1a-b, 2B-3C
    rf"|{_ROMAN}(?:{_RANGE_TAIL_ROMAN})?"  # I, II, XI, XI–XII
    rf"|{_HIER_NUM}(?:{_RANGE_TAIL})?"  # 1, 1.1, 1-2, 1.1–2.3
    rf"|{_LETTER}"  # A, B, C (standalone)
    rf")",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_caption_label(text: str) -> Optional[str]:
    """Extract the image-type label and number from a caption *text*.

    Returns ``(normalized_type, number_string)`` on success, or ``None`` if
    the text does not appear to be a valid caption label.

    Parameters
    ----------
    text:
        Raw caption text, e.g. ``"Figure 1: Revenue by Quarter"``.

    Returns
    -------
    tuple[str, str] or None
        ``("figure", "1")`` for ``"Figure 1: Revenue …"``, or ``None``.
    """
    if not text or not text.strip():
        return None

    cleaned = _normalise_unicode(text).strip()

    m = _CAPTION_RE.match(cleaned)
    if m is None:
        return None

    raw_type = m.group("type").lower()
    raw_number = _normalise_number(m.group("number"))

    # Validate the extracted type against the curated list (after stripping
    # trailing dot so "fig." and "fig" both work).
    canonical = VALID_IMAGE_TYPES.get(raw_type)
    if canonical is None:
        # The raw type matched our regex alternation, but after stripping
        # the dot it may no longer be in the dict (edge case with mixed
        # abbreviations).  Re-check directly.
        canonical = VALID_IMAGE_TYPES.get(
            raw_type + ".", VALID_IMAGE_TYPES.get(raw_type)
        )

    if canonical is None:
        return None

    # return (canonical, raw_number)
    # return (raw_type, raw_number)
    return f"{raw_type} {raw_number}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalise_unicode(text: str) -> str:
    """Replace common Unicode dashes and non-breaking spaces with ASCII."""
    replacements = {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2012": "-",  # figure dash
        "\u2015": "-",  # horizontal bar
        "\u00a0": " ",  # non-breaking space
        "\u202f": " ",  # narrow non-breaking space
    }
    for uchar, ascii_char in replacements.items():
        text = text.replace(uchar, ascii_char)
    return text


def _normalise_number(raw_number: str) -> str:
    """Collapse whitespace and unify dashes in a captured number string."""
    n = raw_number.strip()
    n = _normalise_unicode(n)
    n = re.sub(r"\s+", " ", n)
    return n

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract caption label from caption text.')
    parser.add_argument('caption', nargs='+', help='Caption text to parse')
    args = parser.parse_args()
    caption_text = ' '.join(args.caption)
    result = extract_caption_label(caption_text)
    if result:
        print(f'{result[0]}: {result[1]}')
    else:
        print('No caption label found.')
    # test= "Table 3.2.1- Deeply nested"
    # print(extract_caption_label(test))
