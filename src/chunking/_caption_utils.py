"""
Internal helpers for extracting caption text from Docling items.

Shared between ``visual_enricher.py`` and ``cross_reference_resolver.py``
to avoid copy-paste duplication.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def get_caption_for_item(item: Any, text_lookup: Dict[str, str]) -> str:
    """Extract the caption text for a picture or table Docling item.

    Tries ``captions[0].cref`` first, then ``children[0].cref`` as fallback.
    """
    captions = getattr(item, "captions", None)
    children = getattr(item, "children", None)

    cref: Optional[str] = None
    if captions:
        cref = getattr(captions[0], "cref", None)
    elif children:
        cref = getattr(children[0], "cref", None)

    if cref:
        return text_lookup.get(cref, "")
    return ""
