"""
``src.utils`` — Shared logging, configuration, file I/O utilities, and
caption extraction.
"""

from src.utils.caption_extractor import extract_caption_label  # noqa: F401

__all__: list[str] = ["extract_caption_label"]
