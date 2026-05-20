"""
Pipeline configuration.

Provides a :class:`PipelineSettings` dataclass with sensible defaults and
lightweight environment-variable override support.  No third-party
configuration library is required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
#  Project root detection
# ---------------------------------------------------------------------------
# Resolved from the location of this file: src/utils/config.py -> src/ -> .
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
#  Settings dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineSettings:
    """Global pipeline configuration.

    All directory fields default to paths under the project root and can
    be overridden via environment variables (see :meth:`from_env`).

    Parameters
    ----------
    chunk_token_limit : int
        Maximum tokens per chunk (default 512).
    chunk_overlap : int
        Token overlap between adjacent chunks (default 64).
    embedding_model : str
        HuggingFace model ID for text embeddings
        (default ``"BAAI/bge-m3"``).
    raw_dir : Path
        Root for raw (immutable) PDF storage.
    data_dir : Path
        Root for intermediate pipeline data.
    output_dir : Path
        Root for final exports (JSONL, Parquet, reports).
    docling_dir : Path
        Docling conversion outputs (JSON, markdown, page images).
    chunks_dir : Path
        Chunk output artifacts.
    metadata_dir : Path
        Metadata export path.
    assets_dir : Path
        Extracted visual assets (images, tables).
    logs_dir : Path
        Pipeline logs.
    """

    # ------------------------------------------------------------------
    #  Chunking parameters
    # ------------------------------------------------------------------
    chunk_token_limit: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "BAAI/bge-m3"

    # ------------------------------------------------------------------
    #  Directory paths (defaults relative to project root)
    # ------------------------------------------------------------------
    raw_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "outputs")
    docling_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "docling"
    )
    chunks_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "outputs" / "chunks"
    )
    metadata_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "outputs" / "metadata"
    )
    assets_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "assets")
    logs_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "logs")

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate settings invariants after construction."""
        if self.chunk_overlap >= self.chunk_token_limit:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_token_limit ({self.chunk_token_limit})"
            )
        if self.chunk_token_limit <= 0:
            raise ValueError(
                f"chunk_token_limit must be positive, got {self.chunk_token_limit}"
            )
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )

    # ------------------------------------------------------------------
    #  Environment-variable override
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> PipelineSettings:
        """Create settings with defaults overridden by environment variables.

        Supported variables are all prefixed with ``EDR_``:

        =========================  ====================
        Environment variable       Settings attribute
        =========================  ====================
        ``EDR_CHUNK_TOKEN_LIMIT``  ``chunk_token_limit``
        ``EDR_CHUNK_OVERLAP``      ``chunk_overlap``
        ``EDR_EMBEDDING_MODEL``    ``embedding_model``
        ``EDR_RAW_DIR``            ``raw_dir``
        ``EDR_DATA_DIR``           ``data_dir``
        ``EDR_OUTPUT_DIR``         ``output_dir``
        ``EDR_DOCLING_DIR``        ``docling_dir``
        ``EDR_CHUNKS_DIR``         ``chunks_dir``
        ``EDR_METADATA_DIR``       ``metadata_dir``
        ``EDR_ASSETS_DIR``         ``assets_dir``
        ``EDR_LOGS_DIR``           ``logs_dir``
        =========================  ====================
        """
        kwargs: dict[str, str | int | Path] = {}

        _parse_int("EDR_CHUNK_TOKEN_LIMIT", kwargs, "chunk_token_limit")
        _parse_int("EDR_CHUNK_OVERLAP", kwargs, "chunk_overlap")
        _parse_str("EDR_EMBEDDING_MODEL", kwargs, "embedding_model")

        for env_key, attr in [
            ("EDR_RAW_DIR", "raw_dir"),
            ("EDR_DATA_DIR", "data_dir"),
            ("EDR_OUTPUT_DIR", "output_dir"),
            ("EDR_DOCLING_DIR", "docling_dir"),
            ("EDR_CHUNKS_DIR", "chunks_dir"),
            ("EDR_METADATA_DIR", "metadata_dir"),
            ("EDR_ASSETS_DIR", "assets_dir"),
            ("EDR_LOGS_DIR", "logs_dir"),
        ]:
            val = os.environ.get(env_key)
            if val is not None:
                kwargs[attr] = Path(val)

        return cls(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _parse_int(env_key: str, kwargs: dict[str, Any], attr: str) -> None:
    val = os.environ.get(env_key)
    if val is not None:
        try:
            kwargs[attr] = int(val)
        except ValueError:
            raise ValueError(
                f"Environment variable {env_key}={val!r} cannot be parsed as int"
            ) from None


def _parse_str(env_key: str, kwargs: dict[str, Any], attr: str) -> None:
    val = os.environ.get(env_key)
    if val is not None:
        kwargs[attr] = val
