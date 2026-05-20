"""
File I/O utility helpers.

Provides atomic write operations, JSON/JSONL serialisation (with optional
``orjson`` acceleration), a lazy Parquet writer, a streaming copy-with-hash
helper, and path utilities.

All functions accept :class:`pathlib.Path` or anything convertible to it.
Heavy dependencies (``pandas``, ``pyarrow``) are imported lazily only when
the corresponding function is called.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

# ---------------------------------------------------------------------------
#  Lazy orjson detection
# ---------------------------------------------------------------------------
_HAS_ORJSON: bool = False
_HAS_ORJSON_ERR: Optional[str] = None
try:
    import orjson  # type: ignore[import-untyped, unused-ignore]  # noqa: F401

    _HAS_ORJSON = True
except ImportError as _exc:
    _HAS_ORJSON_ERR = str(_exc)


def _json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize *obj* to a JSON string.

    Uses ``orjson`` when installed (faster) and falls back to stdlib
    ``json`` otherwise.
    """
    if _HAS_ORJSON:
        # orjson returns bytes -> decode
        return orjson.dumps(obj).decode("utf-8")  # type: ignore[attr-defined]
    return json.dumps(obj, **kwargs)


# ---------------------------------------------------------------------------
#  Atomic write helpers
# ---------------------------------------------------------------------------


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write *content* (text) to *path*.

    A temporary file is written next to the target and then renamed,
    making the write appear atomic on most filesystems.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding=encoding)
        tmp.replace(path)
    except BaseException:
        _cleanup_tmp(tmp)
        raise


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    """Atomically write *content* (bytes) to *path*.

    See :func:`atomic_write_text` for atomicity guarantees.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(content)
        tmp.replace(path)
    except BaseException:
        _cleanup_tmp(tmp)
        raise


def _cleanup_tmp(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
#  JSON helpers
# ---------------------------------------------------------------------------


def write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    **kwargs: Any,
) -> None:
    """Write *obj* as pretty-printed JSON to *path*.

    Uses ``orjson`` internally when installed; falls back to ``json.dumps``.
    """
    content = _json_dumps(obj, indent=indent, **kwargs)
    atomic_write_text(path, content)


def read_json(path: str | Path) -> Any:
    """Read and deserialize JSON from *path*."""
    content = Path(path).read_text(encoding="utf-8")
    if _HAS_ORJSON:
        return orjson.loads(content)  # type: ignore[attr-defined]
    return json.loads(content)


# ---------------------------------------------------------------------------
#  JSONL helpers
# ---------------------------------------------------------------------------


def write_jsonl(
    path: str | Path,
    records: Sequence[dict[str, Any]],
    **kwargs: Any,
) -> None:
    """Write *records* as newline-delimited JSON to *path*.

    Each record is serialised individually so large sequences are written
    without building one giant string in memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(_json_dumps(rec, **kwargs))
                fh.write("\n")
        tmp.replace(path)
    except BaseException:
        _cleanup_tmp(tmp)
        raise


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file and return them as a list.

    For memory-efficient iteration over large files see :func:`iter_jsonl`.
    """
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL *path* one at a time (memory-friendly).

    Blank lines are silently skipped.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if _HAS_ORJSON:
                yield orjson.loads(line)  # type: ignore[attr-defined]
            else:
                yield json.loads(line)


# ---------------------------------------------------------------------------
#  Parquet helper (lazy import)
# ---------------------------------------------------------------------------


def write_parquet(
    path: str | Path,
    records: list[dict[str, Any]],
    **kwargs: Any,
) -> None:
    """Write *records* as a Parquet file via ``pandas`` + ``pyarrow``.

    Both ``pandas`` and ``pyarrow`` are imported lazily at call time so
    that importing this module does not require those heavy dependencies.
    """
    import pandas as pd  # type: ignore[import-untyped]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False, **kwargs)


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet file and return records as a list of dicts."""
    import pandas as pd  # type: ignore[import-untyped]

    df = pd.read_parquet(Path(path))
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
#  Copy-with-hash (streaming, memory-friendly)
# ---------------------------------------------------------------------------


def copy_with_hash(src: str | Path, dst: str | Path) -> str:
    """Copy *src* to *dst* and return the SHA-256 hex digest of the data.

    The file is streamed in 1 MiB chunks so that large files are handled
    without loading entirely into memory.
    """
    import hashlib

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            f_out.write(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
#  Path helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist; return the path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
