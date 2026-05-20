"""
Tests for ``src.utils`` package (Sub-Task 2).

Covers:
- All new packages import cleanly.
- PipelineSettings defaults and validation.
- Logging configuration helper.
- Atomic write helpers for text, bytes, JSON, and JSONL.
- Path helper (ensure_dir).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
#  1.  Package import sanity
# ---------------------------------------------------------------------------


class TestPackageImports:
    """Verify that all expected sub-packages are importable."""

    def test_src_importable(self) -> None:
        import src  # noqa: F401

    def test_ingestion_importable(self) -> None:
        import src.ingestion  # noqa: F401

    def test_extraction_importable(self) -> None:
        import src.extraction  # noqa: F401

    def test_normalization_importable(self) -> None:
        import src.normalization  # noqa: F401

    def test_metadata_importable(self) -> None:
        import src.metadata  # noqa: F401

    def test_chunking_importable(self) -> None:
        import src.chunking  # noqa: F401

    def test_validation_importable(self) -> None:
        import src.validation  # noqa: F401

    def test_utils_importable(self) -> None:
        import src.utils  # noqa: F401

    def test_utils_logging_importable(self) -> None:
        from src.utils.logging import configure_logging, get_logger  # noqa: F401

    def test_utils_file_io_importable(self) -> None:
        from src.utils.file_io import (  # noqa: F401
            atomic_write_bytes,
            atomic_write_text,
            copy_with_hash,
            ensure_dir,
            iter_jsonl,
            read_json,
            read_jsonl,
            read_parquet,
            write_json,
            write_jsonl,
            write_parquet,
        )

    def test_utils_config_importable(self) -> None:
        from src.utils.config import PipelineSettings  # noqa: F401

    def test_schemas_still_importable(self) -> None:
        """Critical: existing schema imports must still work."""
        from src.schemas import DocumentSchema  # noqa: F401


# ---------------------------------------------------------------------------
#  2.  PipelineSettings
# ---------------------------------------------------------------------------


class TestPipelineSettings:
    def test_defaults(self) -> None:
        from src.utils.config import PipelineSettings

        settings = PipelineSettings()
        assert settings.chunk_token_limit == 512
        assert settings.chunk_overlap == 64
        assert settings.embedding_model == "BAAI/bge-m3"
        assert str(settings.raw_dir).endswith("data/raw")
        assert str(settings.logs_dir).endswith("logs")

    def test_custom_values(self) -> None:
        from src.utils.config import PipelineSettings

        settings = PipelineSettings(
            chunk_token_limit=1024,
            chunk_overlap=128,
            embedding_model="intfloat/e5-mistral-7b-instruct",
        )
        assert settings.chunk_token_limit == 1024
        assert settings.chunk_overlap == 128
        assert settings.embedding_model == "intfloat/e5-mistral-7b-instruct"

    def test_overlap_must_be_less_than_limit(self) -> None:
        from src.utils.config import PipelineSettings

        with pytest.raises(ValueError, match="chunk_overlap"):
            PipelineSettings(chunk_token_limit=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            PipelineSettings(chunk_token_limit=100, chunk_overlap=200)

    def test_token_limit_must_be_positive(self) -> None:
        from src.utils.config import PipelineSettings

        with pytest.raises(ValueError, match="chunk_token_limit"):
            PipelineSettings(chunk_token_limit=0)
        with pytest.raises(ValueError, match="chunk_token_limit"):
            PipelineSettings(chunk_token_limit=-10)

    def test_overlap_must_be_non_negative(self) -> None:
        from src.utils.config import PipelineSettings

        with pytest.raises(ValueError, match="chunk_overlap"):
            PipelineSettings(chunk_overlap=-1)

    def test_from_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.utils.config import PipelineSettings

        monkeypatch.setenv("EDR_CHUNK_TOKEN_LIMIT", "2048")
        monkeypatch.setenv("EDR_CHUNK_OVERLAP", "256")
        monkeypatch.setenv("EDR_EMBEDDING_MODEL", "test-model")
        monkeypatch.setenv("EDR_RAW_DIR", "/tmp/test_raw")
        monkeypatch.setenv("EDR_LOGS_DIR", "/tmp/test_logs")

        settings = PipelineSettings.from_env()
        assert settings.chunk_token_limit == 2048
        assert settings.chunk_overlap == 256
        assert settings.embedding_model == "test-model"
        assert settings.raw_dir == Path("/tmp/test_raw")
        assert settings.logs_dir == Path("/tmp/test_logs")
        # Non-overridden fields keep their defaults
        assert settings.chunk_token_limit == 2048

    def test_from_env_invalid_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.utils.config import PipelineSettings

        monkeypatch.setenv("EDR_CHUNK_TOKEN_LIMIT", "not-a-number")
        with pytest.raises(ValueError, match="EDR_CHUNK_TOKEN_LIMIT"):
            PipelineSettings.from_env()


# ---------------------------------------------------------------------------
#  3.  Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_configure_logging_defaults(self) -> None:
        from src.utils.logging import configure_logging, get_logger

        configure_logging()
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == 0  # inherits from root

    def test_configure_logging_custom_level(self) -> None:
        from src.utils.logging import configure_logging

        configure_logging(level="DEBUG")
        import logging

        assert logging.getLogger().level == logging.DEBUG

    def test_get_logger(self) -> None:
        from src.utils.logging import get_logger

        logger = get_logger("my.module")
        assert logger.name == "my.module"
        assert logger.level == 0  # not set explicitly


# ---------------------------------------------------------------------------
#  4.  File I/O helpers
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_atomic_write_text(self, tmp_path: Path) -> None:
        from src.utils.file_io import atomic_write_text

        target = tmp_path / "hello.txt"
        atomic_write_text(target, "Hello, world!")
        assert target.read_text() == "Hello, world!"

    def test_atomic_write_bytes(self, tmp_path: Path) -> None:
        from src.utils.file_io import atomic_write_bytes

        target = tmp_path / "data.bin"
        atomic_write_bytes(target, b"\x00\x01\x02\xff")
        assert target.read_bytes() == b"\x00\x01\x02\xff"

    def test_atomic_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        from src.utils.file_io import atomic_write_text

        target = tmp_path / "nested" / "sub" / "file.txt"
        atomic_write_text(target, "content")
        assert target.exists()
        assert target.read_text() == "content"

    def test_write_and_read_json(self, tmp_path: Path) -> None:
        from src.utils.file_io import read_json, write_json

        target = tmp_path / "data.json"
        obj = {"key": "value", "number": 42}
        write_json(target, obj)
        assert target.exists()
        restored = read_json(target)
        assert restored == obj

    def test_write_and_read_jsonl(self, tmp_path: Path) -> None:
        from src.utils.file_io import read_jsonl, write_jsonl

        target = tmp_path / "records.jsonl"
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        write_jsonl(target, records)
        restored = read_jsonl(target)
        assert restored == records

    def test_iter_jsonl(self, tmp_path: Path) -> None:
        from src.utils.file_io import iter_jsonl, write_jsonl

        target = tmp_path / "stream.jsonl"
        records = [{"n": i} for i in range(5)]
        write_jsonl(target, records)
        seen = list(iter_jsonl(target))
        assert seen == records

    def test_copy_with_hash(self, tmp_path: Path) -> None:
        import hashlib

        from src.utils.file_io import copy_with_hash

        data = b"some data to hash"
        src = tmp_path / "source.bin"
        src.write_bytes(data)
        dst = tmp_path / "dest.bin"
        hexdigest = copy_with_hash(src, dst)
        assert dst.exists()
        assert dst.read_bytes() == data
        assert len(hexdigest) == 64  # SHA-256 hex
        expected = hashlib.sha256(data).hexdigest()
        assert hexdigest == expected

    def test_ensure_dir(self, tmp_path: Path) -> None:
        from src.utils.file_io import ensure_dir

        d = tmp_path / "a" / "b" / "c"
        result = ensure_dir(d)
        assert result == d
        assert d.is_dir()


# ---------------------------------------------------------------------------
#  5.  Config defaults do not require any env vars
# ---------------------------------------------------------------------------


def test_config_defaults_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with a clean environment, PipelineSettings should construct
    successfully."""
    from src.utils.config import PipelineSettings

    # Clear EDR_ variables
    for key in list(os.environ):
        if key.startswith("EDR_"):
            monkeypatch.delenv(key, raising=False)
    settings = PipelineSettings()
    assert settings.chunk_token_limit == 512
