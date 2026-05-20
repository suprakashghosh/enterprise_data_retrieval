"""
``src.validation`` — Output quality validation.

Validates Docling outputs and pipeline artifacts for completeness,
consistency, and structural integrity.  Produces structured
``ValidationReport`` instances for every check run.

Public API
----------
::

    from src.validation import (
        ValidationCheck,
        ValidationReport,
        ValidationSeverity,
        DoclingValidator,
        validate_docling_output,
    )
"""

from src.validation.docling_validator import DoclingValidator, validate_docling_output
from src.validation.models import ValidationCheck, ValidationReport, ValidationSeverity

__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "ValidationSeverity",
    "DoclingValidator",
    "validate_docling_output",
]
