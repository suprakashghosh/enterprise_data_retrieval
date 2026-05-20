"""
Pydantic models for validation reports.

Provides :class:`ValidationCheck`, :class:`ValidationReport`, and the
:data:`ValidationSeverity` type used across all validation stages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
#  Severity type
# ---------------------------------------------------------------------------

ValidationSeverity = Literal["info", "warning", "error", "critical"]

# ---------------------------------------------------------------------------
#  ValidationCheck
# ---------------------------------------------------------------------------


class ValidationCheck(BaseModel):
    """A single validation check result.

    Attributes:
        check_name: Short identifier for the check (e.g. ``"page_count"``).
        passed: Whether the check passed.
        severity: Severity level of the check.
        message: Human-readable summary of the check result.
        details: Optional detailed description or evidence.
        metadata: Optional extensible metadata dict.
    """

    check_name: str = Field(..., description="Short identifier for the check")
    passed: bool = Field(..., description="Whether the check passed")
    severity: ValidationSeverity = Field(default="error", description="Severity level")
    message: str = Field(default="", description="Human-readable summary")
    details: str = Field(default="", description="Detailed description or evidence")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extensible metadata"
    )


# ---------------------------------------------------------------------------
#  ValidationReport
# ---------------------------------------------------------------------------


class ValidationReport(BaseModel):
    """Complete validation result for a single document.

    Attributes:
        doc_id: UUID of the validated document.
        is_valid: ``True`` when all critical checks pass.
        checks: Ordered list of all :class:`ValidationCheck` results.
        errors: Shortcut list of error-level messages.
        warnings: Shortcut list of warning-level messages.
        summary: Human-readable summary of the validation outcome.
        created_at: Timestamp when the report was created.
    """

    doc_id: UUID = Field(..., description="UUID of the validated document")
    is_valid: bool = Field(default=True, description="Overall validation result")
    checks: List[ValidationCheck] = Field(
        default_factory=list, description="All check results"
    )
    errors: List[str] = Field(default_factory=list, description="Error-level messages")
    warnings: List[str] = Field(
        default_factory=list, description="Warning-level messages"
    )
    summary: str = Field(default="", description="Human-readable validation summary")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Report creation timestamp"
    )
