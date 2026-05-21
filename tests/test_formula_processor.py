"""
Tests for ``src.normalization.formula_processor`` (Sub-Task 10 — Process
Formulas and Link to Explanatory Text).

Uses fakes throughout — no real Docling dependency.

Covers:
- Public imports.
- process_formula populates/refines latex, text_approximation,
  formula_type, and variables.
- Inline vs display detection.
- Variable extraction from LaTeX and plain text.
- Empty/no-formula-data handling.
- generate_formula_relationships creates ``explains`` for nearby
  explanatory text.
- Optional nearby footnote / has_formula behavior.
- No duplicate/self relationships.
- process_formulas updates the document and preserves non-formulas.
- Deterministic/stable behavior across repeated runs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from src.normalization import (
    ElementRegistry,
    generate_formula_relationships,
    process_formula,
    process_formulas,
)
from src.schemas import (
    BoundingBox,
    CaptionSchema,
    DocumentSchema,
    ElementSchema,
    FootnoteSchema,
    FormulaSchema,
    RelationshipSchema,
    TextBlockSchema,
)


# ===================================================================
#  Helpers — synthetic elements and documents
# ===================================================================

_DOC_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def _bbox(
    left: float = 0.0,
    top: float = 0.0,
    right: float = 0.3,
    bottom: float = 0.1,
) -> BoundingBox:
    return BoundingBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        coord_system="normalized",
    )


def _make_doc(page_count: int = 5) -> DocumentSchema:
    return DocumentSchema(
        doc_id=_DOC_ID,
        title="Test Formula Doc",
        source_path="/fake/formula_test.pdf",
        file_hash="formula1234",
        page_count=page_count,
        created_at=datetime(2025, 1, 1),
    )


def _make_formula(
    elem_id: Optional[str] = None,
    page_num: int = 1,
    reading_order: int = 0,
    section_path: str = "",
    content: str = "",
    latex: str = "",
    text_approximation: str = "",
    formula_type: str = "display",
    variables: Optional[List[str]] = None,
    bbox: Optional[BoundingBox] = None,
) -> FormulaSchema:
    return FormulaSchema(
        element_id=uuid.UUID(elem_id or "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        section_path=section_path,
        element_type="formula",
        content=content,
        latex=latex,
        text_approximation=text_approximation,
        formula_type=formula_type,
        variables=variables or [],
    )


def _make_text_block(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    bbox: Optional[BoundingBox] = None,
) -> TextBlockSchema:
    return TextBlockSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        element_type="text_block",
        content=content,
    )


def _make_caption(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    bbox: Optional[BoundingBox] = None,
) -> CaptionSchema:
    return CaptionSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        element_type="caption",
        content=content,
    )


def _make_footnote(
    elem_id: str,
    content: str,
    page_num: int = 1,
    reading_order: int = 0,
    bbox: Optional[BoundingBox] = None,
) -> FootnoteSchema:
    return FootnoteSchema(
        element_id=uuid.UUID(elem_id),
        doc_id=_DOC_ID,
        page_num=page_num,
        bbox=bbox or _bbox(),
        reading_order=reading_order,
        element_type="footnote",
        content=content,
    )


def _build_registry(elements: List[ElementSchema]) -> ElementRegistry:
    reg = ElementRegistry()
    for elem in elements:
        reg.add(elem)
    return reg


# ===================================================================
#  1.  Public imports
# ===================================================================


class TestPublicImports:
    """All public API symbols are importable."""

    def test_imports(self) -> None:
        assert callable(process_formula)
        assert callable(generate_formula_relationships)
        assert callable(process_formulas)

    def test_import_from_normalization(self) -> None:
        from src.normalization import (
            generate_formula_relationships as g,
            process_formula as p,
            process_formulas as ps,
        )

        assert callable(p)
        assert callable(g)
        assert callable(ps)


# ===================================================================
#  2.  process_formula — basic field population
# ===================================================================


class TestProcessFormulaBasicPopulation:
    """process_formula populates/refines all formula fields."""

    def test_preserves_existing_latex(self) -> None:
        """Existing LaTeX is preserved."""
        elem = _make_formula(
            latex=r"E = mc^2",
            content="E = mc^2",
        )
        result = process_formula(elem)
        assert result.latex == r"E = mc^2"
        assert result.text_approximation != ""
        assert result.formula_type in ("inline", "display")

    def test_populates_from_content_fallback(self) -> None:
        """Falls back to content when latex is empty."""
        elem = _make_formula(
            latex="",
            content=r"\alpha + \beta = \gamma",
        )
        result = process_formula(elem)
        assert "alpha" in result.latex
        assert result.latex != ""
        assert result.text_approximation != ""

    def test_populates_from_dl_doc(self) -> None:
        """dl_doc fields are used when no existing fields."""
        elem = _make_formula(latex="", content="")
        dl_doc = {"latex": r"x^2 + y^2 = z^2"}
        result = process_formula(elem, dl_doc=dl_doc)
        assert "x^2" in result.latex
        assert "y^2" in result.latex

    def test_populates_text_approximation(self) -> None:
        """text_approximation is populated from LaTeX."""
        elem = _make_formula(latex=r"\frac{a}{b}")
        result = process_formula(elem)
        assert result.text_approximation != ""
        assert "/" in result.text_approximation

    def test_preserves_existing_text_approximation(self) -> None:
        """Existing text_approximation is kept."""
        elem = _make_formula(
            latex=r"E = mc^2",
            text_approximation="E = m c squared",
        )
        result = process_formula(elem)
        assert result.text_approximation == "E = m c squared"

    def test_populates_variables(self) -> None:
        """Variables list is populated."""
        elem = _make_formula(latex=r"E = mc^2")
        result = process_formula(elem)
        assert len(result.variables) > 0
        # Expect 'E', 'm', 'c' or similar
        assert "E" in result.variables or "m" in result.variables

    def test_preserves_existing_variables(self) -> None:
        """Existing variables list is replaced (processor re-extracts)."""
        elem = _make_formula(
            latex=r"x + y = z",
            variables=["x", "y"],
        )
        result = process_formula(elem)
        # Variables should be re-extracted from latex
        assert "x" in result.variables
        assert "y" in result.variables
        assert "z" in result.variables


# ===================================================================
#  3.  process_formula — inline vs display detection
# ===================================================================


class TestProcessFormulaInlineDisplay:
    """Inline vs display detection heuristics."""

    def test_explicit_inline_type(self) -> None:
        """Explicit formula_type='inline' is respected."""
        elem = _make_formula(latex=r"x + y", formula_type="inline")
        result = process_formula(elem)
        assert result.formula_type == "inline"

    def test_explicit_display_type(self) -> None:
        """Explicit formula_type='display' is respected."""
        elem = _make_formula(latex=r"x + y", formula_type="display")
        result = process_formula(elem)
        assert result.formula_type == "display"

    def test_dollar_dollar_display(self) -> None:
        r"""$$...$$ delimiters → display."""
        elem = _make_formula(content=r"$$\int_a^b f(x) dx$$")
        result = process_formula(elem)
        assert result.formula_type == "display"

    def test_bracket_display(self) -> None:
        r"""\[...\] delimiters → display."""
        elem = _make_formula(content=r"\[\sum_{i=1}^n i\]")
        result = process_formula(elem)
        assert result.formula_type == "display"

    def test_inline_dollar(self) -> None:
        r"""$...$ delimiters → inline."""
        elem = _make_formula(content=r"the value $x$ is positive")
        result = process_formula(elem)
        assert result.formula_type == "inline"

    def test_equation_env_display(self) -> None:
        """\\begin{equation} → display."""
        elem = _make_formula(content=r"\begin{equation}E = mc^2\end{equation}")
        result = process_formula(elem)
        assert result.formula_type == "display"

    def test_multiline_display(self) -> None:
        """Multiline content → display."""
        elem = _make_formula(content="x = 1\ny = 2")
        result = process_formula(elem)
        assert result.formula_type == "display"

    def test_bbox_wide_display(self) -> None:
        """Wide bbox (width > 0.4) → display."""
        wide_bbox = _bbox(left=0.0, top=0.0, right=0.5, bottom=0.1)
        elem = _make_formula(latex=r"x + y", bbox=wide_bbox)
        result = process_formula(elem)
        assert result.formula_type == "display"

    def test_bbox_narrow_inline(self) -> None:
        """Narrow bbox (width < 0.15) → inline."""
        narrow_bbox = _bbox(left=0.4, top=0.0, right=0.5, bottom=0.1)
        elem = _make_formula(latex=r"x", bbox=narrow_bbox)
        result = process_formula(elem)
        assert result.formula_type == "inline"


# ===================================================================
#  4.  process_formula — variable extraction
# ===================================================================


class TestProcessFormulaVariableExtraction:
    """Variable/symbol name extraction."""

    def test_extracts_greek_letters(self) -> None:
        """Greek letter commands become variable names."""
        elem = _make_formula(latex=r"\alpha + \beta = \gamma")
        result = process_formula(elem)
        assert "alpha" in result.variables
        assert "beta" in result.variables
        assert "gamma" in result.variables

    def test_extracts_single_letters(self) -> None:
        """Standalone single letters are extracted."""
        elem = _make_formula(latex=r"x + y = z")
        result = process_formula(elem)
        assert "x" in result.variables
        assert "y" in result.variables
        assert "z" in result.variables

    def test_extracts_latex_commands(self) -> None:
        """Named LaTeX commands like \\sin are extracted."""
        elem = _make_formula(latex=r"\sin(x) + \cos(x)")
        result = process_formula(elem)
        assert "sin" in result.variables
        assert "cos" in result.variables

    def test_no_duplicate_variables(self) -> None:
        """Variables list has unique entries."""
        elem = _make_formula(latex=r"x + x = 2x")
        result = process_formula(elem)
        # 'x' should appear only once
        assert result.variables.count("x") == 1

    def test_begin_end_not_extracted(self) -> None:
        r"""``\begin`` and ``\end`` are not extracted as variables."""
        elem = _make_formula(
            latex=r"\begin{equation}E = mc^2\end{equation}",
        )
        result = process_formula(elem)
        assert "begin" not in result.variables
        assert "end" not in result.variables

    def test_sorted_variables(self) -> None:
        """Variables are sorted deterministically."""
        elem = _make_formula(latex=r"z + y + x")
        result = process_formula(elem)
        assert result.variables == sorted(result.variables)

    def test_from_plain_text(self) -> None:
        """Variable extraction from plain text (no LaTeX)."""
        elem = _make_formula(content="x + y = z", latex="")
        result = process_formula(elem)
        assert "x" in result.variables
        assert "y" in result.variables
        assert "z" in result.variables


# ===================================================================
#  5.  process_formula — empty/malformed data handling
# ===================================================================


class TestProcessFormulaEmpty:
    """Graceful handling of empty or malformed inputs."""

    def test_fully_empty(self) -> None:
        """Completely empty formula does not crash."""
        elem = _make_formula(latex="", content="")
        result = process_formula(elem)
        assert result.latex == ""
        # Should still have sensible defaults
        assert isinstance(result.text_approximation, str)
        assert result.formula_type in ("inline", "display")
        assert isinstance(result.variables, list)

    def test_no_dl_doc(self) -> None:
        """Calling without dl_doc does not fail."""
        elem = _make_formula(latex=r"x = 1")
        result = process_formula(elem, dl_doc=None)
        assert result.latex == r"x = 1"

    def test_dl_doc_is_none(self) -> None:
        """Explicit None dl_doc is handled."""
        elem = _make_formula(latex=r"a^2 + b^2")
        result = process_formula(elem, dl_doc=None)
        assert result.latex == r"a^2 + b^2"

    def test_dl_doc_missing_fields(self) -> None:
        """dl_doc with missing formula fields is handled."""
        elem = _make_formula(latex=r"x")
        dl_doc = {"other_field": "value"}
        result = process_formula(elem, dl_doc=dl_doc)
        assert result.latex == r"x"

    def test_return_is_new_instance(self) -> None:
        """Returned element is a new frozen instance (model_copy)."""
        elem = _make_formula(latex=r"E = mc^2")
        result = process_formula(elem)
        assert result is not elem  # Different object
        assert result.model_config.get("frozen")  # Still frozen


# ===================================================================
#  6.  generate_formula_relationships — explains
# ===================================================================


class TestGenerateFormulaRelationshipsExplains:
    """``explains`` relationships from nearby text blocks."""

    def test_nearby_text_block_explains(self) -> None:
        """Text block close to formula gets 'explains'."""
        formula_bbox = _bbox(left=0.1, top=0.3, right=0.4, bottom=0.4)
        text_bbox = _bbox(left=0.1, top=0.18, right=0.5, bottom=0.28)
        formula = _make_formula(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc01",
            page_num=1,
            reading_order=1,
            bbox=formula_bbox,
            latex=r"E = mc^2",
        )
        text = _make_text_block(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc02",
            content="This equation describes energy-mass equivalence.",
            page_num=1,
            reading_order=0,
            bbox=text_bbox,
        )
        registry = _build_registry([formula, text])

        rels = generate_formula_relationships(formula, registry)
        explains = [r for r in rels if r.relationship_type == "explains"]
        assert len(explains) >= 1
        assert explains[0].source_id == text.element_id
        assert explains[0].target_id == formula.element_id

    def test_text_mentioning_equation_adjacent_order(self) -> None:
        """Text block that mentions 'Equation' and is near in order."""
        formula = _make_formula(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc11",
            page_num=1,
            reading_order=2,
            latex=r"\sum_{i=1}^n i",
        )
        text = _make_text_block(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc12",
            content="As shown in Equation 3, the sum converges.",
            page_num=1,
            reading_order=0,
            bbox=_bbox(left=0.7, top=0.7, right=0.9, bottom=0.8),
        )
        registry = _build_registry([formula, text])

        rels = generate_formula_relationships(formula, registry)
        explains = [r for r in rels if r.relationship_type == "explains"]
        assert len(explains) >= 1

    def test_distant_text_no_explains(self) -> None:
        """Text far away and not mentioning formula → no explains."""
        formula_bbox = _bbox(left=0.1, top=0.1, right=0.3, bottom=0.2)
        text_bbox = _bbox(left=0.7, top=0.7, right=0.9, bottom=0.8)
        formula = _make_formula(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc21",
            page_num=1,
            reading_order=0,
            bbox=formula_bbox,
            latex=r"x = 1",
        )
        text = _make_text_block(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc22",
            content="This is unrelated text about the weather.",
            page_num=1,
            reading_order=10,
            bbox=text_bbox,
        )
        registry = _build_registry([formula, text])

        rels = generate_formula_relationships(formula, registry)
        explains = [r for r in rels if r.relationship_type == "explains"]
        assert len(explains) == 0

    def test_different_page_no_relationship(self) -> None:
        """Elements on different pages do not get relationships."""
        formula = _make_formula(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc31",
            page_num=1,
            latex=r"x = 1",
        )
        text = _make_text_block(
            elem_id="cccccccc-cccc-cccc-cccc-cccccccccc32",
            content="About equation 1.",
            page_num=2,
        )
        registry = _build_registry([formula, text])

        rels = generate_formula_relationships(formula, registry)
        explains = [r for r in rels if r.relationship_type == "explains"]
        assert len(explains) == 0


# ===================================================================
#  7.  generate_formula_relationships — has_formula
# ===================================================================


class TestGenerateFormulaRelationshipsHasFormula:
    """``has_formula`` relationships from caption/container elements."""

    def test_caption_mentioning_equation(self) -> None:
        """Caption mentioning 'Equation' gets 'has_formula'."""
        formula = _make_formula(
            elem_id="dddddddd-dddd-dddd-dddd-dddddddddd01",
            page_num=1,
            reading_order=1,
            latex=r"f(x) = ax^2 + bx + c",
        )
        caption = _make_caption(
            elem_id="dddddddd-dddd-dddd-dddd-dddddddddd02",
            content="Equation 5: Quadratic formula",
            page_num=1,
            reading_order=0,
        )
        registry = _build_registry([formula, caption])

        rels = generate_formula_relationships(formula, registry)
        has_formula = [r for r in rels if r.relationship_type == "has_formula"]
        assert len(has_formula) >= 1
        assert has_formula[0].source_id == caption.element_id
        assert has_formula[0].target_id == formula.element_id

    def test_caption_nearby_spatially(self) -> None:
        """Caption spatially close to formula gets 'has_formula'."""
        form_bbox = _bbox(left=0.1, top=0.3, right=0.5, bottom=0.4)
        cap_bbox = _bbox(left=0.1, top=0.41, right=0.5, bottom=0.45)
        formula = _make_formula(
            elem_id="dddddddd-dddd-dddd-dddd-dddddddddd11",
            page_num=1,
            bbox=form_bbox,
            latex=r"\int f(x) dx",
        )
        caption = _make_caption(
            elem_id="dddddddd-dddd-dddd-dddd-dddddddddd12",
            content="Figure caption",
            page_num=1,
            bbox=cap_bbox,
        )
        registry = _build_registry([formula, caption])

        rels = generate_formula_relationships(formula, registry)
        has_formula = [r for r in rels if r.relationship_type == "has_formula"]
        assert len(has_formula) >= 1

    def test_caption_far_away_no_formula_keyword(self) -> None:
        """Caption far away and not mentioning formula → no has_formula."""
        form_bbox = _bbox(left=0.1, top=0.1, right=0.3, bottom=0.2)
        cap_bbox = _bbox(left=0.7, top=0.7, right=0.9, bottom=0.8)
        formula = _make_formula(
            elem_id="dddddddd-dddd-dddd-dddd-dddddddddd21",
            page_num=1,
            bbox=form_bbox,
            latex=r"x = 1",
        )
        caption = _make_caption(
            elem_id="dddddddd-dddd-dddd-dddd-dddddddddd22",
            content="Some distant caption.",
            page_num=1,
            bbox=cap_bbox,
        )
        registry = _build_registry([formula, caption])

        rels = generate_formula_relationships(formula, registry)
        has_formula = [r for r in rels if r.relationship_type == "has_formula"]
        assert len(has_formula) == 0


# ===================================================================
#  8.  generate_formula_relationships — refers_to footnotes
# ===================================================================


class TestGenerateFormulaRelationshipsRefersTo:
    """``refers_to`` relationships for nearby footnotes."""

    def test_footnote_nearby(self) -> None:
        """Footnote spatially close to formula."""
        form_bbox = _bbox(left=0.1, top=0.2, right=0.5, bottom=0.3)
        fn_bbox = _bbox(left=0.1, top=0.31, right=0.3, bottom=0.35)
        formula = _make_formula(
            elem_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeee01",
            page_num=1,
            bbox=form_bbox,
            latex=r"x = y",
        )
        fn = _make_footnote(
            elem_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeee02",
            content="This is a footnote about the formula.",
            page_num=1,
            bbox=fn_bbox,
        )
        registry = _build_registry([formula, fn])

        rels = generate_formula_relationships(formula, registry)
        refers_to = [r for r in rels if r.relationship_type == "refers_to"]
        assert len(refers_to) >= 1
        assert refers_to[0].source_id == formula.element_id
        assert refers_to[0].target_id == fn.element_id

    def test_footnote_far_away_no_relationship(self) -> None:
        """Footnote far away from formula → no refers_to."""
        form_bbox = _bbox(left=0.1, top=0.1, right=0.3, bottom=0.2)
        fn_bbox = _bbox(left=0.7, top=0.7, right=0.9, bottom=0.8)
        formula = _make_formula(
            elem_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeee11",
            page_num=1,
            bbox=form_bbox,
            latex=r"x = y",
        )
        fn = _make_footnote(
            elem_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeee12",
            content="Distant footnote.",
            page_num=1,
            bbox=fn_bbox,
        )
        registry = _build_registry([formula, fn])

        rels = generate_formula_relationships(formula, registry)
        refers_to = [r for r in rels if r.relationship_type == "refers_to"]
        assert len(refers_to) == 0


# ===================================================================
#  9.  generate_formula_relationships — edge cases
# ===================================================================


class TestGenerateFormulaRelationshipsEdgeCases:
    """Self-references, duplicates, determinism."""

    def test_no_self_reference(self) -> None:
        """Formula should not relate to itself."""
        formula = _make_formula(
            elem_id="ffffffff-ffff-ffff-ffff-fffffffffff1",
            page_num=1,
        )
        registry = _build_registry([formula])
        rels = generate_formula_relationships(formula, registry)
        self_refs = [
            r
            for r in rels
            if r.source_id == formula.element_id and r.target_id == formula.element_id
        ]
        assert len(self_refs) == 0

    def test_no_duplicate_relationships(self) -> None:
        """Calling twice on same inputs gives same IDs (deterministic)."""
        form_bbox = _bbox(left=0.1, top=0.3, right=0.5, bottom=0.4)
        text_bbox = _bbox(left=0.1, top=0.18, right=0.5, bottom=0.28)
        formula = _make_formula(
            elem_id="ffffffff-ffff-ffff-ffff-fffffffffff2",
            page_num=1,
            reading_order=1,
            bbox=form_bbox,
            latex=r"E = mc^2",
        )
        text = _make_text_block(
            elem_id="ffffffff-ffff-ffff-ffff-fffffffffff3",
            content="This equation is famous.",
            page_num=1,
            reading_order=0,
            bbox=text_bbox,
        )
        registry = _build_registry([formula, text])

        rels1 = generate_formula_relationships(formula, registry)
        rels2 = generate_formula_relationships(formula, registry)

        ids1 = {r.relationship_id for r in rels1}
        ids2 = {r.relationship_id for r in rels2}
        assert ids1 == ids2

    def test_deterministic_across_runs(self) -> None:
        """Same inputs produce identical relationship outputs."""
        form_bbox = _bbox(left=0.1, top=0.3, right=0.5, bottom=0.4)
        text_bbox = _bbox(left=0.1, top=0.18, right=0.5, bottom=0.28)
        formula = _make_formula(
            elem_id="ffffffff-ffff-ffff-ffff-fffffffffff4",
            page_num=1,
            reading_order=1,
            bbox=form_bbox,
            latex=r"a^2 + b^2 = c^2",
        )
        text = _make_text_block(
            elem_id="ffffffff-ffff-ffff-ffff-fffffffffff5",
            content="The Pythagorean theorem.",
            page_num=1,
            reading_order=0,
            bbox=text_bbox,
        )
        registry = _build_registry([formula, text])

        rels_a = generate_formula_relationships(formula, registry)
        rels_b = generate_formula_relationships(formula, registry)

        assert len(rels_a) == len(rels_b)
        for ra, rb in zip(rels_a, rels_b):
            assert ra.relationship_id == rb.relationship_id
            assert ra.source_id == rb.source_id
            assert ra.target_id == rb.target_id
            assert ra.relationship_type == rb.relationship_type


# ===================================================================
#  10.  process_formulas — document-wide helper
# ===================================================================


class TestProcessFormulas:
    """Document-wide formula processing."""

    def test_processes_all_formulas(self) -> None:
        """All formula elements are processed."""
        doc = _make_doc(page_count=2)
        f1 = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999901",
            page_num=1,
            reading_order=0,
            latex=r"E = mc^2",
        )
        f2 = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999902",
            page_num=2,
            reading_order=0,
            latex=r"\frac{a}{b}",
        )
        elements = {
            str(f1.element_id): f1,
            str(f2.element_id): f2,
        }
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([f1, f2])

        result = process_formulas(doc, registry)

        for elem in result.elements.values():
            assert isinstance(elem, FormulaSchema)
            assert elem.text_approximation != ""
            assert elem.formula_type in ("inline", "display")
            assert isinstance(elem.variables, list)

    def test_preserves_non_formula_elements(self) -> None:
        """Non-formula elements are left unchanged."""
        doc = _make_doc()
        formula = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999911",
            page_num=1,
            latex=r"x = 1",
        )
        text = _make_text_block(
            elem_id="99999999-9999-9999-9999-999999999912",
            content="Some text.",
            page_num=1,
        )
        elements = {
            str(formula.element_id): formula,
            str(text.element_id): text,
        }
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([formula, text])

        result = process_formulas(doc, registry)

        assert str(formula.element_id) in result.elements
        assert str(text.element_id) in result.elements
        text_elem = result.elements[str(text.element_id)]
        assert text_elem.content == "Some text."
        assert isinstance(text_elem, TextBlockSchema)

    def test_empty_document(self) -> None:
        """Empty document does not crash."""
        doc = _make_doc()
        registry = _build_registry([])
        result = process_formulas(doc, registry)
        assert len(result.elements) == 0

    def test_no_formulas_no_changes(self) -> None:
        """Document with no formula elements passes through."""
        doc = _make_doc()
        text = _make_text_block(
            elem_id="99999999-9999-9999-9999-999999999921",
            content="No formulas here.",
            page_num=1,
        )
        elements = {str(text.element_id): text}
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([text])

        result = process_formulas(doc, registry)

        assert len(result.elements) == 1
        assert str(text.element_id) in result.elements

    def test_no_duplicate_registry_entries(self) -> None:
        """process_formulas does not create duplicate page/type entries."""
        doc = _make_doc(page_count=1)
        formula = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999930",
            page_num=1,
            reading_order=0,
            latex=r"E = mc^2",
        )
        elements = {str(formula.element_id): formula}
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([formula])

        # Capture counts before processing
        before_page = len(registry.get_by_page(1))
        before_type = len(registry.get_by_type("formula"))

        process_formulas(doc, registry)

        # After processing, the formula should still appear exactly once
        assert len(registry.get_by_page(1)) == before_page
        assert len(registry.get_by_type("formula")) == before_type

    def test_no_duplicate_relationships_across_formulas(self) -> None:
        """Multiple formulas produce no duplicate relationship IDs."""
        doc = _make_doc(page_count=1)
        f1 = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999931",
            page_num=1,
            reading_order=1,
            bbox=_bbox(left=0.1, top=0.3, right=0.4, bottom=0.4),
            latex=r"E = mc^2",
        )
        f2 = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999932",
            page_num=1,
            reading_order=3,
            bbox=_bbox(left=0.6, top=0.3, right=0.9, bottom=0.4),
            latex=r"\int f(x) dx",
        )
        text = _make_text_block(
            elem_id="99999999-9999-9999-9999-999999999933",
            content="This is about Equation 1.",
            page_num=1,
            reading_order=0,
            bbox=_bbox(left=0.1, top=0.15, right=0.5, bottom=0.28),
        )
        elements = {
            str(f1.element_id): f1,
            str(f2.element_id): f2,
            str(text.element_id): text,
        }
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([f1, f2, text])

        result = process_formulas(doc, registry)

        rel_ids = [r.relationship_id for r in result.relationships]
        assert len(rel_ids) == len(set(rel_ids)), "Duplicate relationship IDs found"

    def test_preserves_existing_relationships(self) -> None:
        """Existing relationships not involving formulas are preserved."""
        doc = _make_doc(page_count=1)
        formula = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999941",
            page_num=1,
            latex=r"x = 1",
        )
        text = _make_text_block(
            elem_id="99999999-9999-9999-9999-999999999942",
            content="About equation 1.",
            page_num=1,
            reading_order=0,
            bbox=_bbox(left=0.1, top=0.15, right=0.5, bottom=0.28),
        )

        # Add a pre-existing relationship
        existing_rel = RelationshipSchema(
            relationship_id=uuid.uuid5(uuid.NAMESPACE_DNS, "pre-existing-rel"),
            source_id=text.element_id,
            target_id=formula.element_id,
            relationship_type="explains",
            metadata={"page_num": 1},
            weight=1.0,
        )

        elements = {
            str(formula.element_id): formula,
            str(text.element_id): text,
        }
        doc = doc.model_copy(
            update={
                "elements": elements,
                "relationships": [existing_rel],
            }
        )
        registry = _build_registry([formula, text])

        result = process_formulas(doc, registry)

        # The existing relationship should still be present
        assert existing_rel.relationship_id in {
            r.relationship_id for r in result.relationships
        }

    def test_deterministic_across_runs(self) -> None:
        """process_formulas produces identical output on repeated calls."""
        doc = _make_doc(page_count=1)
        formula = _make_formula(
            elem_id="99999999-9999-9999-9999-999999999951",
            page_num=1,
            reading_order=0,
            latex=r"a + b = c",
        )
        elements = {str(formula.element_id): formula}
        doc = doc.model_copy(update={"elements": elements})
        registry = _build_registry([formula])

        result1 = process_formulas(doc, registry)
        result2 = process_formulas(doc, registry)

        # Compare element latex
        for key in elements:
            e1 = result1.elements[key]
            e2 = result2.elements[key]
            assert isinstance(e1, FormulaSchema) and isinstance(e2, FormulaSchema)
            assert e1.latex == e2.latex
            assert e1.text_approximation == e2.text_approximation
            assert e1.formula_type == e2.formula_type
            assert e1.variables == e2.variables

        # Compare relationship IDs
        rel_ids1 = {r.relationship_id for r in result1.relationships}
        rel_ids2 = {r.relationship_id for r in result2.relationships}
        assert rel_ids1 == rel_ids2


# ===================================================================
#  11.  Process formula with dl_doc as source
# ===================================================================


class TestProcessFormulaWithDlDoc:
    """Formula processing via dl_doc data source."""

    def test_dl_doc_latex_field(self) -> None:
        """dl_doc with explicit 'latex' field."""
        elem = _make_formula(latex="", content="")
        dl_doc = {"latex": r"x^3 + y^3 = z^3"}
        result = process_formula(elem, dl_doc=dl_doc)
        assert "x^3" in result.latex

    def test_dl_doc_text_fallback(self) -> None:
        """dl_doc with 'text' field (alternative latex field)."""
        elem = _make_formula(latex="", content="")
        dl_doc = {"text": r"\sin^2 x + \cos^2 x = 1"}
        result = process_formula(elem, dl_doc=dl_doc)
        assert "sin" in result.latex

    def test_dl_doc_formula_type_field(self) -> None:
        """dl_doc with explicit formula_type."""
        elem = _make_formula(latex=r"x")
        dl_doc = {"formula_type": "inline"}
        result = process_formula(elem, dl_doc=dl_doc)
        assert result.formula_type == "inline"

    def test_dl_doc_inline_flag(self) -> None:
        """dl_doc with boolean inline flag."""
        elem = _make_formula(latex=r"x")
        dl_doc = {"inline": True}
        result = process_formula(elem, dl_doc=dl_doc)
        assert result.formula_type == "inline"

    def test_dl_doc_display_flag(self) -> None:
        """dl_doc with boolean display flag."""
        elem = _make_formula(latex=r"x")
        dl_doc = {"display": True}
        result = process_formula(elem, dl_doc=dl_doc)
        assert result.formula_type == "display"
