"""
``src.normalization`` — Normalisation, element registry, hierarchy
reconstruction, table processing, and formula processing.

Converts raw Docling output into the project's internal schema,
preserves page numbers, bounding boxes, reading order, section
assignments, captions, and layout proximity.  Reconstructs the
document hierarchy and assigns section paths to every element.
Enriches table elements with structured representations (markdown,
HTML, JSON, plain-text summary), detects multi-page spanning
tables, and generates table-related relationships.
Enriches formula elements with LaTeX, text approximation,
inline/display classification, variable extraction, and
formula-specific relationships.

Public API
----------
::

    from src.normalization import (
        DOCLING_TYPE_TO_INTERNAL_TYPE,
        ElementRegistry,
        assign_section_paths,
        build_hierarchy,
        detect_spanning_tables,
        generate_formula_relationships,
        generate_table_relationships,
        normalize_document,
        preserve_proximity,
        process_formula,
        process_formulas,
        process_table,
        process_tables,
    )
"""

from src.normalization.docling_normalizer import (
    DOCLING_TYPE_TO_INTERNAL_TYPE,
    ElementRegistry,
    normalize_document,
    preserve_proximity,
)
from src.normalization.formula_processor import (
    generate_formula_relationships,
    process_formula,
    process_formulas,
)
from src.normalization.hierarchy_builder import (
    assign_section_paths,
    build_hierarchy,
)
from src.normalization.table_processor import (
    detect_spanning_tables,
    generate_table_relationships,
    process_table,
    process_tables,
)

__all__: list[str] = [
    "DOCLING_TYPE_TO_INTERNAL_TYPE",
    "ElementRegistry",
    "assign_section_paths",
    "build_hierarchy",
    "detect_spanning_tables",
    "generate_formula_relationships",
    "generate_table_relationships",
    "normalize_document",
    "preserve_proximity",
    "process_formula",
    "process_formulas",
    "process_table",
    "process_tables",
]
