import re
from typing import Optional, Tuple

from docling_core.types.doc import CodeItem, FormulaItem

# ---------------------------------------------------------------------------
# Caption-label extraction
# ---------------------------------------------------------------------------

# Conditional imports for interactive / notebook use
try:
    from src.extraction.docling_extractor import run_extraction_pipeline
    from src.ingestion.ingestor import ingest_pdf
    from src.validation.docling_validator import validate_docling_output
except ImportError:
    run_extraction_pipeline = None  # type: ignore[assignment]
    ingest_pdf = None  # type: ignore[assignment]
    validate_docling_output = None  # type: ignore[assignment]

from src.extraction.docling_extractor import run_extraction_pipeline
from src.ingestion.ingestor import ingest_pdf
from src.validation.docling_validator import validate_docling_output

source_path = "data/original/2502.04644v1.pdf"
pdf_schema = ingest_pdf(source=source_path)
dl_doc, doc = run_extraction_pipeline(pdf_schema)
validation_report = validate_docling_output(dl_doc, doc)


# %%
from src.normalization.docling_normalizer import normalize_document

normalized_doc = normalize_document(doc, dl_doc)
# %%

# %%
# Matching caption to pictures
picture_objects = getattr(dl_doc, "pictures", [])
text_objects = getattr(dl_doc, "texts", [])
table_objects = getattr(dl_doc, "tables", [])
for picture_object in picture_objects + table_objects:
    # for key, value in picture_object.iter_items():

    caption_text_reference = None
    captions = getattr(picture_object, "captions", [])
    if len(captions) == 1:
        # print(captions[0])
        caption_text_reference = getattr(captions[0], "cref")
    else:
        children = getattr(picture_object, "children", [])
        if len(children) > 0:
            caption_text_reference = getattr(children[0], "cref")
    if caption_text_reference:
        matched_text_object = next(
            (
                t
                for t in text_objects
                if getattr(t, "self_ref") == caption_text_reference
            ),
            None,
        )
        print(
            getattr(picture_object, "self_ref"),
            getattr(matched_text_object, "text", ""),
        )


# %%
# validation_report
# obj = getattr(extracted_pdf_doc, "pictures", [])
# obj = getattr(extracted_pdf_doc, "tables", [])
# obj = getattr(doc, "key_value_items", [])
# Print extracted formulas
#
#
# code_blocks = [
#         item for item, _ in doc.iterate_items() if isinstance(item, CodeItem)
#     ]
# print(f"Code blocks found: {len(code_blocks)}")
# for i, item in enumerate(code_blocks, 1):
#     print(f"\n  Code block {i}:")
#     print(f"    Language: {item.code_language}")
#     print(f"    Text: {item.text[:100]}{'...' if len(item.text) > 100 else ''}")

# formulas = [
#     item for item, _ in doc.iterate_items() if isinstance(item, FormulaItem)
# ]
# print(f"\nFormulas found: {len(formulas)}")
# for i, item in enumerate(formulas, 1):
#     print(f"\n  Formula {i}:")
#     print(f"    Text: {item.text[:100]}{'...' if len(item.text) > 100 else ''}")
#
# print(extracted_pdf_doc)
