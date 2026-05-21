from docling_core.types.doc import CodeItem, FormulaItem

from src.extraction.docling_extractor import run_extraction_pipeline
from src.ingestion.ingestor import ingest_pdf

from src.validation.docling_validator import validate_docling_output

source_path= 'data/original/2502.04644v1.pdf'
pdf_schema= ingest_pdf(source=source_path)
dl_doc, doc= run_extraction_pipeline(pdf_schema)
validation_report = validate_docling_output(dl_doc, doc)


#%%
from src.normalization.docling_normalizer import normalize_document
normalized_doc= normalize_document(doc, dl_doc)
#%%

#%%

#%%

#%%
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
