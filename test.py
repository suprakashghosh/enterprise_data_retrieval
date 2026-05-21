
import datetime
import json
import logging
import time
from pathlib import Path

import pandas as pd
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    granite_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash

_log = logging.getLogger(__name__)


# def main():
logging.basicConfig(level=logging.INFO)

# data_folder = Path(__file__).parent / "../../tests/data"
# input_doc_path = data_folder / "pdf/2206.01062.pdf"
input_doc_path= 'data/original/2502.04644v1.pdf'
###########################################################################

# The sections below demo combinations of PdfPipelineOptions and backends.
# Tip: Uncomment exactly one section at a time to compare outputs.

# PyPdfium without EasyOCR
# --------------------
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = False
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=False)

# PyPdfium with EasyOCR
# -----------------
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)

# Docling Parse without EasyOCR
# -------------------------
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = False
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)


# Docling Parse with EasyOCR (default)
# -------------------------------
# Enables OCR and table structure with EasyOCR, using automatic device
# selection via AcceleratorOptions. Adjust languages as needed.
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options = TableStructureOptions(
    do_cell_matching=True
)
pipeline_options.ocr_options.lang = ["es"]
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.AUTO
)
pipeline_options.do_code_enrichment = True
pipeline_options.do_formula_enrichment = True
pipeline_options.generate_picture_images = True
pipeline_options.generate_page_images = True
pipeline_options.images_scale = 2
pipeline_options.do_picture_classification = True
pipeline_options.do_picture_description = True


pipeline_options.picture_description_options = granite_picture_description
# pipeline_options.code_formula_options

# Docling Parse with EasyOCR (CPU only)
# -------------------------------------
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True
# pipeline_options.ocr_options.use_gpu = False  # <-- set this.
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)


# Docling Parse with Tesseract
# ----------------------------
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
# pipeline_options.ocr_options = TesseractOcrOptions()

# Docling Parse with Tesseract CLI
# --------------------------------
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
# pipeline_options.ocr_options = TesseractCliOcrOptions()

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


###########################################################################

start_time = time.time()
conv_result = doc_converter.convert(input_doc_path)
end_time = time.time() - start_time

_log.info(f"Document converted in {end_time:.2f} seconds.")

## Export results
output_dir = Path("Scratch")
output_dir.mkdir(parents=True, exist_ok=True)
doc_filename = conv_result.input.file.stem

# Export Docling document JSON format:
with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
    fp.write(json.dumps(conv_result.document.export_to_dict()))

# Export Text format (plain text via Markdown export):
with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
    fp.write(conv_result.document.export_to_markdown(strict_text=True))

# Export Markdown format:
with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
    fp.write(conv_result.document.export_to_markdown())

# Export Document Tags format:
with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
    fp.write(conv_result.document.export_to_doctags())

rows = []
for (
    content_text,
    content_md,
    content_dt,
    page_cells,
    page_segments,
    page,
) in generate_multimodal_pages(conv_result):
    dpi = page._default_image_scale * 72

    rows.append(
        {
            "document": conv_result.input.file.name,
            "hash": conv_result.input.document_hash,
            "page_hash": create_hash(
                conv_result.input.document_hash + ":" + str(page.page_no - 1)
            ),
            "image": {
                "width": page.image.width,
                "height": page.image.height,
                "bytes": page.image.tobytes(),
            },
            "cells": page_cells,
            "contents": content_text,
            "contents_md": content_md,
            "contents_dt": content_dt,
            "segments": page_segments,
            "extra": {
                "page_num": page.page_no + 1,
                "width_in_points": page.size.width,
                "height_in_points": page.size.height,
                "dpi": dpi,
            },
        }
    )

# Generate one parquet from all documents
df_result = pd.json_normalize(rows)
now = datetime.datetime.now()
output_filename = output_dir / f"multimodal_{now:%Y-%m-%d_%H%M%S}.parquet"
df_result.to_parquet(output_filename)

end_time = time.time() - start_time

_log.info(
    f"Document converted and multimodal pages generated in {end_time:.2f} seconds."
)
# if __name__ == "__main__":
#     main()

for element, _level in conv_result.document.iterate_items():
    print(element)

#%%
# df_result= pd.read_parquet('Scratch/multimodal_2026-05-21_162607.parquet')
# print(df_result)
