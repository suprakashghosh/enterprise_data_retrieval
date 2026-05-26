
#Chunking
import asyncio
import datetime
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Iterable, List, Optional

# from tkinter import image_types
import pandas as pd
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_HERON_101,
    DOCLING_LAYOUT_V2,
    LayoutModelConfig,
)
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    LayoutOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
    granite_picture_description,
    smolvlm_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel
from pydantic.fields import Field
from rich.console import Console
from rich.panel import Panel
from torch import tensor
from transformers import AutoTokenizer

from src.chunking.docling_chunker import build_chunk_metadata_list
from src.multimodal_embeddings.multimodal_embeddings import model
from src.schemas.single_chunk_schema_model import SingleChunkModel
from src.utils.caption_extractor import extract_caption_label
from src.utils.instructor_api_response import get_llm_response_from_instructor

_log = logging.getLogger(__name__)


# def main():
logging.basicConfig(level=logging.INFO)

input_doc_path= 'data/original/2502.04644v1.pdf'
###########################################################################
pipeline_options = PdfPipelineOptions()

#Accelerator options
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.AUTO
)

#Table structure
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options = TableStructureOptions(
    do_cell_matching=True,
    mode=TableFormerMode.ACCURATE,
)
# pipeline_options.table_structure_options.do_cell_matching = True #Same as above
# pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE #Same as above


#OCR
pipeline_options.do_ocr = True
pipeline_options.ocr_options.lang = ['eng',"es"]
# pipeline_options.ocr_options = TesseractOcrOptions()
# pipeline_options.ocr_options = TesseractCliOcrOptions()
#
#Code and formula enrichment and options
pipeline_options.do_code_enrichment = False
pipeline_options.do_formula_enrichment = False
preset_name= 'codeformulav2' #preset_name: Name of the preset to use ('codeformulav2' or 'granite_docling')
pipeline_options.code_formula_options= CodeFormulaVlmOptions.from_preset(preset_name)

#Get pictures
pipeline_options.generate_picture_images = True
pipeline_options.generate_page_images = True
pipeline_options.generate_table_images= True
pipeline_options.images_scale = 2

#Picture classification
pipeline_options.do_picture_classification = True

#Picture description
pipeline_options.do_picture_description = False
pipeline_options.picture_description_options = granite_picture_description #Can also be smolvlm_picture_description

#Layout options
pipeline_options.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_XLARGE)



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

doc_filename = conv_result.input.file.stem
doc_filename_hash = create_hash(doc_filename)
base_output_dir = Path(f'data/raw/{doc_filename_hash}')
output_dir = base_output_dir
suffix = 0
while output_dir.exists():
    suffix += 1
    output_dir = base_output_dir.with_name(f"{base_output_dir.name}_{suffix}")
output_dir.mkdir(parents=True, exist_ok=True)




pickle_path = output_dir / f"{doc_filename}.pkl"
with pickle_path.open("wb") as fp:
    pickle.dump(conv_result.document, fp)

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



# from src.chunking.visual_enricher import enrich_visual_chunks
# %load_ext autoreload
# %autoreload 2
chunks_metadata= build_chunk_metadata_list(conv_result= conv_result, output_dir= output_dir)













# rows = []
# for (
#     content_text,
#     content_md,
#     content_dt,
#     page_cells,
#     page_segments,
#     page,
# ) in generate_multimodal_pages(conv_result):
#     dpi = page._default_image_scale * 72

#     rows.append(
#         {
#             "document": conv_result.input.file.name,
#             "hash": conv_result.input.document_hash,
#             "page_hash": create_hash(
#                 conv_result.input.document_hash + ":" + str(page.page_no - 1)
#             ),
#             "image": {
#                 "width": page.image.width,
#                 "height": page.image.height,
#                 "bytes": page.image.tobytes(),
#             },
#             "cells": page_cells,
#             "contents": content_text,
#             "contents_md": content_md,
#             "contents_dt": content_dt,
#             "segments": page_segments,
#             "extra": {
#                 "page_num": page.page_no + 1,
#                 "width_in_points": page.size.width,
#                 "height_in_points": page.size.height,
#                 "dpi": dpi,
#             },
#         }
#     )

# # Generate one parquet from all documents
# df_result = pd.json_normalize(rows)
# now = datetime.datetime.now()
# output_filename = output_dir / f"multimodal_{now:%Y-%m-%d_%H%M%S}.parquet"
# df_result.to_parquet(output_filename)

# end_time = time.time() - start_time

# _log.info(
#     f"Document converted and multimodal pages generated in {end_time:.2f} seconds."
# )
# if __name__ == "__main__":
#     main()

# for element, _level in conv_result.document.iterate_items():
#     print(element)

#%%
# df_result= pd.read_parquet('Scratch/multimodal_2026-05-21_162607.parquet')
# print(df_result)
