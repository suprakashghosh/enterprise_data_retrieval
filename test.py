
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
from src.chunking.cross_reference_resolver import resolve_cross_references

# %load_ext autoreload
# %autoreload 2
chunk_metadatas, text_lookup, pic_table_lookup= build_chunk_metadata_list(conv_result= conv_result, output_dir= output_dir)
chunk_metadatas_with_refers_to= resolve_cross_references(chunk_metadatas=chunk_metadatas,
                                                        pic_table_lookup= pic_table_lookup,
                                                        text_lookup= text_lookup,
                                                        document_name= conv_result.document.origin.filename,
                                                        output_dir= output_dir)

#Build multimodal embeddings
from src.multimodal_embeddings.multimodal_embeddings import model
from src.retrieval.embedding_pipeline import (
    attach_embeddings,
    build_encode_items,
    encode_batch,
)

items = build_encode_items(chunk_metadatas)
embeddings_list = encode_batch(items, model, batch_size=32)
docs = attach_embeddings(chunk_metadatas, embeddings_list,document_name= conv_result.document.origin.filename,
output_dir= output_dir)

#Populate refers to
import numpy as np

from src.retrieval.similarity import (
    compute_cosine_similarity_matrix,
    populate_relates_to,
)

embeddings_2d = np.stack(embeddings_list)          # np.ndarray (n, d)
enriched = populate_relates_to(chunk_metadatas_with_refers_to, embeddings_2d, top_k=3,document_name= conv_result.document.origin.filename,
output_dir= output_dir)
