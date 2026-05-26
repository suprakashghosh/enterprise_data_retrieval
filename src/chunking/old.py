EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300  # set to a small number for illustrative purposes

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,  # optional, defaults to True
)
chunk_iter = chunker.chunk(dl_doc=conv_result.document)



text_objects = getattr(conv_result.document, "texts", [])
text_lookup = {
    getattr(t, "self_ref"): getattr(t, "text", "")
    for t in text_objects
    if getattr(t, "self_ref", None) is not None
}


picture_objects = getattr(conv_result.document, "pictures", [])
table_objects = getattr(conv_result.document, "tables", [])
picture_and_table_object_lookup= {
    getattr(t, "self_ref"): t
    for t in picture_objects+table_objects
    if getattr(t, "self_ref", None) is not None
}
picture_and_table_object_section_lookup= {
    getattr(t, "self_ref"): []
    for t in picture_objects+table_objects
    if getattr(t, "self_ref", None) is not None
} #initialize section mapping with blank section for each ref

class ChunkMetadata(BaseModel):
    # embedding: List[float]
    embedding_type: str #Can be text, image or textual_description_of_image
    document_name: str
    document_type: str
    # page_number: int
    chunk_text: str #In case of image embeddings, this will be the caption text
    chunk_types: List[str] #Within meta.doc_items, get the label key
    section_name: List[str] #Within meta, headings
    # section_number: str
    sequence_number: int
    image_type: Optional[str] #NA, Picture, Table
    image_uri: Optional[str] #Within meta.doc_items, if self_ref contains #/pictures/ or if the label is of the type picture
    # image_caption: Optional[str]
    # image_number: Optional[str] #The image number extracted from the caption
    # relationships: List[str] #Any image references or other chunks with high semantic similarity. Language such as 'As explained in image xyz' or 'As explained in section xyz'
    refers_to: List[str] #A → B | Element A references element B (e.g., "see Table 3") |
    relates_to: List[str] #A ↔ B | General semantic relatedness (undirected in data, stored as symmetric) |
    # keywords: List[str]


#There are three types of chunks

# Write chunks to markdown file


chunk_json_list= []
chunk_metadatas= []
items_to_encode= []
# Relationships not yet extracted; initialize as empty list
refers_to = []
relates_to= []

chunks_file = output_dir / f"{doc_filename}_chunks.md"
document_name= conv_result.document.origin.filename
document_type = conv_result.document.origin.mimetype
with chunks_file.open("w", encoding="utf-8") as md_file:
    for i, chunk in enumerate(chunk_iter):


        single_chunk_model=SingleChunkModel(**chunk.export_json_dict())
        # document_name= single_chunk_model.meta.origin.filename
        chunk_text = chunker.contextualize(chunk=chunk)
        items_to_encode.append(chunk_text)
        embedding_type= 'Text'
        section_name= single_chunk_model.meta.headings
        sequence_number= i
        doc_items= single_chunk_model.meta.doc_items
        chunk_types= []
        image_label= 'NA'
        image_uri= ''
        image_type='NA'
        for item in doc_items:
            chunk_types.append(item.label)
            if item.label in ['picture', 'table']:
                ref= item.self_ref
                image_type= item.label
                image_uri= picture_and_table_object_lookup[ref].image.uri
                picture_and_table_object_section_lookup[ref].extend(section_name) #picture_and_table_object_section_lookup values are lists.



        # Create the Chunk object and append to list
        chunk_metadata_obj = ChunkMetadata(
            embedding_type=embedding_type,
            document_name=document_name,
            document_type= document_type,
            chunk_text=chunk_text,
            chunk_types=chunk_types,
            section_name=section_name,
            sequence_number=sequence_number,
            image_type=image_type,
            image_uri=image_uri,
            refers_to = refers_to,
            relates_to= relates_to
        )
        chunk_metadatas.append(chunk_metadata_obj)

        md_file.write("#"*100)
        md_file.write("\n")
        md_file.write(str(chunk.export_json_dict()))
        chunk_json_list.append(chunk.export_json_dict())
        md_file.write(f"\n*## Chunk {i}*\n\n")
        md_file.write("**chunk.text:**\n\n```\n")
        md_file.write(chunk.text)
        md_file.write("\n```\n\n")
        md_file.write("**chunker.contextualize(chunk):**\n\n```\n")
        md_file.write(chunk_text)
        md_file.write("\n```\n\n---\n\n")
        md_file.write("#"*100)

json_file= output_dir / f"{doc_filename}_chunks.json"
with json_file.open("w", encoding="utf-8") as json_file:
    json.dump(chunk_json_list, json_file, indent=2)

# Write ChunkMetadata objects to a JSON file
chunks_metadata_file = output_dir / f"{doc_filename}_chunks_metadata.json"
with chunks_metadata_file.open("w", encoding="utf-8") as f:
    json.dump([chunk.model_dump() for chunk in chunk_metadatas], f, indent=2)





def get_caption_for_picture(ref):
    object= picture_and_table_object_lookup[ref]
    # 2. Extract the reference using a clean fallback chain
    caption_text_reference = None

    # Get the collections safely
    captions = getattr(object, "captions", [])
    children = getattr(object, "children", [])

    # Prioritize captions, fall back to children
    if captions:
        # If you only ever care about the first one, extract it cleanly
        caption_text_reference = getattr(captions[0], "cref", None)
    elif children:
        caption_text_reference = getattr(children[0], "cref", None)

    # 3. Direct O(1) assignment
    caption_text = text_lookup.get(caption_text_reference, "")
    return caption_text




async def get_description_and_keywords_for_image(image_url, image_caption):

    class ImageDescription(BaseModel):
        """
        You are an experienced image annotator.
        You will be provided with an image from a document.
        The image can be a picture. chart, graph, table, formula etc.
        This image has to be indexed for a retrieval augmented generation system.
        Your role is to provide a detailed description of the image which will then be converted into an embedding vector for semantic retrieval.
        You should also provide all keywords which can be used to identify the image for a BM25 retriever system.
        """
        description: str = Field(..., description= 'A detailed description of the image which captures all of its main features and facets.')
        keywords: List[str]= Field(..., description= 'A list of the keywords which can be used to describe or identify the image.')

    system_prompt= 'You are an experienced image annotator. You will be provided with an image. You must accurately provide the details.'
    user_input= f'Please find the image attached. The image caption is: {image_caption}.'
    response= get_llm_response_from_instructor(user_input= user_input,
                                                system_prompt=system_prompt,
                                                response_format= ImageDescription,
                                                image_url=image_url
    )
    return f"Image Caption: {image_caption}\n Image Description: {response.description}\n Keywords: {response.keywords}"

def get_textual_description_of_picture_or_image_chunk(ref):
    caption_text= get_caption_for_picture(ref)
    # image_number= extract_caption_label(text= caption_text)
    image_uri= picture_and_table_object_lookup[ref].image.uri
    image_description_and_keywords= asyncio.run(get_description_and_keywords_for_image(image_url=image_uri, image_caption=caption_text))
    return image_description_and_keywords

sequence_number=i
for ref, obj in picture_and_table_object_lookup.items():

    #Get the common items which will repeat across all chunk types
    chunk_types= [obj.label]
    section_name=picture_and_table_object_section_lookup[ref]
    image_type= obj.label
    image_uri= obj.image.uri
    relationships= []

    #First process the objects for embedding type 'Image' where the image itself is encoded
    embedding_type='image'
    chunk_text= get_caption_for_picture(ref) #When encoding the image directly, we will leave the chunk text as the caption for the image
    sequence_number=sequence_number+1
    items_to_encode.append(image_uri)
    chunk_metadata_obj = ChunkMetadata(
        embedding_type=embedding_type,
        document_name=document_name,
        document_type= document_type,
        chunk_text=chunk_text,
        chunk_types=chunk_types,
        section_name=section_name,
        sequence_number=sequence_number,
        image_type=image_type,
        image_uri=image_uri,
        refers_to = refers_to,
        relates_to= relates_to
    )
    chunk_metadatas.append(chunk_metadata_obj)

    #Then process the objects for embedding type 'textual_description_of_image' where the textual description is generated and that is encoded.
    embedding_type= 'textual_description_of_image'
    chunk_text=get_textual_description_of_picture_or_image_chunk(ref=ref)
    sequence_number=sequence_number+1
    items_to_encode.append(chunk_text)
    chunk_metadata_obj = ChunkMetadata(
        embedding_type=embedding_type,
        document_name=document_name,
        document_type= document_type,
        chunk_text=chunk_text,
        chunk_types=chunk_types,
        section_name=section_name,
        sequence_number=sequence_number,
        image_type=image_type,
        image_uri=image_uri,
        refers_to = refers_to,
        relates_to= relates_to
    )
    chunk_metadatas.append(chunk_metadata_obj)
