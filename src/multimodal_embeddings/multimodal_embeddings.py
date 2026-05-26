from sentence_transformers import SentenceTransformer
from torch import tensor

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

# Encode images
# col_labels = [
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
# ]
col_labels= [
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/pictures/picture_0.png',
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/pictures/picture_1.png',
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/pictures/picture_2.png',
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/pictures/picture_3.png',
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/pictures/picture_4.png',
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/tables/table_0.png',
    '/home/suprakashg/Downloads/Projects/enterprise_data_retrieval/data/raw/879e2871-78e8-501e-863c-1ac36673950c/docling/tables/table_1.png'
]
img_embeddings = model.encode(col_labels)

# Encode text queries (one matching + one hard negative per image)
row_labels = [
    "LLM Mind Map",
    'The overall workflow of Agentic Reasoning.',
    "mathematical formula",
    'line charts of pass rate vs max tool calls on a domain wise basis',
    'line chart',
    ' More calling for agentic tools, the better the model does. Red line denotes Gemini Deep Research',
    "a chat between users",
    "logical reasoning fallacy by AI chatbot",
    'mind map',
    'graphical representation of interactions between players',
    'tables',
    'agentic reasoning figures by models',
    'Table 1: Performance comparison on GPQA dataset across Physics, Chemistry, and Biology.',
    'human experts via AI models',
    'Performance comparison with human experts on the GPQA extended set.'
    ''

]
text_embeddings = model.encode(row_labels)

# Compute cross-modal similarities
similarities = model.similarity(text_embeddings, img_embeddings)
# similarities= tensor([[0.5115, 0.1078],
#         [0.1999, 0.1108],
#         [0.1255, 0.6749],
#         [0.1283, 0.2704]])


import pandas as pd

similarity_df = pd.DataFrame(similarities.numpy(), index=row_labels, columns=col_labels)
# print(similarity_df.to_markdown())
similarity_df.to_csv("data/raw/879e2871-78e8-501e-863c-1ac36673950c/query vs image similarity.csv")

# print("Similarities (row: text query, column: image):")
# # Print column header
# header = "{:<55s}".format("") + "".join("{:>12s}".format(c) for c in col_labels)
# print(header)
# for i, row in enumerate(row_labels):
#     line = "{:<55s}".format(row[:52])
#     for j in range(similarities.shape[1]):
#         line += "{:>12.4f}".format(similarities[i, j].item())
#     print(line)
