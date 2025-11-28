from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from typing import List
from sentence_transformers import SentenceTransformer

from functions import download_pdf, extract_pdf_content, chunk_content, embed_chunks, build_faiss_index
# hf_pipeline = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base"
# )

#llm = HuggingFacePipeline(pipeline=hf_pipeline)


def pipeline(doc_links: List[str]):
    "Initiating pipeline"
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    all_chunks = []

    for url in doc_links:
        pdf_path = download_pdf(url)
        content = extract_pdf_content(pdf_path)
        chunks = chunk_content(content)       
        em_chunks = embed_chunks(chunks, model)
        all_chunks.extend(em_chunks)

    print(f"Finished pipeline. Stored {len(all_chunks)} chunks embeddings")
    return all_chunks