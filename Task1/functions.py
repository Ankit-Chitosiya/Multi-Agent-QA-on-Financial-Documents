import os
import numpy as np
import requests
import pdfplumber
import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import base64
import pytesseract
from PIL import Image
import io, base64


def download_pdf(url: str, save_dir="pdfs") -> str:
    print("Downloading pdf")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
    return filename



def extract_pdf_content(pdf_path: str) -> Dict:
    print("Extracting pdf content")
    content = {"text": [], "tables": [], "figures": []}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Text
            text = page.extract_text()
            if text:
                content["text"].append({"page": page_num, "content": text})

            # Tables
            tables = page.extract_tables()
            for t_id, table in enumerate(tables):
                table_str = "\n".join([", ".join([cell or "" for cell in row]) for row in table])
                content["tables"].append({
                    "page": page_num,
                    "table_id": t_id,
                    "content": table_str
                })

            # Figures (basic: extract image bytes)
            images = page.images
            for f_id, img in enumerate(images):
                x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                # crop the image region
                cropped = page.within_bbox((x0, top, x1, bottom)).to_image(resolution=150)
                img_bytes = cropped.original.save("tmp.png")  # hack, can be improved
                with open("tmp.png", "rb") as imf:
                    b64 = base64.b64encode(imf.read()).decode("utf-8")
                content["figures"].append({
                    "page": page_num,
                    "figure_id": f_id,
                    "content": b64  # stored as base64
                })

    return content



def chunk_content(content: Dict, chunk_size=1000, chunk_overlap=100) -> List[Dict]:
    print("Creating Chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    # Text → split
    for t in content["text"]:
        split_texts = splitter.split_text(t["content"])
        for j, c in enumerate(split_texts):
            chunks.append({
                "content": c,
                "modality": "text",
                "page": t["page"]
            })

    # Tables → keep whole
    for tbl in content["tables"]:
        chunks.append({
            "content": tbl["content"],
            "modality": "table",
            "page": tbl["page"]
        })

    # Figures → keep as single chunk
    for fig in content["figures"]:
        chunks.append({
            "content": fig["content"],
            "modality": "figure",
            "page": fig["page"]
        })

    return chunks



# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain

# def generate_chunk_heading(llm, chunk: str) -> str:
    
#     # Define prompt
#     prompt = PromptTemplate(
#         input_variables=["chunk"],
#         template=(
#             "You are given a piece of text from a financial document.\n"
#             "Generate a very concise heading (maximum 10 words) "
#             "that best summarizes it.\n\n"
#             "Text:\n{chunk}\n\n"
#             "Heading:"
#         )
#     )

   
#     chain = LLMChain(llm=llm, prompt=prompt)

    
#     heading = chain.run({"chunk": chunk}).strip()

#     enriched_chunk = f"Heading: {heading} | Content: {chunk}"
#     return enriched_chunk

def extract_text_from_image(base64_image) -> str:
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    text = pytesseract.image_to_string(image)
    return text.strip()


def embed_chunks(chunks: List[Dict], model) -> List[Dict]:
    print("Generating embeddings")
    contents = []
    for ch in chunks:
        if ch["modality"] == "figure":
            content=extract_text_from_image(ch["content"])
            contents.append(content)
        else:
            contents.append(ch["content"])

    vectors = model.encode(contents, convert_to_numpy=True)

    # attach embeddings back
    for i, emb in enumerate(vectors):
        chunks[i]["embedding"] = emb
    return chunks


def build_faiss_index(
    chunks: List[Dict], dim: int,
    faiss_path="/Multi-Agent-QA-on-Financial-Documents/Task1_result/finance.index", # File path for saving
    meta_path="//Multi-Agent-QA-on-Financial-Documents/Task1_result/chunks.jsonl"
):
    print("Storing in faiss")

    index = faiss.IndexFlatL2(dim)
    embeddings = [c["embedding"] for c in chunks]
    index.add(np.array(embeddings, dtype="float32"))

    # Save FAISS index
    faiss.write_index(index, faiss_path)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            meta = {k: v for k, v in c.items() if k != "embedding"}
            f.write(json.dumps(meta) + "\n")

    return index

print("Module loaded: functions.py, All funcions are ready to use.")