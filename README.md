# Multi-Agent QA on Financial Documents

_From Raw Financial Reports to Structured, Reasoned Answers_

This repository contains **Task 1** and **Task 2** of a financial document analysis project, implemented in Python and Jupyter notebooks. The project focuses on extracting structured financial data from documents analyzing and enhancing answers using web search.

---

## Project Overview

The project is divided into two main tasks:

### Task 1 – Financial Document Chunking and Storage
- Stores embeddings of financial documents in **FAISS**.  
- Allows retrieval of relevant document chunks using **semantic search**.  
- Supports structured extraction of **text, tables, figures, and financial metrics**.

### Task 2 – Dynamic Multi-Agent Query Answering
- Fully dynamic multi-agent system for processing queries.  
- Uses **OpenAI LLMs** for high-quality responses and structured output.
- Check States before calling any Agent.
- Integrates **SERP API** to fetch data from the web.
- Maintain Trace(agent calling, work history) throughout the Pipeline.
- Combines local document knowledge (Task 1) with web search for comprehensive answers.

---

## Usage

1. Open the Kaggle notebook: `task1_task2.ipynb`.

2. **Task 1 – Financial Document Chunking and Storage**
   - Run the cells sequentially to make chunks and store embeddings in FAISS.  
   - Structured table and financial metric extraction will be performed automatically.

3. **Task 2 – Dynamic Multi-Agent System**
   - Run the cells sequentially to define all agents and build the pipeline graph.  
   - Ensure the required API keys are set (see below).

---

## API Keys

**Important:** Task 2 requires API keys. Without them, the LLMs and web search agent will fail.

- `OPENAI_API_KEY` → Used for embeddings and LLM predictions.  
- `SERPAPI_API_KEY` → Used for web search queries.  

> Add these keys in your environment variables or directly in the notebook (not recommended for public repositories).

---

## 🧠 Tech Stack & Tools

**Languages & Frameworks**

- Python 3.10+

- Jupyter Notebook

- LangChain / LangGraph

**Machine Learning & NLP**

- OpenAI GPT-4o-mini, GPT-3.5-turbo

- SentenceTransformers / OpenAI Embeddings

- HuggingFace Transformers

**Vector Search & Storage**

- FAISS (Facebook AI Similarity Search)

- In-Memory Docstore

**Document Processing**

- pdfplumber (PDF text, table, and figure extraction)

- json, pandas for data handling
  
- OCR for images

**Web Search and APIs**

- SERP API (Web search integration)

- OpenAI API (LLM, embeddings)

**Visualization & Debugging**

- Mermaid (LangGraph flow visualization)

**Environment**

- Kaggle Notebook

### Python Libraries

```bash
pip install langchain langgraph langchain_community faiss-cpu sentence-transformers openai pydantic numpy requests
