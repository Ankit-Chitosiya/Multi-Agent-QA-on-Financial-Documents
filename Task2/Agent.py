import os
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json
import faiss
import logging
import httpx
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure logging for clearer runtime diagnostics. Import-time downloads may be long;
# logs will help track progress and detect where things block.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
llm_1=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
llm_2=ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

# Defining State
class AgentState(TypedDict):
    query: str
    active_agent: str
    last_agent: Optional[str]=None

    retrieved_chunks: Optional[List[Dict]] = [] 
    retrieved_metadata: Optional[List[Dict]] = []
    
    current_summary: Optional[str] = None     
    
    extracted_entities: Optional[Dict[str, Any]] = None
    reasoning_trace: List[str] = [] 
    
    next_agent: Optional[str] = None            
    final_answer: Optional[str] = None          
    error: Optional[str] = None  
    trace: List[Dict[str, Any]]
    


class ControllerDecision(BaseModel):
    next_agent: str = Field(description="Next agent to call. One of: retriever, table, math, web_search, aggregator.")
    reasoning: str = Field(description="Reasoning for choosing this agent.")

structured_llm_cont = llm_1.with_structured_output(ControllerDecision)

async def ControllerAgent(state: AgentState):
    """
    Hybrid controller that decides next agent:
    Uses deterministic logic when possible.
    Falls back to LLM reasoning only if needed.
    """

    query = state.get("query", "")

    called_agents = [t["agent"] for t in state["trace"] if "agent" in t]
    available_agents = ["retriever", "table", "math", "web_search"]
    remaining_agents = [a for a in available_agents if a not in called_agents]

    retrieved_chunks = state.get("retrieved_metadata", []) or []
    next_agent = None
    reasoning = ""

    # Deterministic Logic 
    if "aggregator" in called_agents:
        # use the END sentinel constant from langgraph
        next_agent = "summarizer"
        reasoning = "Aggregator has completed; summarizer will handle output next."
    
    elif retrieved_chunks and any("table" in c.get("type", "").lower() for c in retrieved_chunks) \
         and "table" not in called_agents and "retriever" in called_agents:
        next_agent = "table"
        reasoning = "Retrieved chunks contain tables, so table agent should handle extraction next."

    elif all(a in called_agents for a in ["retriever","math", "table"]):
        next_agent = "aggregator"
        reasoning = "All major agents have completed their tasks, now aggregator should combine results."

    # LLM logic
    else:
        allowed_agents = remaining_agents + ["aggregator"]
        compact_state = {
            "called_agents": called_agents,
            "retrieved_chunks_count": len(retrieved_chunks),
            "retrieved_metadata": state.get("retrieved_metadata", []),
            "extracted_entities": state.get("extracted_entities", {}),
            "current_summary": state.get("current_summary"),
        }

        prompt = f"""
You are a decision controller in a multi-agent system.

Available agents: {allowed_agents}

Each agent role:
- retriever: fetch relevant text chunks.
- table: interpret data from tables.
- math: perform numerical operations.
- web_search: perform web search releted to query(Select when only neccesaary,retriver results not sufficient).
- aggregator: combine results from previous agents.

Current state:
{json.dumps(compact_state, indent=2)}

User query:
"{query}"

Pick the next logical agent to call. You must choose from the available agents only. If any agent is not in the available agents, you can't choose it.
"""
        decision = await structured_llm_cont.ainvoke(prompt)
        next_agent = decision.next_agent
        reasoning = decision.reasoning
        if next_agent not in allowed_agents:
            next_agent = "aggregator"
            reasoning = f"LLM suggested invalid agent '{decision.next_agent}'; defaulting to aggregator."

    state["reasoning_trace"].append(reasoning)
    state["trace"].append({
        "agent": "controller",
        "decision": next_agent,
        "reasoning": reasoning,
    })
    state["next_agent"] = next_agent

    return state


# Defining Retrieval Agent

# FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", r"C:\Projects\Financial docs\Multi-Agent-QA-on-Financial-Documents\finance.new.index")
# METADATA_PATH = os.getenv("METADATA_PATH", r"C:\Projects\Financial docs\Multi-Agent-QA-on-Financial-Documents\chunks.new.jsonl")

# # Lazy-loaded resources to avoid long downloads at import time.
# # You can set MODEL_LOCAL_ONLY=1 in the environment to prevent network downloads
# # when constructing the sentence-transformer.
# _faiss_index = None
# _metadata_list = None
# _embedding_model = None

# def get_faiss_index():
#     global _faiss_index
#     if _faiss_index is None:
#         try:
#             logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
#             _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
#             logger.info("FAISS index loaded (size=%d).", _faiss_index.ntotal if hasattr(_faiss_index,'ntotal') else -1)
#         except Exception as e:
#             logger.exception("Failed to load FAISS index: %s", e)
#             # Keep None and let callers handle missing index gracefully
#             _faiss_index = None
#     return _faiss_index

# def get_metadata_list():
#     global _metadata_list
#     if _metadata_list is None:
#         try:
#             logger.info("Loading metadata list from %s", METADATA_PATH)
#             with open(METADATA_PATH, "r", encoding="utf-8") as f:
#                 _metadata_list = [json.loads(line) for line in f]
#             logger.info("Loaded %d metadata records.", len(_metadata_list))
#         except FileNotFoundError:
#             logger.exception("Metadata path not found: %s", METADATA_PATH)
#             _metadata_list = []
#         except Exception as e:
#             logger.exception("Failed to read metadata file: %s", e)
#             _metadata_list = []
#     return _metadata_list

# def get_embedding_model():
#     global _embedding_model
#     if _embedding_model is None:
#         try:
#             local_only = os.getenv("MODEL_LOCAL_ONLY", "0") in ("1", "true", "True")
#             # SentenceTransformer supports local_files_only via kwargs depending on version
#             # Using a more defensive approach: if local_only set, create a model only if cached locally
#             logger.info("Initializing embedding model intfloat/multilingual-e5-large (local_only=%s)", local_only)
#             # If the model auto-downloads and you want to avoid network traffic, set MODEL_LOCAL_ONLY=1
#             if local_only:
#                 _embedding_model = SentenceTransformer("intfloat/multilingual-e5-large", local_files_only=True)
#             else:
#                 _embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
#             logger.info("Embedding model initialized.")
#         except Exception as e:
#             logger.exception("Failed to initialize embedding model: %s", e)
#             _embedding_model = None
#     return _embedding_model


# def retrieval_agent(state: AgentState):
#     try:
#         state.setdefault("reasoning_trace", [])
#         state.setdefault("trace", [])

#         query_text = state["query"]
#         state["active_agent"] = "retriever"

#         # Encode query  and normalize
#         emb_model = get_embedding_model()
#         if emb_model is None:
#             raise RuntimeError("Embedding model is not available. Check logs and MODEL_LOCAL_ONLY setting.")

#         query_vector = emb_model.encode(
#             [f"query: {query_text}"],
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         )

#         # Search in FAISS
#         faiss_idx = get_faiss_index()
#         if faiss_idx is None:
#             # No index available -> return safe empty retrieval
#             state.setdefault("reasoning_trace", []).append("FAISS index not available; skipping retrieval.")
#             state["retrieved_chunks"] = []
#             state["retrieved_metadata"] = []
#             state.setdefault("trace", []).append({
#                 "agent": "retriever",
#                 "tool": "FAISS Vector Search",
#                 "input": query_text,
#                 "output_count": 0,
#                 "error": "FAISS index missing",
#                 "handoff-to": "controller",
#             })
#             return state

#         distances, indices = faiss_idx.search(query_vector, 5)
#         indices = indices.tolist()[0]

#         # Retrieve chunks and metadata
#         retrieved_chunks = []
#         retrieved_metadata = []
#         metadata_list = get_metadata_list()
#         for idx in indices:
#             # FAISS may return -1 when no neighbor found; guard against invalid indices
#             if not isinstance(idx, int) or idx < 0 or idx >= len(metadata_list):
#                 logger.debug("Skipping invalid index from FAISS: %s", idx)
#                 continue
#             entry = metadata_list[idx]
#             retrieved_chunks.append(entry.get("content", ""))
#             retrieved_metadata.append(entry)

#         # Updating state
#         state["retrieved_chunks"] = retrieved_chunks
#         state["retrieved_metadata"] = retrieved_metadata
#         state["reasoning_trace"].append(f"Retrieved {len(retrieved_chunks)} chunks from FAISS index.")
#         state["trace"].append({
#             "agent": "retriever",
#             "tool": "FAISS Vector Search",
#             "input": query_text,
#             "output_count": len(retrieved_chunks),
#             "handoff-to": "controller"
#         })

#     except Exception as e:
#         state["error"] = str(e)
#         state.setdefault("trace", []).append({
#             "agent": "retriever",
#             "tool": "FAISS Vector Search",
#             "input": state.get("query", ""),
#             "error": str(e),
#             "handoff-to": None
#         })
    
#     return state

# Load environment variables if you use .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ===============================
# Load FAISS & Metadata at Import
# (same as your working notebook)
# ===============================

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", r"C:\\Projects\\Financial_docs\\Multi-Agent-QA-on-Financial-Documents\\finance.new.index")
METADATA_PATH    = os.getenv("METADATA_PATH",    r"C:\\Projects\\Financial_docs\\Multi-Agent-QA-on-Financial-Documents\\chunks.new.jsonl")

# ---- Load FAISS index ----
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# ---- Load metadata ----
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata_list = [json.loads(line) for line in f]

# ---- Load embedding model ----
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")


# ===============================
# Retrieval Agent (simple + clean)
# ===============================

def retrieval_agent(state: dict):

    try:
        state.setdefault("reasoning_trace", [])
        state.setdefault("trace", [])

        query_text = state["query"]
        state["active_agent"] = "retriever"

        # ---- Encode Query ----
        query_vector = embedding_model.encode(
            [f"query: {query_text}"],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # ---- Search FAISS ----
        distances, indices = faiss_index.search(query_vector, 5)
        indices = indices.tolist()[0]

        # ---- Collect retrieved results ----
        retrieved_chunks = []
        retrieved_metadata = []

        for idx in indices:
            retrieved_chunks.append(metadata_list[idx]["content"])
            retrieved_metadata.append(metadata_list[idx])

        # ---- Update state ----
        state["retrieved_chunks"] = retrieved_chunks
        state["retrieved_metadata"] = retrieved_metadata

        state["reasoning_trace"].append(
            f"Retrieved {len(retrieved_chunks)} chunks from FAISS index."
        )

        state["trace"].append({
            "agent": "retriever",
            "tool": "FAISS Vector Search",
            "input": query_text,
            "output_count": len(retrieved_chunks),
            "handoff-to": "controller"
        })

    except Exception as e:
        state["error"] = str(e)
        state.setdefault("trace", []).append({
            "agent": "retriever",
            "tool": "FAISS Vector Search",
            "input": state.get("query", ""),
            "error": str(e),
            "handoff-to": None
        })

    return state




# Defining table node
async def table_agent(state: AgentState):
    query = state["query"]
    retrieved_metadata = state.get("retrieved_metadata", [])
    state.setdefault("extracted_entities", {})
    state.setdefault("reasoning_trace", [])
    state.setdefault("trace", [])

    # Extract only table-like chunks
    table_chunks = "\n\n".join([
        c.get("content", "") for c in retrieved_metadata
        if "table" in c.get("modality", c.get("type", "")).lower() and c.get("content", "").strip()
    ])

    if not table_chunks.strip():
        msg = "No table data found; skipping table extraction."
        state["reasoning_trace"].append(msg)
        state["trace"].append({
            "agent": "table",
            "tool": "Structured Table Extractor",
            "input": query,
            "output": msg,
            "handoff-to": "controller"
        })
        state["next_agent"] = None
        return state

    # Defining schema
    class FinancialMetric(BaseModel):
        value: float
        unit: str

    class FinancialTableOutput(BaseModel):
        metrics: list[FinancialMetric]
        notes: str

    structured_llm_table = llm_1.with_structured_output(FinancialTableOutput)

    prompt = f"""
    You are a financial data extraction expert.
    Extract structured financial metrics and key notes related to the query below.

    Query: {query}

    Tables:
    {table_chunks}
    """

    result = await structured_llm_table.ainvoke(prompt)

    # Update State
    state["extracted_entities"]["table_data"] = result.dict()
    state["reasoning_trace"].append("Table Agent extracted structured data successfully.")
    state["next_agent"] = "math"
    state["trace"].append({
        "agent": "table",
        "tool": "Structured Table Extractor",
        "input": query,
        "output": result.dict(),
        "handoff-to": "controller"
    })
    # leave state["next_agent"] = "math" for controller to consider next

    return state



# Defining math node
async def math_agent(state: AgentState):
    """
    Math agent:
    - Takes query, retrieved text, and structured table data.
    - Performs any necessary mathematical or quantitative reasoning using an LLM.
    - Stores results in the state under 'math_results'.
    """

    query = state.get("query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    table_data = state.get("extracted_entities", {}).get("table_data", {})

    state["active_agent"] = "math"

    # Define structured output for math reasoning
    class MathComputationResult(BaseModel):
        reasoning: str = Field(description="Step-by-step explanation of the math process.")
        result: str = Field(description="Final numeric or analytical answer to the query.")

    structured_llm_math = llm_1.with_structured_output(MathComputationResult)

    # Build compact prompt
    prompt = f"""
You are a financial mathematics agent.
You are given:
1. A user query related to finance or numerical reasoning.
2. Retrieved context from financial documents.
3. Structured table data extracted earlier.

Your task is to perform the required mathematical or quantitative computation.
Be explicit about calculations, units, and logic.

Query:
"{query}"

Retrieved Text:
{retrieved_chunks}

Structured Table Data:
{json.dumps(table_data, indent=2)}

Return the step-by-step reasoning and the final computed result.
    """

    try:
        # Use async invoke to avoid blocking the event loop
        computation = await structured_llm_math.ainvoke(prompt)

        # Save to state
        
        # computation is a pydantic model or structured output
        # use .dict() safely
        comp_dict = computation.dict() if hasattr(computation, 'dict') else {
            'reasoning': getattr(computation, 'reasoning', None),
            'result': getattr(computation, 'result', None)
        }

        state["math_results"] = {
            "reasoning": comp_dict.get('reasoning'),
            "result": comp_dict.get('result'),
        }

        state["reasoning_trace"].append("Math Agent performed quantitative reasoning.")
        state["trace"].append({
            "agent": "math",
            "tool": "LLM-based quantitative reasoning",
            "input": {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "table_data": table_data,
            },
            "output": comp_dict,
            "handoff-to": "aggregator",
        })

        # Hand off to aggregator next
        state["next_agent"] = "aggregator"

    except Exception as e:
        state["error"] = str(e)
        state["trace"].append({
            "agent": "math",
            "tool": "LLM-based quantitative reasoning",
            "input": query,
            "output": None,
            "error": str(e),
            "handoff-to": None
        })

    return state


# Defining websearch node
async def websearch_agent(state: AgentState) -> AgentState:
    """
    Websearch agent: use `query` to fetch web search results and store them in the state.
    """
    state.setdefault("trace", [])
    state.setdefault("reasoning_trace", [])
    # standardize agent name to 'web_search' so graph and traces align
    state["active_agent"] = "web_search"

    query = state.get("query", "").strip()
    if not query:
        state["reasoning_trace"].append("Empty query; skipping websearch.")
        state["websearch_results"] = []
        state["trace"].append({
            "agent": "web_search",
            "reasoning": "no query found"
        })
        return state

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),  # uncomment if using a different key
            "num": 5,
            "hl": "en"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://serpapi.com/search", params=params)
            resp.raise_for_status()
            result_json = resp.json()

        # Extract organic results
        organic = result_json.get("organic_results", [])
        # Simplify / structure what's stored
        simplified = []
        for r in organic:
            simplified.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet")
            })

        state["websearch_results"] = simplified
        state["reasoning_trace"].append(f"Websearch fetched {len(simplified)} result(s).")
        state["trace"].append({
            "agent": "web_search",
            "tool": "SerpAPI",
            "input": query,
            "output": simplified,
            "handoff-to": None
        })

    except Exception as e:
        state["websearch_results"] = []
        state["reasoning_trace"].append(f"Websearch failed: {e}")
        state["trace"].append({
            "agent": "web_search",
            "tool": "SerpAPI",
            "input": query,
            "error": str(e),
            "handoff-to": None
        })

    return state





# Defining aggregator node
def aggregator_agent(state: AgentState):
    """
    Aggregator agent: merges results from all previous agents into final answer
    without using LLM. Just collects and combines.
    """
    final_parts = []

    if state.get("retrieved_metadata"):
        retrieved_texts = [chunk.get("content", "") for chunk in state["retrieved_metadata"]]
        final_parts.append("Retrieved Chunks:\n" + "\n".join(retrieved_texts))

    if state.get("extracted_entities"):
        final_parts.append("Extracted Entities:\n" + str(state["extracted_entities"]))

    if state.get("current_summary"):
        final_parts.append("Current Summary:\n" + state["current_summary"])
    if state.get("websearch_results"):
        # Convert search results list into readable string
        try:
            ws_text = "\n".join([f"{r.get('title','[no title]')} - {r.get('link','')}" for r in state.get('websearch_results', [])])
        except Exception:
            ws_text = str(state.get('websearch_results'))
        final_parts.append("Websearch results:\n" + ws_text)

    
    final_answer = "\n\n".join(final_parts)

    # Update state
    state["final_answer"] = final_answer
    state["current_summary"] = final_answer
    state["active_agent"] = "aggregator"
    state["trace"].append({
        "agent": "aggregator",
        "decision": "final_answer",
        "reasoning": "Collected and merged outputs from all previous agents."
    })

    state["next_agent"] = "summarizer"  
    return state




async def summarizer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Summarizer agent: takes the current_summary from state, produces a concise, 
    query-specific summary, and updates final_answer.
    """

    current_summary = state.get("current_summary", "")
    query = state.get("query", "")

    if not current_summary:
        # Nothing to summarize
        state["final_answer"] = "No content available to summarize."
        return state

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a financial assistant. Summarize the following content specifically in context of the user's query."
        ),
        HumanMessagePromptTemplate.from_template(
            "User Query: {query}\nContent: {current_summary}\n\nProvide a concise answer relevant to the query."
        )
    ])

    formatted_prompt = prompt.format_messages(query=query, current_summary=current_summary)

    # use async invoke to avoid blocking the pipeline
    try:
        response = await llm_2.ainvoke(formatted_prompt)
        # response may be an object with .content
        summary = getattr(response, 'content', None) or (response[0].content if isinstance(response, list) and response else str(response))
    except Exception as e:
        logger.exception("Summarizer LLM call failed: %s", e)
        # fallback to current_summary if LLM is not available
        summary = current_summary

    state["final_answer"] = summary
    state["current_summary"] = summary
    state["active_agent"] = "summarizer"
    state["trace"].append({
        "agent": "summarizer",
        "decision": "summarized content",
        "reasoning": "Produced concise, query-specific summary from current summary."
    })
    state["next_agent"] = None 

    return state



def check(state: AgentState):
    return state["next_agent"]



graph=StateGraph(AgentState)
# Defining all nodes
graph.add_node("controller", ControllerAgent)
graph.add_node("retriever", retrieval_agent)
graph.add_node("table", table_agent)
graph.add_node("math", math_agent)
graph.add_node("web_search", websearch_agent)
graph.add_node("summarizer", summarizer_agent)
graph.add_node("aggregator", aggregator_agent)


# Defining edges
graph.add_edge(START, "controller")
graph.add_conditional_edges("controller",check,
    {
        "retriever": "retriever",
        "table": "table",
        "math": "math",
        "aggregator": "aggregator",
        "web_search": "web_search",
    }
)

graph.add_edge("retriever","controller")
graph.add_edge("table","controller")
graph.add_edge("math","controller")
graph.add_edge("web_search", "controller")
graph.add_edge("aggregator","summarizer")
graph.add_edge("summarizer",END)


pipeline=graph.compile()


import asyncio

async def main():
    query = "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.The metric capital expenditures was directly extracted from the company 10K. The line item name, as seen in the 10K, was: Purchases of property, plant and equipment (PP&E). Can you give the answer by websearch?"

    initial_state = {
        "query": query,
        "trace": [],
        "reasoning_trace": [],
        "retrieved_chunks": [],
        "retrieved_metadata": [],
        "extracted_entities": {},
    }

    result_state = await pipeline.ainvoke(initial_state)

    print(result_state["final_answer"])
    print("Reasoning Trace:\n", "\n".join(result_state["reasoning_trace"]))
    print("Final Trace:\n", result_state["trace"])

print("Graph compiled successfully.")

if __name__ == "__main__":
    # Running main is now explicit to prevent execution during import. This avoids
    # starting heavy downloads or LLM requests unintentionally in test/import scenarios.
    logger.info("Starting pipeline main()")
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Error while running main(): %s", e)
