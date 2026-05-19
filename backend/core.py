# ============================================================
# STREAMLIT CLOUD VERSION — DO NOT USE .env or SSL PATCHES
# ============================================================

import streamlit as st
import os

# Load API keys from Streamlit Secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]
os.environ["PINECONE_REGION"] = st.secrets["PINECONE_REGION"]

# ============================================================
# NORMAL IMPORTS
# ============================================================

import json
import time
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.rag_config import PINECONE_INDEX_NAME, make_pinecone_embeddings

# ============================================================
# 1. Embeddings (Pinecone Inference — same model/dim as ingestion)
# ============================================================

embeddings = make_pinecone_embeddings()

# #region agent log
_AGENT_EMB_DIM_LOGGED = False


def _agent_dbg_log(
    location: str,
    message: str,
    data: dict,
    hypothesis_id: str,
    run_id: str = "pre-fix",
) -> None:
    try:
        _log_path = os.path.join(os.path.dirname(__file__), "..", "debug-6d7fad.log")
        payload = {
            "sessionId": "6d7fad",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion


# ============================================================
# 2. Vector DB (Pinecone)
# ============================================================

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

# ============================================================
# 3. Chat Model (Gemini)
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# ============================================================
# 4. TOOL: Retrieve top 4 documents
# ============================================================

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant LangChain documentation for RAG."""

    # #region agent log
    global _AGENT_EMB_DIM_LOGGED
    if not _AGENT_EMB_DIM_LOGGED:
        _AGENT_EMB_DIM_LOGGED = True
        try:
            _probe = embeddings.embed_query("__agent_dim_probe__")
            _dim = len(_probe) if isinstance(_probe, list) else getattr(_probe, "shape", [None])[-1]
            _agent_dbg_log(
                "backend/core.py:retrieve_context",
                "embedding query vector dimension probe",
                {
                    "index_name": PINECONE_INDEX_NAME,
                    "embedding_model": getattr(embeddings, "model", None),
                    "embedding_dimension_field": getattr(embeddings, "dimension", None),
                    "query_params_dimension": (getattr(embeddings, "query_params", {}) or {}).get(
                        "dimension"
                    ),
                    "query_vector_dim": _dim,
                },
                "H1",
            )
        except Exception as _probe_err:
            _agent_dbg_log(
                "backend/core.py:retrieve_context",
                "embedding probe failed",
                {"error_type": type(_probe_err).__name__, "error": str(_probe_err)[:200]},
                "H3",
            )
    # #endregion

    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

# ============================================================
# 5. MAIN LLM CALL
# ============================================================


def _assistant_content_to_text(content: Any) -> str:
    """Turn LangChain message content into plain text for the UI.

    Gemini (e.g. via ``ChatGoogleGenerativeAI``) may return a list of blocks such as
    ``{'type': 'text', 'text': '...', 'extras': {'signature': '...'}}``. Using ``str()``
    on that list exposes ``signature``; we only join human-readable ``text`` fields.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts).strip()
    return str(content).strip()


def run_llm(query: str) -> Dict[str, Any]:

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain "
        "documentation using retrieved context. "
        "Always call the retrieval tool first, then answer the question. "
        "Always cite sources. "
        "If documentation is missing, clearly state that."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})

    answer = _assistant_content_to_text(response["messages"][-1].content)

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs,
    }

# ============================================================
# 6. STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    result = run_llm("What are deep agents?")
    print(result["answer"])
    print("\nSources:")
    for doc in result["context"]:
        print("-", doc.metadata.get("source"))