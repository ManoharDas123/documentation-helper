# ============================================================
# STREAMLIT CLOUD VERSION — DO NOT USE .env or SSL PATCHES
# ============================================================

import streamlit as st
import os

# Load API keys from Streamlit Secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

# ============================================================
# NORMAL IMPORTS
# ============================================================

from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool

from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# 1. Embeddings (Local via Ollama → works in Cloud)
# ============================================================

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ============================================================
# 2. Vector DB (Pinecone)
# ============================================================

vectorstore = PineconeVectorStore(
    index_name="langchain-docs-helper-pinecode",
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

    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

# ============================================================
# 5. MAIN LLM CALL
# ============================================================

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

    answer = response["messages"][-1].content

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