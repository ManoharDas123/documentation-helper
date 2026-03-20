# from typing import Any, Dict
#
# import os, ssl, certifi, urllib3
#
# # Make Requests, urllib3, and Pinecone use certifi bundle
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# os.environ["SSL_CERT_FILE"] = certifi.where()
#
# # Disable Pinecone SSL verification
# os.environ["PINECONE_SSL_VERIFY"] = "false"
#
# # Disable urllib3 SSL warnings (optional)
# urllib3.disable_warnings()
#
# # Global SSL bypass
# ssl._create_default_https_context = ssl._create_unverified_context
#
#
# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain.chat_models import init_chat_model
# from langchain.messages import ToolMessage
# from langchain.tools import tool
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI
#
# load_dotenv()
#
# # Initialize embeddings (same as ingestion.py)
# # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
#
# # import ssl, certifi, os
# # os.environ['SSL_CERT_FILE'] = certifi.where()
# # ssl._create_default_https_context = ssl._create_unverified_context
#
# # Initialize vector store
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-helper-pinecode", embedding=embeddings
# )
# # Initialize chat model
# # model = init_chat_model("llama3.1:8b", model_provider="ollama")
#
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",  # or "gemini-1.5-pro"
#     temperature=0
# )
#
#
#
# @tool(response_format="content_and_artifact")
# def retrieve_context(query: str):
#     """Retrieve relevant documentation to help answer user queries about LangChain."""
#     # Retrieve top 4 most similar documents
#     retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
#
#     # Serialize documents for the model
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#
#     # Return both serialized content and raw documents
#     return serialized, retrieved_docs
#
#
# def run_llm(query: str) -> Dict[str, Any]:
#     """
#     Run the RAG pipeline to answer a query using retrieved documentation.
#
#     Args:
#         query: The user's question
#
#     Returns:
#         Dictionary containing:
#             - answer: The generated answer
#             - context: List of retrieved documents
#     """
#     # Create the agent with retrieval tool
#     system_prompt = (
#         "You are a helpful AI assistant that answers questions about LangChain documentation. "
#         "You have access to a tool that retrieves relevant documentation. "
#         "Use the tool to find relevant information before answering questions. "
#         "Always cite the sources you use in your answers. "
#         "If you cannot find the answer in the retrieved documentation, say so."
#     )
#
#     agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
#
#     # Build messages list
#     messages = [{"role": "user", "content": query}]
#
#     # Invoke the agent
#     response = agent.invoke({"messages": messages})
#
#     # Extract the answer from the last AI message
#     answer = response["messages"][-1].content
#
#     # Extract context documents from ToolMessage artifacts
#     context_docs = []
#     for message in response["messages"]:
#         # Check if this is a ToolMessage with artifact
#         if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
#             # The artifact should contain the list of Document objects
#             if isinstance(message.artifact, list):
#                 context_docs.extend(message.artifact)
#
#     return {
#         "answer": answer,
#         "context": context_docs
#     }
#
#
# if __name__ == '__main__':
#     result = run_llm(query="what are deep agents?")
#     print(result)

# ============================================================
#  FIX FOR CORPORATE WINDOWS LAPTOPS (NOKIA, ZSCALER, ETC.)
#  Must appear BEFORE ANY imports that make HTTPS calls.
# ============================================================

import os
import ssl
import certifi
import urllib3

# Use Windows system certificates
try:
    import pip_system_certs.wrapt  # type: ignore
except Exception:
    pass  # pip-system-certs is optional but recommended

# Override SSL verification for Pinecone + Requests + urllib3
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["PINECONE_SSL_VERIFY"] = "false"

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

# ============================================================
# NORMAL IMPORTS
# ============================================================

from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool

from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # Load PINECONE + GOOGLE API keys safely


# ============================================================
# 1. Embeddings (Local via Ollama → NO SSL ISSUES)
# ============================================================

embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ============================================================
# 2. Vector DB (Pinecone)
# ============================================================
# Pinecone must use SSL overrides given above.
# If index does not exist → automatically handled.

vectorstore = PineconeVectorStore(
    index_name="langchain-docs-helper-pinecode",
    embedding=embeddings,
)


# ============================================================
# 3. Chat Model (Gemini 2.5 Flash or 1.5 Pro)
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
    """
    Run a fully tool-enabled Gemini RAG pipeline.
    """

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain "
        "documentation using retrieved context. "
        "Always call the retrieval tool first, then answer the question. "
        "Always cite sources. "
        "If documentation is missing, clearly state that."
    )

    # Gemini 2.5 supports tool calling natively 🎉
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    # Extract answer
    answer = response["messages"][-1].content

    # Extract tool artifacts (documents)
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