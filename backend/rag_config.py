"""Pinecone RAG settings — ingest and query must use the same model and dimension."""

# Index dimension in Pinecone (must match index settings). Your index is 768.
PINECONE_INDEX_NAME = "langchain-ollma-index"

# Hosted embeddings — must match between ingestion.py and backend/core.py
PINECONE_EMBED_MODEL = "llama-text-embed-v2"
# Matryoshka output size; must equal the index vector dimension (768 for your index).
PINECONE_EMBED_DIMENSION = 768
