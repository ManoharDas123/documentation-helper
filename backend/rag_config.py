"""Pinecone RAG settings — ingest and query must use the same model and dimension."""

from __future__ import annotations

from langchain_pinecone import PineconeEmbeddings

# Must match the Pinecone index name exactly (your console shows `langchain-ollma-index`).
PINECONE_INDEX_NAME = "langchain-ollma-index"

PINECONE_EMBED_MODEL = "llama-text-embed-v2"
# Must equal the index vector dimension (768 for your index).
PINECONE_EMBED_DIMENSION = 768


def make_pinecone_embeddings() -> PineconeEmbeddings:
    """Build Pinecone inference embeddings aligned to ``PINECONE_EMBED_DIMENSION``.

    LangChain's ``PineconeEmbeddings`` keeps a ``dimension`` field for defaults but does
    **not** pass it into ``inference.embed()`` — only keys inside ``document_params`` /
    ``query_params`` are sent. Matryoshka size must therefore appear in those dicts.
    """
    d = PINECONE_EMBED_DIMENSION
    return PineconeEmbeddings(
        model=PINECONE_EMBED_MODEL,
        dimension=d,
        document_params={
            "input_type": "passage",
            "truncate": "END",
            "dimension": d,
        },
        query_params={
            "input_type": "query",
            "truncate": "END",
            "dimension": d,
        },
    )
