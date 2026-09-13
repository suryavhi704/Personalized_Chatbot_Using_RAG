"""
Retriever: turns a user query into an embedding and searches the
persistent Chroma vector store for the most relevant chunks.

Uses the same cached singletons as ingestion.py, so it never loads
a second copy of the embedding model or opens a second Chroma client.
"""

from backend.ingestion import get_embedding_manager, get_vector_store_manager
from backend.config import TOP_K


def retrieve_relevant_chunks(query: str, top_k: int = TOP_K) -> list[str]:
    embedding_manager = get_embedding_manager()
    vector_store = get_vector_store_manager()

    query_embedding = embedding_manager.generate_embeddings([query])[0]
    results = vector_store.query(query_embedding, top_k=top_k)

    documents = results.get("documents", [[]])[0]
    return documents
