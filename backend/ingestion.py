"""
Ingestion pipeline: loads the source text file, splits it into chunks,
embeds each chunk, and writes it into a persistent Chroma collection.

Two things make this safe to import from a Streamlit app that reruns
on every user interaction:

1. get_embedding_manager() / get_vector_store_manager() are cached
   with functools.lru_cache(maxsize=1), so the (expensive) embedding
   model and the Chroma client are only created ONCE per process,
   no matter how many times Streamlit reruns the script.
2. add_documents() checks the collection count before writing, so
   even if ingestion were somehow triggered twice, it's a no-op the
   second time instead of a duplicate-write / lock conflict.
"""

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter

from backend.config import (
    DATA_FILE,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# -------------------------------------------------
# Embedding Manager
# -------------------------------------------------
class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # len() of one probe embedding works across sentence-transformers
        # versions, unlike get_sentence_embedding_dimension() which was
        # renamed in newer releases.
        self.dimension = len(self.model.encode(["dimension probe"])[0])
        print(f"Embedding dimensions = {self.dimension}")

    def generate_embeddings(self, texts: list[str]):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings


# -------------------------------------------------
# Vector Store Manager
# -------------------------------------------------
class VectorStoreManager:
    def __init__(self, persist_dir: Path = VECTORSTORE_DIR, collection_name: str = COLLECTION_NAME):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"Initialized vector store collection: {collection_name}")
        print(f"Existing documents in collection: {self.collection.count()}")

    def add_documents(self, chunks: list[str], embeddings):
        if self.collection.count() > 0:
            print("Vector store already populated — skipping ingestion.")
            return

        ids = [f"chunk-{i}" for i in range(len(chunks))]
        metadatas = [{"source": "college.txt", "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
            documents=chunks,
        )
        print(f"Added {len(chunks)} chunks to the vector store.")

    def query(self, query_embedding, top_k: int = 4):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )


# -------------------------------------------------
# Process-wide singletons
# -------------------------------------------------
@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@lru_cache(maxsize=1)
def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager()


# -------------------------------------------------
# Data Loading + Chunking
# -------------------------------------------------
def load_and_chunk_document() -> list[str]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find source data file at: {DATA_FILE}. "
            f"Make sure data/college.txt is committed to the repo."
        )

    text = DATA_FILE.read_text(encoding="utf-8")

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# -------------------------------------------------
# Public entrypoint
# -------------------------------------------------
def run_ingestion_pipeline() -> VectorStoreManager:
    """
    Builds (or reuses) the vector store. Safe to call on every
    Streamlit rerun — the singleton caching above and the
    collection-count check inside add_documents() together make
    repeated calls a cheap no-op after the first one.
    """
    vector_store = get_vector_store_manager()

    if vector_store.collection.count() == 0:
        chunks = load_and_chunk_document()
        embedding_manager = get_embedding_manager()
        embedded_texts = embedding_manager.generate_embeddings(chunks)
        vector_store.add_documents(chunks, embedded_texts)

    return vector_store
