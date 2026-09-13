from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL,
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD
)

from backend.ingestion import VectorStoreManager


# -------------------------------------------------
# Embedding Loader
# -------------------------------------------------
class RetrievalEmbeddingManager:

    def __init__(self, model_name=EMBEDDING_MODEL):

        self.model_name = model_name

        print("Loading retrieval embedding model...")

        self.model = SentenceTransformer(self.model_name)


    def generate_query_embedding(self, query):

        embedding = self.model.encode([query])

        return embedding[0]

# -------------------------------------------------
# RAG Retriever
# -------------------------------------------------
class RAGRetriever:

    def __init__(
        self,
        embedding_manager,
        vector_store
    ):

        self.embedding_manager = embedding_manager
        self.vector_store = vector_store


    def retrieve(
        self,
        query,
        top_k=TOP_K_RESULTS,
        score_threshold=SIMILARITY_THRESHOLD
    ):

        query_embedding = self.embedding_manager.generate_query_embedding(
            query
        )


        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )


        retrieved_documents = []


        if results["documents"] and results["documents"][0]:

            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]


            for index, (
                doc_id,
                metadata,
                document,
                distance
            ) in enumerate(
                zip(ids, metadatas, documents, distances)
            ):

                retrieved_documents.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "similarity_score": round(1 - distance, 4),
                "rank": index + 1
            })


            print(
                f"Retrieved {len(retrieved_documents)} documents"
            )


        else:
            print("No relevant documents found")


        return retrieved_documents


# -------------------------------------------------
# Retriever Factory Function
# -------------------------------------------------
def build_retriever():

    embedding_manager = RetrievalEmbeddingManager()


    vector_store = VectorStoreManager(
        persist_directory=VECTOR_STORE_PATH,
        collection_name=COLLECTION_NAME
    )


    retriever = RAGRetriever(
        embedding_manager,
        vector_store
    )


    return retriever