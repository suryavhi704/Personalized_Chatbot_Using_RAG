import os
import uuid
import chromadb

from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.config import (
    DATA_PATH,
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL
)

# -------------------------------------------------
# Document Loader
# -------------------------------------------------
def load_documents(file_path=DATA_PATH):

    loader = TextLoader(file_path)

    documents = loader.load()

    return documents

# -------------------------------------------------
# Text Chunking
# -------------------------------------------------
def split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_documents = text_splitter.split_documents(documents)

    return chunked_documents

# -------------------------------------------------
# Embedding Manager
# -------------------------------------------------
class EmbeddingManager:

    def __init__(self, model_name=EMBEDDING_MODEL):

        self.model_name = model_name

        print("Loading embedding model:", self.model_name)

        self.model = SentenceTransformer(self.model_name)

        print(
            "Embedding dimensions =",
            self.model.get_sentence_embedding_dimension()
        )
    def generate_embeddings(self, text):

        embeddings = self.model.encode(
            text,
            show_progress_bar=True
        )

        print("Embeddings shape:", embeddings.shape)

        return embeddings
    

# -------------------------------------------------
# Vector Store Manager
# -------------------------------------------------
    

class VectorStoreManager:

    def __init__(
        self,
        persist_directory=VECTOR_STORE_PATH,
        collection_name=COLLECTION_NAME
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self._initialize_store()


    def _initialize_store(self):

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "College chatbot RAG vector store"
            }
        )

        print(
            "Initialized vector store collection:",
            self.collection_name
        )

        print(
            "Existing documents in collection:",
            self.collection.count()
        )


    def add_documents(self, documents, embeddings):

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents does not match embeddings"
            )

        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []


        for index, (doc, embedding_vector) in enumerate(
            zip(documents, embeddings)
        ):

            doc_id = f"doc_{uuid.uuid4()}"

            ids.append(doc_id)

            metadata = dict(doc.metadata)

            metadata["doc_index"] = index
            metadata["content_length"] = len(doc.page_content)

            all_metadata.append(metadata)

            documents_content.append(doc.page_content)

            embeddings_list.append(
                embedding_vector.tolist()
            )


        if ids:

            self.collection.add(
                ids=ids,
                metadatas=all_metadata,
                documents=documents_content,
                embeddings=embeddings_list
            )

            print(
                "Total documents added in vector store =",
                len(documents_content)
            )

        else:
            print("No documents found for insertion")


        print(
            "Total documents in collection:",
            self.collection.count()
        )


# -------------------------------------------------
# Main ingestion pipeline
# -------------------------------------------------
def run_ingestion_pipeline():

    vector_store = VectorStoreManager()


    # Avoid duplicate insertion
    if vector_store.collection.count() > 0:
        print("Vector store already contains data")
        return


    documents = load_documents()

    chunks = split_documents(documents)

    print("Total chunks created:", len(chunks))


    embedding_manager = EmbeddingManager()


    texts = [doc.page_content for doc in chunks]

    embedded_texts = embedding_manager.generate_embeddings(texts)


    vector_store.add_documents(
        chunks,
        embedded_texts
    )


    print("Ingestion pipeline completed successfully")
            
        

