import os

from dotenv import load_dotenv


load_dotenv()


# ------------------------------
# Paths
# ------------------------------
DATA_PATH = r"C:\Users\DELL\Desktop\RAG_Project\data\college.txt"
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "college_rag_collection"


# ------------------------------
# Chunking configuration
# ------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ------------------------------
# Embedding Model
# ------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ------------------------------
# Retrieval configuration
# ------------------------------
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.0


# ------------------------------
# LLM configuration
# ------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.1
MAX_TOKENS = 1024