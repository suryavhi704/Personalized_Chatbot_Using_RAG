"""
Central configuration for the RAG pipeline.

Every other module imports paths and constants from here — never
hardcode a path string (and especially never a Windows-style
absolute path like C:\\Users\\...) anywhere else in the project.
Using pathlib like this makes the project work unchanged on
Windows, macOS, Linux, and Streamlit Cloud.
"""

from pathlib import Path
import os

# -------------------------------------------------
# Base paths (portable — resolved relative to this file, not the OS)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "college.txt"

# Single canonical name for the vector store folder.
# Use this constant everywhere — never write "vector_store" or
# "vectorstore" as a string literal anywhere else in the codebase.
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# -------------------------------------------------
# Embedding model
# -------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------------------------
# Chroma collection
# -------------------------------------------------
COLLECTION_NAME = "college_rag_collection"

# -------------------------------------------------
# Text chunking
# -------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# -------------------------------------------------
# Retrieval
# -------------------------------------------------
TOP_K = 4

# -------------------------------------------------
# GROQ LLM
# -------------------------------------------------
# NOTE: llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16
# for free/developer tier accounts. openai/gpt-oss-120b is Groq's
# recommended 1:1 replacement. If Groq deprecates this one too, check
# https://console.groq.com/docs/deprecations and swap the string here —
# this is the ONLY place the model name should ever be set.
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


def get_groq_api_key() -> str:
    """
    Reads the GROQ API key from Streamlit secrets first (this is what's
    used on Streamlit Cloud), falling back to a local environment
    variable / .env file (used when running locally in VS Code).

    The key itself is never hardcoded here — only the lookup logic is.
    """
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in .streamlit/secrets.toml "
            "(Streamlit Cloud -> App settings -> Secrets) or in a local "
            ".env file (local development)."
        )
    return key
