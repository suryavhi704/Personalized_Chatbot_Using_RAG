"""
Top-level RAG orchestration — the only function app.py needs to call.
"""

import traceback

from backend.retrieval import retrieve_relevant_chunks
from backend.generation import generate_answer


def generate_rag_response(query: str) -> str:
    if not query or not query.strip():
        return "Please ask a question."

    try:
        context_chunks = retrieve_relevant_chunks(query)
        answer = generate_answer(query, context_chunks)
        return answer
    except Exception:
        # Print the FULL traceback (not just str(e)) so the real cause
        # — a bad model name, missing key, rate limit, etc. — shows up
        # in Streamlit Cloud's "Manage app" -> logs panel, instead of
        # only ever seeing the generic message below in the chat UI.
        print("RAG pipeline error:")
        traceback.print_exc()
        return "Sorry, something went wrong while generating a response. Please try again."
