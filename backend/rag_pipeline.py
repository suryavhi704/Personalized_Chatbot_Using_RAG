from backend.retrieval import build_retriever
from backend.generation import generate_answer


# -------------------------------------------------
# Initialize Retriever
# -------------------------------------------------
retriever = build_retriever()


# -------------------------------------------------
# Main RAG Pipeline
# -------------------------------------------------
def generate_rag_response(query):

    retrieved_docs = retriever.retrieve(query)

    print("RETRIEVED DOCS =", retrieved_docs)

    final_answer = generate_answer(
        query,
        retrieved_docs
    )

    print("FINAL ANSWER FROM PIPELINE =", final_answer)

    return final_answer
