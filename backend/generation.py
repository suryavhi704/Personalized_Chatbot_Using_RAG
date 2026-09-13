"""
Generation: builds the grounded prompt and calls the GROQ LLM.
"""

from langchain_groq import ChatGroq

from backend.config import GROQ_MODEL_NAME, get_groq_api_key

PROMPT_TEMPLATE = """You are miro.ai, a helpful assistant for Techno Engineering College Banipur.
Answer the question using ONLY the context below. If the answer is not
contained in the context, say you don't have that information instead
of guessing.

Context:
{context}

Question:
{question}

Answer:
"""


def _get_llm():
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=get_groq_api_key(),
        temperature=0.2,
    )


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = _get_llm()
    response = llm.invoke(prompt)
    return response.content
