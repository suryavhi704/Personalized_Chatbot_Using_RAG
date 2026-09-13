import re

from langchain_groq import ChatGroq

from backend.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS
)


# -------------------------------------------------
# LLM Initialization
# -------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)


# -------------------------------------------------
# Clean Model Response
# -------------------------------------------------
def clean_response(text):

    cleaned_text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    return cleaned_text.strip()


# -------------------------------------------------
# Build Prompt
# -------------------------------------------------
def build_prompt(context, query):

    prompt = f"""
You are miro.ai, a helpful AI assistant for Techno Engineering College Banipur.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context,
say:
"Sorry, I could not find relevant information."

------------------------------
CONTEXT:
{context}
------------------------------

USER QUERY:
{query}

ANSWER:
"""

    return prompt


# -------------------------------------------------
# Generate Final Answer
# -------------------------------------------------
def generate_answer(query, retrieved_docs):

    # ---------------------------------------------
    # Build Context
    # ---------------------------------------------
    context = "\n\n".join(
        [doc["document"] for doc in retrieved_docs]
    ) if retrieved_docs else ""


    # ---------------------------------------------
    # Empty Context Handling
    # ---------------------------------------------
    if not context:

        return "Sorry, I could not find relevant information."


    # ---------------------------------------------
    # Build Prompt
    # ---------------------------------------------
    prompt = build_prompt(
        context,
        query
    )


    # ---------------------------------------------
    # Debug Prints
    # ---------------------------------------------
    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)

    print("\n========== FINAL PROMPT ==========")
    print(prompt)


    # ---------------------------------------------
    # Generate Response
    # ---------------------------------------------
    response = llm.invoke(prompt)


    print("\n========== RAW LLM RESPONSE ==========")
    print(response)


    # ---------------------------------------------
    # Extract Content
    # ---------------------------------------------
    final_answer = clean_response(
        response.content
    )


    print("\n========== FINAL ANSWER ==========")
    print(final_answer)

    # ---------------------------------------------
    # Empty Response Handling
    # ---------------------------------------------
    if not final_answer.strip():

        final_answer = (
            "Sorry, I could not generate a response."
        )

    return final_answer
