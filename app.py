import streamlit as st

from backend.ingestion import run_ingestion_pipeline
from backend.rag_pipeline import generate_rag_response


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="miro.ai",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------------------------
# Run Ingestion Once
# -------------------------------------------------
run_ingestion_pipeline()


# -------------------------------------------------
# Session State
# -------------------------------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>

    .stApp {
        background-color: black;
    }

    .welcome-container {
        height: 85vh;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }

    .welcome-text {
        color: #00AEEF;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        font-family: Arial, sans-serif;
    }

    .main-title {
        color: #00AEEF;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .sub-title {
        color: white;
        text-align: center;
        margin-bottom: 25px;
        font-size: 18px;
    }

    div[data-testid="stChatMessage"] {

    background-color: #111111;

    border-radius: 18px;

    padding: 18px;

    margin-bottom: 18px;
}


/* User + Assistant Text */

div[data-testid="stChatMessage"] p{

    color: #F8F9FA !important;

    font-size: 18px !important;

    font-weight: 500;

    line-height: 1.8;

    letter-spacing: 0.3px;
}


/* Headings */

div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3{

    color: #FFFFFF !important;
}


/* Lists */

div[data-testid="stChatMessage"] li{

    color: #F8F9FA !important;

    font-size: 18px;
}


/* Bold text */

div[data-testid="stChatMessage"] strong{

    color: #00AEEF !important;
}


/* Chat input text */

/* User typing text */

textarea{

    color:#000000 !important;

    font-size:18px !important;

    font-weight:500;

    caret-color:#00AEEF !important;
}


/* Placeholder text */

textarea::placeholder{

    color:#7A7A7A !important;

    opacity:1;
}

    div[data-testid="stChatInput"] {
        margin-top: 20px;
    }

    .chat-popup {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999;
    }

    div[data-testid="stButton"] > button:first-child {
        border-radius: 50%;
        width: 80px;
        height: 80px;
        background-color: #00AEEF;
        color: white;
        font-size: 32px;
        border: none;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Landing Page
# -------------------------------------------------
if not st.session_state.chat_open:

    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-text">
                Welcome to Techno Institute Of Engineering And Management 
            </div>
        </div>
        """,
        unsafe_allow_html=True 
    )

    st.markdown(
        '<div class="chat-popup">',
        unsafe_allow_html=True
    )

    if st.button("💬"):

        st.session_state.chat_open = True

        st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Main Chat Interface
# -------------------------------------------------
else:

    st.markdown(
        """
        <div class="main-title">
            miro.ai
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub-title">
            Your Personalized College Assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")


    # ---------------------------------------------
    # Display Previous Messages
    # ---------------------------------------------
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


    # ---------------------------------------------
    # Chat Input
    # ---------------------------------------------
    user_query = st.chat_input(
        "Ask your question here..."
    )


    # ---------------------------------------------
    # Generate Response
    # ---------------------------------------------
    if user_query:

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        # Display user message
        with st.chat_message("user"):
            st.write(user_query)

        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner("Generating response..."):

                response = generate_rag_response(
                    user_query
                )

                print("STREAMLIT RESPONSE =", response)

                if response is None:

                    response = (
                        "Sorry, I could not generate a response."
                    )

                st.write(response)

        # Store assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


    # -------------------------------------------------
    # Sidebar
    # -------------------------------------------------
    with st.sidebar:

        st.title("miro.ai")

        st.markdown("---")

        if st.button("Clear Chat"):

            st.session_state.messages = []

            st.rerun()

        if st.button("Back to Home"):

            st.session_state.chat_open = False

            st.rerun()