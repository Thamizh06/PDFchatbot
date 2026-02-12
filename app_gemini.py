import streamlit as st
import tempfile
from gemini import (
    load_and_split_pdf,
    create_vectorstore,
    create_qa_chain,
    ask_bot
)

st.set_page_config(page_title="VZ Chatbot", layout="wide")
st.title("VZ Chatbot")

# Setup
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# PDF Upload Section
uploaded_file = st.file_uploader(
    "Upload a PDF to start chatting",
    type="pdf"
)

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Reading PDF and building knowledge base..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        chunks = load_and_split_pdf(temp_pdf_path)
        vectorstore = create_vectorstore(chunks)
        st.session_state.qa_chain = create_qa_chain(vectorstore)

    st.success("PDF processed! You can now ask questions.")


# DISPLAY CHAT HISTORY
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤖 Thinking...")

            response = ask_bot(
                st.session_state.qa_chain,
                user_input
            )

            answer = response["answer"]

            # Typing animation
            import time
            typed_text = ""
            for char in answer:
                typed_text += char
                message_placeholder.markdown(typed_text + "▌")
                time.sleep(0.01)

            message_placeholder.markdown(typed_text)

        # Save assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
