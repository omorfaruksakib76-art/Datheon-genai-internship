"""Task 4: Interactive Full-Stack AI Chatbot with Streamlit.

Wraps the Task 2 RAG pipeline in a web UI with streaming output,
chat history and session state management.

Run:
    streamlit run task4_app.py
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from rag import build_retriever, load_documents, stream_answer  # noqa: E402

st.set_page_config(page_title="Document Chatbot", page_icon="💬")
st.title("💬 Document Q&A Chatbot")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Build knowledge base", disabled=not files):
        with st.spinner("Indexing documents..."):
            paths = []
            tmp = Path(tempfile.mkdtemp())
            for f in files:
                p = tmp / f.name
                p.write_bytes(f.getvalue())
                paths.append(str(p))
            st.session_state.retriever = build_retriever(load_documents(paths))
        st.success("Ready! Ask your questions.")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Chat input ----------
if prompt := st.chat_input("Ask something about your documents"):
    if st.session_state.retriever is None:
        st.warning("Upload documents and click 'Build knowledge base' first.")
        st.stop()

    # conversation memory: previous turns (before this prompt)
    history = [
        ("human" if m["role"] == "user" else "ai", m["content"]) for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_answer(prompt, st.session_state.retriever, history))
    st.session_state.messages.append({"role": "assistant", "content": reply})
