"""Task 2: Document Q&A Assistant with RAG & Vector Embeddings.

Pipeline: load PDF/text -> recursive chunking -> Gemini embeddings -> ChromaDB
-> retrieve top-k chunks -> Gemini answers using only that context.
Reusable by the CLI (task2_cli.py) and the Streamlit app (task4_app.py).
"""
import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def load_documents(paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        loader = PyPDFLoader(p) if Path(p).suffix.lower() == ".pdf" else TextLoader(p, encoding="utf-8")
        docs.extend(loader.load())
    return docs


def build_retriever(docs: list[Document], persist_dir: str | None = None, k: int = 4):
    """Chunk documents, embed them and store them in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return store.as_retriever(search_kwargs={"k": k})


def _format(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def stream_answer(question: str, retriever, history: list[tuple[str, str]] | None = None) -> Iterator[str]:
    """Yield answer tokens. `history` is a list of ("human"|"ai", text) tuples."""
    context = _format(retriever.invoke(question))
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    chain = PROMPT | llm | StrOutputParser()
    yield from chain.stream({"context": context, "question": question, "history": history or []})


def answer(question: str, retriever, history=None) -> str:
    return "".join(stream_answer(question, retriever, history))


def get_sources(question: str, retriever) -> list[Document]:
    return retriever.invoke(question)
