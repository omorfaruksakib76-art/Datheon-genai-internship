# Datheon Python + Generative AI & LangChain Internship

Three tasks built with Python, Google Gemini and LangChain.

| Task | File | What it does |
|------|------|--------------|
| 1. LLM Prompt Pipeline & JSON Extractor | `task1_extractor.py` | Extracts invoice / resume data from raw text into Pydantic-validated JSON (few-shot prompts, auto-retry on invalid output) |
| 2. Document Q&A with RAG | `rag.py`, `task2_cli.py` | PDF/TXT -> recursive chunking -> Gemini embeddings -> ChromaDB -> grounded answers |
| 4. Streamlit AI Chatbot | `task4_app.py` | Web UI over the RAG pipeline with streaming tokens, chat history and session state |

## Setup

```bash
pip install -r requirements.txt
```
Create a `.env` file with:
```
GOOGLE_API_KEY=your_gemini_api_key
```
(free key at aistudio.google.com). Optional: `GEMINI_MODEL`, `EMBEDDING_MODEL`.

## Run

```bash
python task1_extractor.py --type invoice --file sample_invoice.txt
python task1_extractor.py --type resume  --file sample_resume.txt
python task2_cli.py sample.txt
streamlit run task4_app.py
```

## Design notes
- **Task 1:** JSON mode + Pydantic validation; on validation error, the error is fed back to the model for a corrected attempt.
- **Task 2:** `RecursiveCharacterTextSplitter` (800 chars, 100 overlap), top-4 retrieval, prompt restricts answers to retrieved context.
- **Task 4:** `st.session_state` holds messages and retriever; `st.write_stream` streams tokens; previous turns are passed as memory.
