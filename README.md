<<<<<<< HEAD
# hr-rag-chat-bot
=======
# Zyro RAG Bot

A Streamlit-powered Retrieval-Augmented Generation (RAG) chatbot for Zyro Dynamics HR policies.

The app loads HR policy PDFs from `zyro-rag-bot/data/`, embeds them with Hugging Face embeddings, stores them in FAISS, and answers employee HR questions using a Groq language model via `langchain-groq`.

## Project structure

- `zyro-rag-bot/app.py` – Streamlit application entrypoint.
- `zyro-rag-bot/requirements.txt` – Python dependencies.
- `zyro-rag-bot/data/` – Directory for HR policy PDF documents.
- `.venv/` – Python virtual environment (local development environment).

## Features

- Document loading from PDF files
- Chunking with recursive text splitting
- Embedding with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector store retrieval
- Groq-based LLM chat responses
- Scope guardrail to keep answers relevant to HR policies

## Prerequisites

- Python 3.11+ (recommended)
- A virtual environment, such as `venv`
- Streamlit installed
- Valid `GROQ_API_KEY` for Groq model access
- Optional `LANGCHAIN_API_KEY` if you want to set it via environment or secrets

## Installation

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r zyro-rag-bot/requirements.txt
```

## Configuration

Create a `.env` file in the repository root or set the values in Streamlit secrets.

Required variables:

```env
GROQ_API_KEY=your_groq_api_key
```

Optional variables:

```env
LANGCHAIN_API_KEY=your_langchain_api_key
LLM_MODEL=llama-3.1-8b-instant
```

If `LLM_MODEL` is not provided, the app defaults to `llama-3.1-8b-instant`.

## Usage

Run the Streamlit app from the repository root:

```bash
streamlit run zyro-rag-bot/app.py
```

Then open the local URL printed by Streamlit in your browser.

## Data

Place your HR policy PDFs in:

```text
zyro-rag-bot/data/
```

The app loads all PDFs from this directory and uses them as the knowledge base.

## Notes

- The application only answers HR-related questions using the loaded policy documents.
- If a question is out of scope, the bot replies with a refusal message.
- The First run may take longer while embeddings and FAISS indexes are built.

## License

This repository does not include a license file. Add one if you plan to distribute or share this project.
>>>>>>> 4269149 (Initial commit)
