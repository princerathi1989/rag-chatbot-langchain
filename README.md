## RAG Chatbot (LangChain + Pinecone + Streamlit)

Production-ready Retrieval-Augmented Generation chatbot that indexes documentation and answers questions with cited sources. It uses LangChain for orchestration, OpenAI for embeddings and chat, Pinecone for vector storage, Tavily for large-scale web extraction, and Streamlit for the UI.

### Features
- **Document ingestion pipeline**: Crawl/map docs, extract raw content, split into chunks, and bulk-index to Pinecone.
- **Chat with memory**: History-aware retrieval to improve follow-up answers.
- **Cited sources**: Each answer includes a list of source URLs.
- **Streamlit UI**: Simple chat interface with a sidebar user profile.

---

## Quickstart

### 1) Prerequisites
- Python 3.10+
- Accounts/API keys:
  - OpenAI (`OPENAI_API_KEY`)
  - Pinecone (`PINECONE_API_KEY`) and an index created
  - Tavily (`TAVILY_API_KEY`)

### 2) Clone and enter the project
```bash
git clone <this-repo-url>
cd rag-chatbot-langchain
```

### 3) Install dependencies
Using Pipenv (recommended):
```bash
pipenv install --dev
pipenv shell
```

Or with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r <(pipenv requirements --dev) || true
# If Pipenv is not desired, install packages manually:
pip install streamlit python-dotenv langchain langchain-openai langchain-pinecone langchain-tavily certifi pinecone-client
```

### 4) Create a Pinecone index
- Embedding model: `text-embedding-3-small` (dimension 1536)
- Suggested name for ingestion: set `PINECONE_INDEX_NAME`
- Suggested name for chat/runtime: set `PINECONE_INDEX_NAME_CRAWL`

If you prefer a single index, set both env vars to the same index name.

### 5) Configure environment variables
Create a `.env` file in the repo root:
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
# If your setup requires an environment or host, also add:
# PINECONE_ENVIRONMENT=your_env   # legacy
# PINECONE_HOST=your_index_host   # serverless hosts

# Index names
PINECONE_INDEX_NAME=your_ingestion_index_name
PINECONE_INDEX_NAME_CRAWL=your_runtime_index_name

# Tavily (for mapping/extracting documentation at scale)
TAVILY_API_KEY=your_tavily_api_key
```

---

## How it works

### Ingestion pipeline (`ingestion.py`)
- Maps the documentation site using Tavily Map
- Extracts raw page content concurrently using Tavily Extract
- Chunks documents with `RecursiveCharacterTextSplitter`
- Stores chunks in Pinecone using OpenAI embeddings

Run ingestion:
```bash
python ingestion.py
```

Notes:
- Default seed site in code: `https://python.langchain.com/`
- Adjust crawl breadth/depth, chunk size/overlap, and batch sizes in `ingestion.py` as needed.

### Chat runtime (`main.py` and `backend/core.py`)
- Uses `ChatOpenAI` with a history-aware retriever
- Pulls prompts from the LangChain Hub
- Retrieves relevant chunks from Pinecone and formats answers with sources

Run the Streamlit app:
```bash
streamlit run main.py
```

---

## Project structure
```
backend/
  core.py          # RAG chain: embeddings, retriever, prompts, chat
ingestion.py       # Crawl, extract, chunk, and index documents to Pinecone
logger.py          # Colored console logging helpers
main.py            # Streamlit UI for chatting with sources
Pipfile            # Dependency management (Pipenv)
README.md          # This file
```

---

## Configuration reference

### Environment variables
- **OPENAI_API_KEY**: OpenAI access for embeddings and chat
- **PINECONE_API_KEY**: Pinecone vector DB access
- **PINECONE_INDEX_NAME**: Index that `ingestion.py` writes to
- **PINECONE_INDEX_NAME_CRAWL**: Index that `backend/core.py` reads from
- **TAVILY_API_KEY**: Required by Tavily extract/map/crawl
- Optional: **PINECONE_ENVIRONMENT** or **PINECONE_HOST** depending on your Pinecone plan

### Embeddings and dimensions
- The code uses `text-embedding-3-small` (dimension 1536). Ensure your Pinecone index matches.

---

## Common issues
- "Index dims mismatch": Recreate the Pinecone index with dimension 1536.
- "Auth errors": Verify `.env` is loaded (the app uses `python-dotenv`). Restart shell after exporting keys.
- "No results / empty answers": Confirm ingestion completed and the chat is pointing to the correct index names.
- SSL errors on macOS: This repo sets `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` via `certifi` in `ingestion.py`.

---

## License
This project is licensed under the terms in `LICENSE`.

---

## Acknowledgements
- LangChain, Pinecone, OpenAI, Tavily, and Streamlit teams and communities.