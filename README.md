# ArxivRAG - RAG for arXiv Papers

A Retrieval-Augmented Generation (RAG) system for querying arXiv papers. Enter topics for search and your question; the system fetches relevant papers, embeds them, and answers using RAG.

## Overview

- **Topics + query workflow**: Enter comma-separated topics for arXiv search and a natural language question for embedding/retrieval.
- **Pre-filtering**: arXiv keyword search (topics) → abstract similarity filter (query) → download and embed.
- **Pluggable chunking**: STRUCTURE_AWARE_OVERLAP is the default strategy.
- **Storage**: MinIO for PDFs, Qdrant for embeddings and paper metadata.

## Chunking Strategies

1. **STRUCTURE_AWARE_OVERLAP** (default, recommended)
   - Preserves academic structure (Abstract, sections, subsections)
   - Target 500–800 tokens, max 900; 10–15% overlap
   - Abstract and Conclusion as standalone chunks; References excluded

2. **SEMANTIC_PARAGRAPH_GROUPING**
   - Groups paragraphs by embedding similarity (threshold ~0.75)
   - Min 300, max 900 tokens; topic-coherent chunks

3. **FIXED_WINDOW_OVERLAP** (baseline)
   - Sliding window: 700 tokens, 150 overlap
   - Simple and fast

4. **SECTION_LEVEL_CHUNKING**
   - One chunk per subsection; split at midpoint if >1500 tokens
   - Maximum semantic integrity

## Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose

### 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` in the project root:

```env
HF_TOKEN=your_huggingface_token_here

# DeepSeek API (required for query answering)
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_deepseek_api_key_here
```

### 3. Start Infrastructure

```powershell
docker-compose up -d
```

This starts:

- **MinIO** (ports 9000, 9001): Stores PDF files
- **Qdrant** (port 6333): Vector database for embeddings and paper metadata

## Usage

### Web interface (FastAPI + Next.js)

1. **Start the API** (from project root):

```powershell
uvicorn src.api.main:app --reload --port 8000
```

2. **Start the frontend**:

```powershell
cd frontend
npm run dev
```

3. Open http://localhost:3000 for a ChatGPT-like chat interface.

**API**:

- `POST /query` — Request body: `{"query": "...", "topics": "..."}`. Both fields required. Topics: comma-separated terms for arXiv search. Query: natural language question for embedding and retrieval. Returns `{"answer": "...", "sources": [...]}`.
- `GET /health` — Health check.

**Features**:

- Topics field: type terms and press comma to lock each term
- Web interface shows cited sources (paper title, arXiv link) when the LLM uses them
