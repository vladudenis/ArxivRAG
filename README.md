# ArxivRAG - Interactive RAG for arXiv Papers

An interactive Retrieval-Augmented Generation (RAG) system for querying arXiv papers. Enter your query directly; the system expands it, searches arXiv, filters by abstract similarity, downloads top papers, and answers using RAG.

## Overview

- **Query-first workflow**: No manual paper entry. Enter your question and the system fetches relevant papers automatically.
- **Smart pre-filtering**: LLM query expansion → arXiv keyword search (max 25) → abstract similarity filter (top 5) → download and embed.
- **Pluggable chunking**: Choose from 4 strategies at session start (STRUCTURE_AWARE_OVERLAP is the default/recommended).
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

# Query expansion (local vLLM)
QUERY_EXPANSION_BASE_URL=http://localhost:8001/v1
QUERY_EXPANSION_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 3. Start Infrastructure

```powershell
docker-compose up -d
```

This starts:

- **MinIO** (ports 9000, 9001): Stores PDF files
- **Qdrant** (port 6333): Vector database for embeddings and paper metadata
- **vLLM** (port 8001): Query expansion (TinyLlama-1.1B-Chat, GPU mode)

## Usage

### Terminal (interactive)

```powershell
python interactive_rag.py
```

**Workflow**:

1. **Select strategy**: Choose chunking strategy (1–4, default 1)
2. **Enter query**: Type your question (e.g., "How does RAG reduce hallucination?")
3. Per query: expand → search arXiv → filter by abstract → download top 5 → embed → retrieve → answer
4. Type `done` to end session

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

- `POST /query` — Request body: `{"query": "..."}`. Returns `{"answer": "...", "sources": [...]}`.
- `GET /health` — Health check.

**Query expansion**:

- vLLM is included in `docker-compose up -d` and runs on port 8001 (GPU mode).

**Features**:

- Session log saved to `{paper_id}_{timestamp}.md` when terminal session ends
- All data cleared automatically when terminal session ends
- Web interface shows cited sources (paper title, arXiv link) when the LLM uses them
