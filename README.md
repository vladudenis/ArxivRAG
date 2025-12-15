# ArxivRAG - RAG Evaluation System

A comprehensive evaluation system for comparing different chunking strategies in Retrieval-Augmented Generation (RAG) pipelines.

## Features

- **Multiple Chunking Strategies**: 10 different strategies including:
  - Fixed-size (character-based)
  - Recursive (hierarchical separators)
  - Token-based (using tiktoken)
  - Sentence-based (using NLTK)
  - Paragraph-based
  - Overlapping variants

- **Comprehensive Metrics**:
  - ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
  - BLEU
  - BERTScore
  - Recall@k

- **Full RAG Pipeline**:
  - Document chunking
  - Embedding generation
  - Top-k retrieval
  - vLLM-powered text generation
  - Automated evaluation

## Setup

### 1. Install Dependencies

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Create/update `.env` file:

```
HF_TOKEN=your_huggingface_token
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 3. Start All Services with Docker Compose

The easiest way to run the entire stack (DynamoDB, MinIO, and vLLM):

```powershell
# Start all services in background
docker-compose up -d

# Check status
docker-compose ps

# View vLLM logs (first time will download model)
docker-compose logs -f vllm
```

**Note**: First startup will download the Llama-2-7b model (~13GB). This may take 10-30 minutes depending on your internet connection.

## Usage

### Run Full Evaluation

```powershell
python experiment_runner.py
```

This will:
1. Download 100 papers from arXiv (if not already downloaded)
2. Generate test queries from abstracts
3. Evaluate 5 chunking strategies
4. Generate comprehensive results report

**Options:**
- `--skip-ingestion`: Skip downloading and processing papers, use existing data (assumes data is already in storage).

### Output Files

- `experiment_results.md` - Comprehensive markdown report
- `experiment_results.json` - Detailed JSON results

## Project Structure

```
ArxivRAG/
├── src/
│   ├── chunking.py          # Chunking strategies
│   ├── embedder.py           # Embedding generation
│   ├── evaluation.py         # Evaluation pipeline
│   ├── llm_client.py         # vLLM client
│   ├── metrics.py            # Evaluation metrics
│   ├── test_queries.py       # Query generation
│   ├── results_analyzer.py   # Results analysis
│   ├── data_loader.py        # arXiv data loader
│   └── storage_manager.py    # DynamoDB/MinIO interface
├── experiment_runner.py      # Main experiment script
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Infrastructure setup
└── .env                      # Configuration
```

## Chunking Strategies

| Strategy | Description | Parameters |
|----------|-------------|------------|
| Fixed-500 | Fixed character chunks | 500 chars, 50 overlap |
| Recursive-500 | Hierarchical splitting | 500 chars, 50 overlap |
| Token-256 | Token-based chunks | 256 tokens, 32 overlap |
| Sentence-500 | Sentence grouping | ~500 chars, 100 overlap |
| Paragraph-Overlap | Paragraph-based | No size limit, 100 overlap |

## Evaluation Metrics

- **ROUGE**: Measures n-gram overlap between generated and reference text
- **BLEU**: Measures precision of n-grams in generated text
- **BERTScore**: Semantic similarity using BERT embeddings
- **Recall@k**: Percentage of queries where relevant document is in top-k results