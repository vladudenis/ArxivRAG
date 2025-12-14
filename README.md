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
VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
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

**For smaller models** (if you have limited GPU memory), edit `docker-compose.yml` and change the vLLM command to:
```yaml
command: >
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  --host 0.0.0.0
  --port 8000
```

### 4. Verify Setup

```powershell
python verify_setup.py
```

This will check:
- Environment configuration
- Python packages
- Docker services (DynamoDB, MinIO, vLLM)

## Usage

### Run Full Evaluation

```powershell
python experiment_runner.py
```

This will:
1. Download 100 papers from arXiv
2. Generate test queries from abstracts
3. Evaluate 10 chunking strategies
4. Generate comprehensive results report

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
├── VLLM_SETUP.md            # vLLM setup guide
└── .env                      # Configuration
```

## Chunking Strategies

| Strategy | Description | Parameters |
|----------|-------------|------------|
| Fixed-500 | Fixed character chunks | 500 chars, 50 overlap |
| Fixed-1000 | Fixed character chunks | 1000 chars, 100 overlap |
| Recursive-500 | Hierarchical splitting | 500 chars, 50 overlap |
| Recursive-1000 | Hierarchical splitting | 1000 chars, 100 overlap |
| Token-256 | Token-based chunks | 256 tokens, 32 overlap |
| Token-512 | Token-based chunks | 512 tokens, 64 overlap |
| Sentence-500 | Sentence grouping | ~500 chars, 100 overlap |
| Sentence-1000 | Sentence grouping | ~1000 chars, 200 overlap |
| Paragraph | Paragraph-based | No size limit, no overlap |
| Paragraph-Overlap | Paragraph-based | No size limit, 100 overlap |

## Evaluation Metrics

- **ROUGE**: Measures n-gram overlap between generated and reference text
- **BLEU**: Measures precision of n-grams in generated text
- **BERTScore**: Semantic similarity using BERT embeddings
- **Recall@k**: Percentage of queries where relevant document is in top-k results

## Notes

- First run will download papers and models (may take time)
- vLLM requires GPU with CUDA support
- For limited GPU memory, use smaller models (see VLLM_SETUP.md)
- Experiment uses 20 test queries by default (configurable in experiment_runner.py)

## Troubleshooting

### vLLM Connection Issues

- Check if vLLM container is running: `docker-compose ps`
- View vLLM logs: `docker-compose logs vllm`
- Test connection: `curl http://localhost:8000/v1/models`
- Restart vLLM: `docker-compose restart vllm`

### Out of Memory (GPU)

- Edit `docker-compose.yml` to use smaller model (TinyLlama, Phi-2)
- Reduce number of test queries in `experiment_runner.py`
- Reduce chunk sizes in strategy definitions

### Docker Issues

- Ensure Docker Desktop is running
- Check GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- Restart services: `docker-compose down && docker-compose up -d`

### Slow Execution

- First run downloads models and papers
- BERTScore calculation can be slow
- Consider reducing number of strategies or test queries for testing
