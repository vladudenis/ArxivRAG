import random
import os
import argparse
from dotenv import load_dotenv
from src.storage_manager import StorageManager
from src.embedder import PaperEmbedder
from src.evaluation import Evaluator
from src.data_loader import download_papers
from src.llm_client import VLLMClient
from src.test_queries import load_test_queries
from src.results_analyzer import ResultsAnalyzer

def main():
    # 0. Load Environment
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip downloading and processing papers, use existing data")
    args = parser.parse_args()
    
    print("Initialize Storage and Embedder...")
    storage = StorageManager()
    storage.init_qdrant()

    papers = []

    if not args.skip_ingestion:
        # 1. Clear Storage (Fresh Start)
        print("\n--- Phase 1: Cleaning Storage ---")
        storage.reset_db()
        storage.reset_bucket()
        storage.reset_qdrant()
        
        # 2. Download Data
        print("\n--- Phase 2: Downloading Data ---")
        # Finding exactly 100 papers
        papers = download_papers(limit=100)
        
        if not papers:
            print("No papers downloaded. Exiting.")
            return
            
        print(f"Downloaded and stored {len(papers)} papers.")
    else:
        print("\n--- Phase 1 & 2: Skipping Ingestion (Using Existing Data) ---")
        papers = storage.get_all_metadata()
        if not papers:
            print("No papers found in storage. Cannot skip ingestion. Exiting.")
            return
        print(f"Loaded {len(papers)} papers from storage.")
    
    # 3. Initialize Embedder
    print("\n--- Phase 3: Initializing Embedder ---")
    embedder = PaperEmbedder(model_name="google/embeddinggemma-300m")
    
    # 4. Initialize vLLM Client
    print("\n--- Phase 4: Initializing vLLM Client ---")
    try:
        llm_client = VLLMClient()
        print("vLLM client initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize vLLM client: {e}")
        print("Will skip RAG evaluation and only run basic retrieval evaluation")
        llm_client = None
    
    # 5. Generate Test Queries
    print("\n--- Phase 5: Generating Test Queries ---")
    # Use subset of papers for queries (e.g., 20 queries)
    num_test_queries = min(20, len(papers))
    test_queries = load_test_queries(papers, num_queries=num_test_queries)
    print(f"Generated {len(test_queries)} test queries")
    
    # 6. Define Strategies
    strategies = [
        # Character-based
        {"name": "Fixed-500", "strategy": "fixed", "chunk_size": 500, "overlap": 50},
        
        # Recursive
        {"name": "Recursive-500", "strategy": "recursive", "chunk_size": 500, "overlap": 50},
        
        # Token-based
        {"name": "Token-256", "strategy": "token", "chunk_size": 256, "overlap": 32},
        
        # Sentence-based
        {"name": "Sentence-500", "strategy": "sentence", "chunk_size": 500, "overlap": 100},
        
        # Paragraph-based
        {"name": "Paragraph-Overlap", "strategy": "paragraph", "chunk_size": 0, "overlap": 100}
    ]
    
    # 7. Run Experiments
    evaluator = Evaluator(embedder, storage, llm_client)
    results = []
    print("\n--- Phase 6: Running Experiments ---")
    
    # Use all downloaded papers for evaluation
    test_papers = papers
    
    for i, strat in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] Testing Strategy: {strat['name']}")
        
        if llm_client is not None:
            # Run full RAG experiment with generation and comprehensive metrics
            try:
                result = evaluator.run_rag_experiment(
                    test_papers,
                    test_queries,
                    strategy=strat['strategy'],
                    chunk_size=strat['chunk_size'],
                    chunk_overlap=strat['overlap'],
                    top_k=5
                )
                result['name'] = strat['name']
                results.append(result)
                print(f"Completed: {strat['name']}")
            except Exception as e:
                print(f"Error running RAG experiment for {strat['name']}: {e}")
                # Fall back to basic retrieval experiment
                print("Falling back to basic retrieval experiment...")
                metrics = evaluator.run_experiment(
                    test_papers,
                    strategy=strat['strategy'],
                    chunk_size=strat['chunk_size'],
                    chunk_overlap=strat['overlap']
                )
                results.append({**strat, **metrics})
        else:
            # Run basic retrieval experiment only
            metrics = evaluator.run_experiment(
                test_papers,
                strategy=strat['strategy'],
                chunk_size=strat['chunk_size'],
                chunk_overlap=strat['overlap']
            )
            results.append({**strat, **metrics})
            print(f"Results: {metrics}\n")
    
    # 8. Generate Reports
    print("\n--- Phase 7: Generating Reports ---")
    
    if llm_client is not None and results:
        # Generate comprehensive markdown report
        analyzer = ResultsAnalyzer(results)
        analyzer.generate_markdown_report("experiment_results.md")
        analyzer.save_json("experiment_results.json")
    else:
        # Generate basic report (backward compatible)
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        header = f"{'Strategy':<25} | {'Hit Rate':<10} | {'MRR':<10} | {'Chunks'}"
        print(header)
        print("-" * 65)
        
        with open("experiment_results.md", "w", encoding="utf-8") as f:
            f.write("# Experiment Results\n\n")
            f.write("| Strategy | Hit Rate | MRR | Num Chunks |\n")
            f.write("|----------|----------|-----|------------|\n")
            
            for res in results:
                row_str = f"{res['name']:<25} | {res.get('hit_rate', 0):.4f}     | {res.get('mrr', 0):.4f}     | {res.get('num_chunks', 0)}"
                print(row_str)
                f.write(f"| {res['name']} | {res.get('hit_rate', 0):.4f} | {res.get('mrr', 0):.4f} | {res.get('num_chunks', 0)} |\n")
                
        print("="*50)
        print("Results saved to experiment_results.md")

if __name__ == "__main__":
    main()

