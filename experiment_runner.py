import random
import argparse
import json
import numpy as np
from dotenv import load_dotenv
from src.storage_manager import StorageManager
from src.embedder import PaperEmbedder
from src.data_loader import download_papers
from src.llm_client import VLLMClient
from src.test_queries import load_test_queries
from src.stage1_evaluator import Stage1Evaluator
from src.stage2_evaluator import Stage2Evaluator
from src.stage3_evaluator import Stage3Evaluator

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define the 5 chunking strategies
CHUNKING_STRATEGIES = [
    {
        'name': 'fixed_token',
        'strategy': 'fixed_token',
        'chunk_size': 512,
        'chunk_overlap': 64  # ~12.5% overlap
    },
    {
        'name': 'section',
        'strategy': 'section',
        'chunk_size': 0,  # Not used for section-based
        'chunk_overlap': 0
    },
    {
        'name': 'paragraph',
        'strategy': 'paragraph',
        'chunk_size': 0,  # Not used for paragraph-based
        'chunk_overlap': 0
    },
    {
        'name': 'sentence_sliding',
        'strategy': 'sentence_sliding',
        'chunk_size': 512,  # Characters
        'chunk_overlap': 64
    },
    {
        'name': 'section_hybrid',
        'strategy': 'section_hybrid',
        'chunk_size': 512,  # Tokens
        'chunk_overlap': 64
    }
]



def run_single_stage(stage_num: int, storage, embedder, papers, queries, best_strategy=None):
    """
    Run a single stage in isolation.
    
    Args:
        stage_num: Stage number (1, 2, or 3)
        storage: StorageManager instance
        embedder: PaperEmbedder instance
        papers: List of paper metadata dicts
        queries: List of query dicts
        best_strategy: Best strategy dict (required for stage 3)
    """
    if stage_num == 1:
        print("\n" + "="*80)
        print("RUNNING STAGE 1 ONLY: DOCUMENT-LEVEL RETRIEVAL EVALUATION")
        print("="*80)
        
        stage1_evaluator = Stage1Evaluator(embedder, storage)
        results = stage1_evaluator.run_all_strategies(papers, queries, CHUNKING_STRATEGIES, top_k=5)
        stage1_evaluator.save_results(results)
        
    elif stage_num == 2:
        print("\n" + "="*80)
        print("RUNNING STAGE 2 ONLY: CHUNK-LEVEL EVIDENCE RETRIEVAL EVALUATION")
        print("="*80)
        
        stage2_evaluator = Stage2Evaluator(embedder, storage)
        results = stage2_evaluator.run_all_strategies(papers, queries, CHUNKING_STRATEGIES, top_k=5)
        stage2_evaluator.save_results(results)
        
    elif stage_num == 3:
        print("\n" + "="*80)
        print("RUNNING STAGE 3 ONLY: INFERENCE-BASED QUALITATIVE VALIDATION")
        print("="*80)
        
        # Use section-based chunking by default for Stage 3
        if best_strategy is None:
            print("No best strategy provided, using section-based chunking by default")
            best_strategy = {
                'strategy_name': 'section',
                'strategy': 'section',
                'chunk_size': 0,
                'chunk_overlap': 0
            }
        
        try:
            llm_client = VLLMClient()
            print("vLLM client initialized successfully")
        except Exception as e:
            import traceback
            print(f"ERROR: Could not initialize vLLM client: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Skipping Stage 3 validation")
            return
        
        stage3_evaluator = Stage3Evaluator(embedder, storage, llm_client)
        validation_results = stage3_evaluator.validate_strategy(
            papers=papers,
            queries=queries,
            strategy=best_strategy['strategy'],
            chunk_size=best_strategy['chunk_size'],
            chunk_overlap=best_strategy['chunk_overlap'],
            top_k=3,
            max_papers=10
        )
        stage3_evaluator.save_results(validation_results, best_strategy)
        
    else:
        raise ValueError(f"Invalid stage number: {stage_num}. Must be 1, 2, or 3.")

def load_best_strategy_from_results():
    """
    Load best strategy from existing Stage 1 and Stage 2 results.
    
    Returns:
        Best strategy dict, or None if results not found
    """
    try:
        with open('stage1_results.json', 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        with open('stage2_results.json', 'r', encoding='utf-8') as f:
            stage2_data = json.load(f)
        
        # Extract results (handle both old phase1 format and new format)
        if 'phase1' in stage1_data:
            stage1_results = stage1_data['phase1']['results']
        else:
            stage1_results = stage1_data.get('results', [])
        
        if 'phase1' in stage2_data:
            stage2_results = stage2_data['phase1']['results']
        else:
            stage2_results = stage2_data.get('results', [])
        
        if not stage1_results or not stage2_results:
            return None
        
        # Convert to format expected by select_best_strategy
        # (combine stage1 and stage2 metrics)
        combined_results = []
        stage1_dict = {r['strategy_name']: r for r in stage1_results}
        stage2_dict = {r['strategy_name']: r for r in stage2_results}
        
        for strategy_name in stage1_dict.keys():
            if strategy_name in stage2_dict:
                combined_results.append({
                    'strategy_name': strategy_name,
                    'strategy': stage1_dict[strategy_name]['strategy'],
                    'chunk_size': stage1_dict[strategy_name]['chunk_size'],
                    'chunk_overlap': stage1_dict[strategy_name]['chunk_overlap'],
                    'stage1': stage1_dict[strategy_name].get('metrics', stage1_dict[strategy_name]),
                    'stage2': stage2_dict[strategy_name].get('metrics', stage2_dict[strategy_name])
                })
        
        if combined_results:
            return Stage2Evaluator.select_best_strategy(stage1_results, stage2_results)
        
        return None
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        return None

def main():
    """
    Main experiment runner.
    
    Execution Flow:
    ===============
    
    FULL PIPELINE:
    - Runs Stages 1 & 2 for ALL 5 chunking strategies
    - Then runs Stage 3 for BEST strategy only
    - Uses same fixed set of 100 papers
    - skip_ingestion controls: reuse existing papers (true) or download new (false)
    
    STAGES:
    - Stage 1: Document-level retrieval (paper identity)
    - Stage 2: Chunk-level evidence retrieval (section-based)
    - Stage 3: Inference-based qualitative validation (best strategy only)
    
    ISOLATED STAGE EXECUTION:
    - Use --stage 1, 2, or 3 to run a single stage in isolation
    - Stage 3 uses section-based chunking by default if no best strategy provided
    """
    # Load environment
    load_dotenv()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run chunking evaluation experiments")
    parser.add_argument(
        "--skip-ingestion",
        type=str,
        choices=['true', 'false'],
        default='false',
        help="Skip downloading papers, use existing data (true) or fetch new papers (false)."
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a single stage in isolation (1, 2, or 3). Stage 3 uses section-based chunking by default."
    )
    args = parser.parse_args()
    
    skip_ingestion = args.skip_ingestion.lower() == 'true'
    run_single_stage_only = args.stage is not None
    
    # Initialize components
    print("Initializing components...")
    storage = StorageManager()
    storage.init_db()
    storage.init_bucket()
    
    embedder = PaperEmbedder(model_name="google/embeddinggemma-300m")
    
    # Detect embedding dimension
    test_embedding = embedder.embed_texts(["test"])
    embedding_dim = test_embedding.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")
    storage.init_qdrant(vector_size=embedding_dim)
    
    if skip_ingestion:
        # Load existing papers
        papers = storage.get_all_metadata()
        if not papers:
            print("Error: No papers found in storage. Cannot skip ingestion.")
            return
        print(f"Loaded {len(papers)} papers from storage")
    else:
        # Clear storage and download fresh papers
        print("Clearing storage for fresh start...")
        storage.reset_db()
        storage.reset_bucket()
        storage.reset_qdrant()
        
        print("Downloading papers...")
        papers = download_papers(limit=100)
        if not papers:
            print("Error: No papers downloaded.")
            return
    
    # Generate queries
    queries = load_test_queries(papers, num_papers=len(papers))
    print(f"Generated {len(queries)} queries")
    
    # If running a single stage, execute it and exit
    if run_single_stage_only:
        best_strategy = None
        # For stage 3, try to load best strategy from existing stage results if available
        if args.stage == 3:
            best_strategy = load_best_strategy_from_results()
            if best_strategy:
                print(f"Loaded best strategy from stage results: {best_strategy['strategy_name']}")
            else:
                print("No stage results found, using section-based chunking by default")
        
        run_single_stage(args.stage, storage, embedder, papers, queries, best_strategy)
        print("\n" + "="*80)
        print(f"STAGE {args.stage} COMPLETE")
        print("="*80)
        return
    
    # Run full pipeline: Stages 1 & 2 for all strategies
    print("\n" + "="*80)
    print("RUNNING FULL PIPELINE: STAGES 1 & 2")
    print("="*80)
    print(f"Using fixed set of {len(papers)} papers")
    print(f"Evaluating {len(queries)} queries")
    
    # Stage 1: Document-level retrieval
    print("\n" + "="*80)
    print("STAGE 1: DOCUMENT-LEVEL RETRIEVAL EVALUATION")
    print("="*80)
    stage1_evaluator = Stage1Evaluator(embedder, storage)
    stage1_results = stage1_evaluator.run_all_strategies(papers, queries, CHUNKING_STRATEGIES, top_k=5)
    stage1_evaluator.save_results(stage1_results)
    
    # Stage 2: Chunk-level evidence retrieval
    print("\n" + "="*80)
    print("STAGE 2: CHUNK-LEVEL EVIDENCE RETRIEVAL EVALUATION")
    print("="*80)
    stage2_evaluator = Stage2Evaluator(embedder, storage)
    stage2_results = stage2_evaluator.run_all_strategies(papers, queries, CHUNKING_STRATEGIES, top_k=5)
    stage2_evaluator.save_results(stage2_results)
    
    # Select best strategy based on Stages 1 & 2 metrics
    best_strategy = Stage2Evaluator.select_best_strategy(stage1_results, stage2_results)
    
    # Stage 3: Qualitative validation (only for best strategy)
    print("\n" + "="*80)
    print("STAGE 3: INFERENCE-BASED QUALITATIVE VALIDATION")
    print("="*80)
    print(f"Validating best strategy: {best_strategy['strategy_name']}")
    
    try:
        llm_client = VLLMClient()
        print("vLLM client initialized successfully")
    except Exception as e:
        import traceback
        print(f"ERROR: Could not initialize vLLM client: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Skipping Stage 3 validation")
    else:
        stage3_evaluator = Stage3Evaluator(embedder, storage, llm_client)
        validation_results = stage3_evaluator.validate_strategy(
            papers=papers,
            queries=queries,
            strategy=best_strategy['strategy'],
            chunk_size=best_strategy['chunk_size'],
            chunk_overlap=best_strategy['chunk_overlap'],
            top_k=3,
            max_papers=10
        )
        stage3_evaluator.save_results(validation_results, best_strategy)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
