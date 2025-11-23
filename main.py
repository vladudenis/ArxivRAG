import argparse
import os
import sys
from src.data_loader import download_papers
from src.embedder import PaperEmbedder
from src.storage import VectorStore
from src.rag import ArxivRAG

def main():
    parser = argparse.ArgumentParser(description="ArxivRAG: Search and Chat with 2025 AI Papers")
    parser.add_argument("--refresh", action="store_true", help="Force refresh of data (download and embed again)")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    args = parser.parse_args()
    
    store_file = "arxiv_rag.h5"
    
    # Check if we need to build the database
    if args.refresh or not os.path.exists(store_file):
        print("Initializing data pipeline...")
        
        # 1. Download
        papers = download_papers(limit=100, year=2025)
        if not papers:
            print("No papers found! Exiting.")
            return
            
        # 2. Embed
        # Extract texts to embed (title + abstract is usually good)
        texts = [f"{p['title']}\n{p['abstract']}" for p in papers]
        
        embedder = PaperEmbedder()
        embeddings = embedder.embed_texts(texts)
        
        # 3. Store
        store = VectorStore(store_file)
        store.save(papers, embeddings)
        print("Database built successfully.")
    else:
        print("Using existing database.")

    # 4. RAG / Query Loop
    # Load resources
    # We need the embedder to embed the query
    embedder = PaperEmbedder()
    store = VectorStore(store_file)
    rag = ArxivRAG(store, embedder)
    
    if args.query:
        response = rag.query(args.query)
        print("\n" + "="*50)
        print(f"Query: {args.query}")
        print("="*50)
        print(response['answer'])
    else:
        print("\nInteractive Mode (type 'exit' to quit)")
        while True:
            q = input("\nEnter your question: ")
            if q.lower() in ['exit', 'quit']:
                break
            if not q.strip():
                continue
                
            response = rag.query(q)
            print("\n" + "-"*50)
            print(response['answer'])
            print("-"*50)

if __name__ == "__main__":
    main()
