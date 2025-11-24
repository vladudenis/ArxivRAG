import argparse
import os
import sys
from dotenv import load_dotenv
from src.data_loader import download_papers
from src.embedder import PaperEmbedder
from src.storage import VectorStore
from src.rag import ArxivRAG

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ArxivRAG: Search and Chat with 2025 AI Papers")
    parser.add_argument("--topic", type=str, help="Initial topic to search for")
    args = parser.parse_args()
    
    store_file = "arxiv_rag.h5"
    
    # 1. Get Search Topic
    if args.topic:
        topic = args.topic
    else:
        print("Welcome to ArxivRAG!")
        topic = input("Enter a topic to search for on arXiv: ").strip()
        if not topic:
            print("No topic provided. Exiting.")
            return

    print(f"\nInitializing data pipeline for topic: '{topic}'...")
    
    # 2. Download
    papers = download_papers(query=topic, limit=100, year=2025)
    if not papers:
        print("No papers found! Exiting.")
        return
        
    # 3. Embed
    print("Embedding papers...")
    texts = [f"{p['title']}\n{p['abstract']}" for p in papers]
    
    embedder = PaperEmbedder()
    embeddings = embedder.embed_texts(texts)
    
    # 4. Store
    store = VectorStore(store_file)
    store.save(papers, embeddings)
    print("Database built successfully.")

    # 5. RAG / Query Loop
    rag = ArxivRAG(store, embedder)
    
    print("\nInteractive Mode (type 'exit' to quit, 'new' to start over)")
    while True:
        q = input("\nEnter your question about these papers: ")
        if q.lower() in ['exit', 'quit']:
            break
        if q.lower() == 'new':
            # Recursive call or just return to let user restart? 
            print("\nRestarting session...\n")
            main() 
            return
            
        if not q.strip():
            continue
            
        response = rag.query(q)
        print("\n" + "-"*50)
        print(response['answer'])
        print("-"*50)

if __name__ == "__main__":
    main()
