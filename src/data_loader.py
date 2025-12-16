import arxiv
import requests
import time
from src.storage_manager import StorageManager

def download_pdf(url):
    """Downloads PDF content from a URL."""
    try:
        # Arxiv PDF links often need 'pdf' instead of 'abs' and .pdf extension logic 
        # But the API returns direct PDF link usually.
        response = requests.get(url, headers={'User-Agent': 'ArxivRAG-Experiment/1.0'})
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to download PDF from {url}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        return None

def download_papers(query="cat:cs.AI OR cat:cs.ML", year=2025, limit=100):
    """
    Downloads metadata and PDFs from arXiv, saving them to storage.
    """
    # Initialize storage
    storage = StorageManager()
    storage.init_db()
    storage.init_bucket()
    
    client = arxiv.Client()
    
    search = arxiv.Search(
        query=query,
        max_results=limit * 2, 
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = []
    count = 0
    
    print(f"Searching arXiv for {query}...")
    
    for result in client.results(search):
        if result.published.year == year:
            paper_id = result.entry_id.split('/')[-1]
            
            # Check if likely already exists
            
            paper_info = {
                "id": paper_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat(),
                "url": result.pdf_url,
                "categories": result.categories # Store categories too
            }
            
            print(f"Processing: {result.title[:50]}...")
            
            # Download PDF
            pdf_bytes = download_pdf(result.pdf_url)
            if pdf_bytes:
                # Save to Storage
                storage.save_paper_metadata(paper_info)
                storage.save_paper_pdf(paper_id, pdf_bytes)
                
                results.append(paper_info)
                count += 1
                if count >= limit:
                    break
                
                # Be nice to arXiv API
                time.sleep(0.5)
            else:
                print(f"Skipping {paper_id} due to PDF download failure.")
                
        elif result.published.year < year:
            # Sorted by date descending
            pass
            
    print(f"Successfully downloaded and stored {len(results)} papers.")
    return results

if __name__ == "__main__":
    download_papers(limit=5)
