import arxiv
from datetime import datetime, timezone

def download_papers(query="cat:cs.AI OR cat:cs.ML", year=2025, limit=100):
    """
    Downloads metadata for papers from arXiv matching the query and year.
    
    Args:
        query (str): The search query for arXiv.
        year (int): The year to filter papers by.
        limit (int): The maximum number of papers to return.
        
    Returns:
        list[dict]: A list of dictionaries containing paper metadata.
    """
    client = arxiv.Client()
    
    # Construct a search query that includes the date range for the specified year
    # arXiv date format: YYYYMMDDHHMM
    start_date = f"{year}01010000"
    end_date = f"{year}12312359"
    
    # Note: arXiv API query syntax for date is 'submittedDate:[start TO end]'
    # However, mixing 'cat' and 'submittedDate' can be tricky with the python wrapper's sort order.
    # We will fetch a bit more and filter client-side to be safe and simple, 
    # but we can try to optimize the query.
    # Let's try to just search for the categories and sort by submitted date descending.
    
    search = arxiv.Search(
        query=query,
        max_results=limit * 2, # Fetch more to allow for filtering if needed
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = []
    
    print(f"Searching arXiv for {query}...")
    
    for result in client.results(search):
        if result.published.year == year:
            paper_info = {
                "id": result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat(),
                "url": result.pdf_url
            }
            results.append(paper_info)
            if len(results) >= limit:
                break
        elif result.published.year < year:
            # Since we sort by descending date, if we hit a paper from before the target year,
            # we can stop (assuming strictly sorted).
            # But sometimes there are updates, so we'll just continue until we have enough or run out.
            # Actually, 'published' is the initial publish date. 'updated' is update date.
            # We'll stick to 'published.year == year'.
            pass
            
    print(f"Found {len(results)} papers from {year}.")
    return results

if __name__ == "__main__":
    # Test run
    papers = download_papers(limit=5)
    for p in papers:
        print(f"[{p['published']}] {p['title']}")
