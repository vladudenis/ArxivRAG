import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ArxivRAG:
    def __init__(self, vector_store, embedder):
        """
        Initializes the RAG system.
        
        Args:
            vector_store: Instance of VectorStore with loaded data.
            embedder: Instance of PaperEmbedder.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.papers, self.embeddings = self.vector_store.load()
        
    def query(self, user_query, k=5):
        """
        Searches for papers relevant to the query.
        
        Args:
            user_query (str): The user's question.
            k (int): Number of results to return.
            
        Returns:
            dict: Contains 'results' (list of papers) and 'answer' (str).
        """
        # Embed query
        query_emb = self.embedder.embed_texts([user_query])
        
        # Calculate similarity
        # embeddings is (N, D), query_emb is (1, D)
        # Result is (1, N)
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            paper = self.papers[idx]
            paper['score'] = float(scores[idx])
            results.append(paper)
            
        # Formulate an "answer"
        # Since we don't have a generative LLM here, we construct a context-based response.
        answer = f"Based on the top {k} papers, here are the relevant findings:\n\n"
        for i, res in enumerate(results):
            answer += f"{i+1}. **{res['title']}** (Score: {res['score']:.4f})\n"
            answer += f"   {res['abstract'][:200]}...\n\n"
            
        return {
            "results": results,
            "answer": answer
        }

if __name__ == "__main__":
    # Mock classes for testing would go here, or integration test in main.
    pass
