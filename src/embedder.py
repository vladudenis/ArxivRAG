from sentence_transformers import SentenceTransformer
import numpy as np

class PaperEmbedder:
    def __init__(self, model_name="google/embeddinggemma-300m"):
        """
        Initializes the embedder with the specified model.
        """
        print(f"Loading model {model_name}...")
        # Trust remote code might be needed for some new models, but let's try without first or check docs.
        # Usually sentence-transformers handles it well.
        # If this fails, we might need to use transformers directly.
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load with SentenceTransformer: {e}")
            print("Falling back to transformers (not implemented in this snippet, assuming ST works or user has it).")
            raise e

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        
        Args:
            texts (list[str]): List of strings to embed.
            
        Returns:
            np.ndarray: Array of embeddings.
        """
        print(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

if __name__ == "__main__":
    embedder = PaperEmbedder()
    emb = embedder.embed_texts(["This is a test paper.", "Another paper about AI."])
    print(f"Embedding shape: {emb.shape}")
