import h5py
import numpy as np
import json

class VectorStore:
    def __init__(self, filename="arxiv_rag.h5"):
        self.filename = filename

    def save(self, papers, embeddings):
        """
        Saves papers and embeddings to HDF5.
        
        Args:
            papers (list[dict]): List of paper metadata.
            embeddings (np.ndarray): Array of embeddings.
        """
        print(f"Saving {len(papers)} papers and embeddings to {self.filename}...")
        
        # Prepare data for storage
        ids = [p.get('id', '') for p in papers]
        titles = [p.get('title', '') for p in papers]
        abstracts = [p.get('abstract', '') for p in papers]
        # Store full metadata as JSON strings for flexibility
        meta_json = [json.dumps(p) for p in papers]
        
        with h5py.File(self.filename, 'w') as f:
            # Store embeddings
            f.create_dataset("embeddings", data=embeddings)
            
            # Store string data using variable-length string type
            dt = h5py.special_dtype(vlen=str)
            
            dset_ids = f.create_dataset("ids", (len(ids),), dtype=dt)
            dset_ids[:] = ids
            
            dset_titles = f.create_dataset("titles", (len(titles),), dtype=dt)
            dset_titles[:] = titles
            
            dset_abstracts = f.create_dataset("abstracts", (len(abstracts),), dtype=dt)
            dset_abstracts[:] = abstracts
            
            dset_meta = f.create_dataset("metadata", (len(meta_json),), dtype=dt)
            dset_meta[:] = meta_json
            
        print("Save complete.")

    def load(self):
        """
        Loads papers and embeddings from HDF5.
        
        Returns:
            tuple: (papers, embeddings)
        """
        print(f"Loading from {self.filename}...")
        with h5py.File(self.filename, 'r') as f:
            embeddings = f['embeddings'][:]
            
            # Load string datasets
            # They come back as bytes in some h5py versions/configurations, so decode if needed
            meta_json = f['metadata'][:]
            
            papers = []
            for m in meta_json:
                if isinstance(m, bytes):
                    m = m.decode('utf-8')
                papers.append(json.loads(m))
                
        print(f"Loaded {len(papers)} papers.")
        return papers, embeddings

if __name__ == "__main__":
    # Test
    store = VectorStore("test.h5")
    dummy_papers = [{"id": "1", "title": "Test", "abstract": "Abstract"}]
    dummy_emb = np.random.rand(1, 768)
    store.save(dummy_papers, dummy_emb)
    p, e = store.load()
    print(p[0]['title'])
