import uuid
import boto3
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient
from qdrant_client.http import models


class StorageManager:
    def __init__(self):
        # MinIO settings
        self.s3_endpoint = "http://localhost:9000"
        self.bucket_name = "arxiv-pdfs"
        self.minio_access_key = "minioadmin"
        self.minio_secret_key = "minioadmin"

        # Qdrant settings
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.qdrant_collection = "arxiv_chunks"
        self.papers_collection = "arxiv_papers"

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            region_name="us-east-1"
        )
        self.qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

    def init_db(self):
        """Creates papers collection in Qdrant if it doesn't exist."""
        try:
            self.qdrant.get_collection(self.papers_collection)
            print(f"Papers collection {self.papers_collection} already exists.")
        except Exception:
            print(f"Creating papers collection {self.papers_collection}...")
            self.qdrant.create_collection(
                collection_name=self.papers_collection,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
            )
            print(f"Papers collection {self.papers_collection} created.")

    def init_bucket(self):
        """Creates MinIO bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists.")
        except ClientError:
            print(f"Creating bucket {self.bucket_name}...")
            self.s3.create_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} created.")

    def save_paper_metadata(self, paper_dict: dict):
        """Saves paper metadata to Qdrant papers collection."""
        paper_id = paper_dict.get("id", "")
        if not paper_id:
            return
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"paper_{paper_id}"))
        payload = {k: str(v) if not isinstance(v, (list, dict)) else v for k, v in paper_dict.items()}
        self.qdrant.upsert(
            collection_name=self.papers_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload=payload
                )
            ]
        )

    def save_paper_pdf(self, paper_id: str, pdf_bytes: bytes):
        """Saves paper PDF to MinIO."""
        key = f"{paper_id}.pdf"
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=pdf_bytes,
            ContentType='application/pdf'
        )

    def get_paper_pdf(self, paper_id: str) -> bytes:
        """Retrieves paper PDF bytes from MinIO."""
        key = f"{paper_id}.pdf"
        response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()

    def get_all_metadata(self) -> list:
        """Retrieves all paper metadata from Qdrant papers collection."""
        results = []
        next_offset = None
        while True:
            points, next_offset = self.qdrant.scroll(
                collection_name=self.papers_collection,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_offset
            )
            for p in points:
                if p.payload:
                    results.append(dict(p.payload))
            if next_offset is None:
                break
        return results

    def reset_db(self):
        """Deletes and recreates the papers collection."""
        try:
            self.qdrant.delete_collection(self.papers_collection)
            print(f"Papers collection {self.papers_collection} deleted.")
        except Exception:
            pass
        self.init_db()

    def reset_bucket(self):
        """Deletes all objects in the bucket and recreates it."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                objects = [{'Key': obj['Key']} for obj in response['Contents']]
                for i in range(0, len(objects), 1000):
                    self.s3.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects[i:i+1000]}
                    )
                print(f"Bucket {self.bucket_name} cleared.")
        except ClientError as e:
            print(f"Error clearing bucket: {e}")

    def init_qdrant(self, vector_size: int = 768):
        """
        Creates Qdrant chunks collection if it doesn't exist.

        Args:
            vector_size: Dimension of embedding vectors (default: 768)
        """
        try:
            collection_info = self.qdrant.get_collection(self.qdrant_collection)
            print(f"Collection {self.qdrant_collection} already exists.")
            existing_size = collection_info.config.params.vectors.size
            if existing_size != vector_size:
                print(f"Warning: Collection vector size ({existing_size}) doesn't match expected ({vector_size})")
        except Exception:
            print(f"Creating collection {self.qdrant_collection} with vector size {vector_size}...")
            self.qdrant.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            print(f"Collection {self.qdrant_collection} created.")

    def reset_qdrant(self):
        """Recreates the Qdrant chunks collection."""
        try:
            self.qdrant.delete_collection(self.qdrant_collection)
            print(f"Collection {self.qdrant_collection} deleted.")
        except Exception:
            pass
        self.init_qdrant()

    def save_embeddings(self, vectors, payloads):
        """
        Saves embeddings and metadata to Qdrant.

        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dicts (must include strategy, paper_id, chunk_text)
        """
        batch_size = 100
        total = len(vectors)

        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_vectors = vectors[i:end]
            batch_payloads = payloads[i:end]

            points = []
            for v, p in zip(batch_vectors, batch_payloads):
                unique_str = f"{p.get('strategy')}_{p.get('paper_id')}_{p.get('chunk_text', '')[:50]}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
                points.append(models.PointStruct(
                    id=point_id,
                    vector=v.tolist() if hasattr(v, 'tolist') else v,
                    payload=p
                ))

            self.qdrant.upsert(
                collection_name=self.qdrant_collection,
                points=points
            )

    def fetch_embeddings(self, strategy: str):
        """
        Fetches all embeddings and payloads for a given strategy.

        Args:
            strategy: Strategy name to filter by

        Returns:
            tuple: (vectors, payloads)
        """
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="strategy",
                    match=models.MatchValue(value=strategy)
                )
            ]
        )

        points = []
        next_offset = None

        while True:
            result, next_offset = self.qdrant.scroll(
                collection_name=self.qdrant_collection,
                scroll_filter=filter_condition,
                limit=1000,
                with_payload=True,
                with_vectors=True,
                offset=next_offset
            )
            points.extend(result)
            if next_offset is None:
                break

        if not points:
            return [], []

        vectors = [p.vector for p in points]
        payloads = [p.payload for p in points]

        return vectors, payloads


if __name__ == "__main__":
    sm = StorageManager()
    sm.init_db()
    sm.init_bucket()
    sm.init_qdrant()
