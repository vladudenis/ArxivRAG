import boto3
import os
import io
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient
from qdrant_client.http import models

class StorageManager:
    def __init__(self):
        # DynamoDB settings
        self.dynamodb_endpoint = "http://localhost:8001"
        self.region = "us-east-1"
        self.table_name = "ArxivPapers"
        
        # MinIO settings
        self.s3_endpoint = "http://localhost:9000"
        self.bucket_name = "arxiv-pdfs"
        self.minio_access_key = "minioadmin"
        self.minio_secret_key = "minioadmin"

        # Qdrant settings
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.qdrant_collection = "arxiv_chunks"

        # Initialize clients
        self.dynamodb = boto3.resource(
            'dynamodb',
            endpoint_url=self.dynamodb_endpoint,
            region_name=self.region,
            aws_access_key_id="dummy",
            aws_secret_access_key="dummy"
        )
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            region_name=self.region
        )
        
        self.qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

    def init_db(self):
        """Creates DynamoDB table if it doesn't exist."""
        try:
            table = self.dynamodb.Table(self.table_name)
            table.load()
            print(f"Table {self.table_name} already exists.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"Creating table {self.table_name}...")
                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {'AttributeName': 'id', 'KeyType': 'HASH'}  # Partition key
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'id', 'AttributeType': 'S'}
                    ],
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                table.wait_until_exists()
                print(f"Table {self.table_name} created.")
            else:
                raise e

    def init_bucket(self):
        """Creates MinIO bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists.")
        except ClientError:
            print(f"Creating bucket {self.bucket_name}...")
            self.s3.create_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} created.")

    def save_paper_metadata(self, paper_dict):
        """Saves paper metadata to DynamoDB."""
        table = self.dynamodb.Table(self.table_name)
        # decimal conversion might be needed for float values if any, but arxiv data is mostly strings/lists
        table.put_item(Item=paper_dict)

    def save_paper_pdf(self, paper_id, pdf_bytes):
        """Saves paper PDF to MinIO."""
        # Clean ID to be safe for S3 key? Arxiv IDs like 1234.5678 are fine.
        key = f"{paper_id}.pdf"
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=pdf_bytes,
            ContentType='application/pdf'
        )

    def get_paper_pdf(self, paper_id):
        """Retrieves paper PDF bytes from MinIO."""
        key = f"{paper_id}.pdf"
        response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()

    def get_all_metadata(self):
        """Retrieves all paper metadata from DynamoDB."""
        table = self.dynamodb.Table(self.table_name)
        response = table.scan()
        data = response['Items']
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            data.extend(response['Items'])
        return data

    def reset_db(self):
        """Deletes and recreates the DynamoDB table."""
        try:
            table = self.dynamodb.Table(self.table_name)
            table.delete()
            table.wait_until_not_exists()
            print(f"Table {self.table_name} deleted.")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                print(f"Error deleting table: {e}")
        
        self.init_db()

    def reset_bucket(self):
        """Deletes all objects in the bucket and recreates it."""
        try:
            # List all objects
            response = self.s3.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                objects = [{'Key': obj['Key']} for obj in response['Contents']]
                # Delete in batches
                for i in range(0, len(objects), 1000):
                    self.s3.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects[i:i+1000]}
                    )
                print(f"Bucket {self.bucket_name} cleared.")
        except ClientError as e:
            print(f"Error clearing bucket: {e}")

    def init_qdrant(self):
        """Creates Qdrant collection if it doesn't exist."""
        try:
            self.qdrant.get_collection(self.qdrant_collection)
            print(f"Collection {self.qdrant_collection} already exists.")
        except Exception:
            print(f"Creating collection {self.qdrant_collection}...")
            self.qdrant.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            print(f"Collection {self.qdrant_collection} created.")

    def reset_qdrant(self):
        """Recreates the Qdrant collection."""
        try:
            self.qdrant.delete_collection(self.qdrant_collection)
            print(f"Collection {self.qdrant_collection} deleted.")
        except Exception:
            pass
        self.init_qdrant()

    def save_embeddings(self, vectors, payloads):
        """
        Saves ebeddings and metadata to Qdrant.
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dicts
        """
        batch_size = 100
        total = len(vectors)
        
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_vectors = vectors[i:end]
            batch_payloads = payloads[i:end]
            
            points = [
                models.PointStruct(
                    id=i + idx,
                    vector=v.tolist() if hasattr(v, 'tolist') else v,
                    payload=p
                )
                for idx, (v, p) in enumerate(zip(batch_vectors, batch_payloads))
            ]
             # Wait, `id` should be unique. `i+idx` is unique within this call, but what if we add more later?
             # Let's use UUIDs generated from the payload content to avoid duplicates.
            import uuid
            
            points = []
            for v, p in zip(batch_vectors, batch_payloads):
                # Create a deterministic ID based on strategy, paper_id, and chunk_index (or text hash)
                # Payload should have 'strategy', 'paper_id', 'chunk_text'
                unique_str = f"{p.get('strategy')}_{p.get('paper_id')}_{p.get('chunk_text')[:50]}"
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

    def fetch_embeddings(self, strategy):
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
        
        # Scroll to get all results
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
    # Initialize infrastructure
    sm = StorageManager()
    sm.init_db()
    sm.init_bucket()
    sm.init_qdrant()
