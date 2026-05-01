"""
MinIO access for frozen evaluation corpora (PDFs + manifest).
"""
from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.rag_constants import EVAL_FROZEN_CORPUS_BUCKET


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    )


def ensure_eval_bucket(bucket: str = EVAL_FROZEN_CORPUS_BUCKET) -> None:
    s3 = _s3_client()
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)


def put_eval_pdf(snapshot_id: str, paper_id: str, pdf_bytes: bytes, bucket: str = EVAL_FROZEN_CORPUS_BUCKET) -> str:
    ensure_eval_bucket(bucket)
    s3 = _s3_client()
    key = f"{snapshot_id}/{paper_id}.pdf"
    s3.put_object(Bucket=bucket, Key=key, Body=pdf_bytes, ContentType="application/pdf")
    return key


def clear_snapshot(snapshot_id: str, bucket: str = EVAL_FROZEN_CORPUS_BUCKET) -> int:
    """Delete all objects under one snapshot prefix. Returns deleted count."""
    ensure_eval_bucket(bucket)
    s3 = _s3_client()
    prefix = f"{snapshot_id}/"
    continuation_token: str | None = None
    deleted = 0

    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        contents = response.get("Contents", [])
        if contents:
            objects = [{"Key": obj["Key"]} for obj in contents]
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            deleted += len(objects)
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")

    return deleted


def get_eval_pdf(snapshot_id: str, paper_id: str, bucket: str = EVAL_FROZEN_CORPUS_BUCKET) -> bytes:
    s3 = _s3_client()
    key = f"{snapshot_id}/{paper_id}.pdf"
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def put_manifest(snapshot_id: str, data: dict[str, Any], bucket: str = EVAL_FROZEN_CORPUS_BUCKET) -> None:
    ensure_eval_bucket(bucket)
    s3 = _s3_client()
    key = f"{snapshot_id}/manifest.json"
    body = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def get_manifest(snapshot_id: str, bucket: str = EVAL_FROZEN_CORPUS_BUCKET) -> dict[str, Any]:
    s3 = _s3_client()
    key = f"{snapshot_id}/manifest.json"
    response = s3.get_object(Bucket=bucket, Key=key)
    raw = response["Body"].read().decode("utf-8")
    return json.loads(raw)
