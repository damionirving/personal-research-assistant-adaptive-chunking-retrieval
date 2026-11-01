from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


class JsonObjectStore(Protocol):
    def read_json(self, key: str, default: Any) -> Any:
        raise NotImplementedError

    def write_json(self, key: str, value: Any) -> None:
        raise NotImplementedError


class FileJsonObjectStore(JsonObjectStore):
    """JSON object store backed by local files."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def read_json(self, key: str, default: Any) -> Any:
        path = self._path_for(key)
        with self._lock:
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def _path_for(self, key: str) -> Path:
        return self._root / key


class S3JsonObjectStore(JsonObjectStore):
    """JSON object store backed by an S3 bucket."""

    def __init__(self, bucket_name: str, prefix: str = "pra-prod") -> None:
        self._bucket_name = bucket_name
        self._prefix = prefix.strip("/")
        self._client = None

    def read_json(self, key: str, default: Any) -> Any:
        try:
            response = self._s3().get_object(
                Bucket=self._bucket_name,
                Key=self._key_for(key),
            )
        except self._client_error_type() as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return default
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_json(self, key: str, value: Any) -> None:
        self._s3().put_object(
            Bucket=self._bucket_name,
            Key=self._key_for(key),
            Body=json.dumps(value, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def _key_for(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _s3(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _client_error_type(self):
        from botocore.exceptions import ClientError

        return ClientError
