"""Embed queries and measure similarity between them."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "text-embedding-3-small"
CACHE_PATH = Path("data/embedding_cache.json")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set (put it in .env)")
        _client = OpenAI()
    return _client


def _key(text: str) -> str:
    return hashlib.sha256(f"{MODEL}:{text}".encode()).hexdigest()[:16]


def _load_disk_cache() -> dict[str, list[float]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_disk_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def embed(texts: list[str], batch_size: int = 128) -> dict[str, np.ndarray]:
    """Return {text: vector}. Only uncached texts hit the API."""
    disk_cache = _load_disk_cache()

    missing = [t for t in texts if _key(t) not in disk_cache]
    if missing:
        client = _get_client()
        print(f"Embedding {len(missing)} new queries "
              f"({len(texts) - len(missing)} cached)")
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            response = client.embeddings.create(model=MODEL, input=batch)
            for text, item in zip(batch, response.data):
                disk_cache[_key(text)] = item.embedding
        _save_disk_cache(disk_cache)
    else:
        print(f"All {len(texts)} queries already embedded")

    return {t: np.array(disk_cache[_key(t)], dtype=np.float32) for t in texts}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))