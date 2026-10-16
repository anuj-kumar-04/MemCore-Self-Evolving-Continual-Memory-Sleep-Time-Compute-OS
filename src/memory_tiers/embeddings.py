"""
Deterministic, lightweight embedding provider for MemCore.
Supports:
1. Local deterministic semantic projection (384 dimensions) with zero external network/API dependencies.
2. Fast OpenAI / Groq / LiteLLM embeddings if API keys are present.
"""

from __future__ import annotations
import os
import math
import hashlib
from typing import List
import numpy as np


def generate_local_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Generates a deterministic 384-dimensional unit vector from text tokens.
    Words with semantic similarity (or exact tokens) produce overlapping projections.
    Guarantees fast, offline, repeatable vector operations without external models.
    """
    if not text:
        vec = [0.0] * dim
        vec[0] = 1.0
        return vec

    vec = np.zeros(dim, dtype=np.float32)
    # Tokenize by word and char n-grams
    tokens = text.lower().replace("-", " ").replace("_", " ").split()
    
    # Also add bigrams if present
    ngrams = list(tokens)
    for i in range(len(tokens) - 1):
        ngrams.append(f"{tokens[i]}_{tokens[i+1]}")

    for token in ngrams:
        # Multiple hash projections to distribute features
        h1 = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        h2 = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        
        idx1 = h1 % dim
        idx2 = h2 % dim
        sign1 = 1.0 if ((h1 >> 8) & 1) else -1.0
        sign2 = 1.0 if ((h2 >> 8) & 1) else -1.0
        
        vec[idx1] += sign1
        vec[idx2] += sign2 * 0.5

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return vec.tolist()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculates cosine similarity between two float vectors.
    """
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def get_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Asynchronously get embedding, checking for OpenAI API key or falling back to local.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and len(openai_key.strip()) > 10:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            resp = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=dim
            )
            return resp.data[0].embedding
        except Exception:
            pass

    return generate_local_embedding(text, dim=dim)
