"""
Embedding backend and service layer for the AI microservice.

Right now we expose a single "fake" backend that generates deterministic
numeric vectors from input text. This keeps tests fast and avoids large
model downloads during development and for GP deliverables.

Later, you can add a real backend (e.g., SentenceTransformer / BGE) and
select it via settings.embeddings_provider.
"""

from __future__ import annotations

from typing import List
import hashlib

import numpy as np

from app.config import settings
from app.utils.logger import logger


class FakeEmbeddingBackend:
    """
    A simple, deterministic embedding backend.

    It maps each input string to a fixed-size numeric vector using SHA-256.
    This is NOT semantically meaningful, but:
      - it is stable across runs,
      - it does not require downloading any models,
      - it behaves like a real model API (text -> [dim] vector).
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector_for_text(self, text: str) -> np.ndarray:
        """
        Generate a deterministic vector for a single text.

        We:
        - compute SHA-256 over the UTF-8 text,
        - interpret the digest bytes as uint8,
        - tile/truncate to the target dimension,
        - return as float32 vector.
        """
        # 32-byte SHA-256 digest
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)

        # Repeat as needed to reach `dim`
        if raw.size < self.dim:
            reps = int(np.ceil(self.dim / raw.size))
            raw = np.tile(raw, reps)

        vec = raw[: self.dim]
        return vec

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a batch of texts into a [batch_size, dim] array.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = np.stack([self._vector_for_text(t) for t in texts], axis=0)

        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            vectors = vectors / norms

        return vectors


class EmbeddingService:
    """
    Thin service wrapper around the chosen embedding backend.

    For now we only support the 'fake' backend. The provider is controlled
    by settings.embeddings_provider, but values other than 'fake' will
    log a warning and still fall back to FakeEmbeddingBackend.

    Later, you can extend `_build_backend` to support:
      - "bge" / "sentence-transformers" using a real model
      - other providers if needed.
    """

    def __init__(self, backend: FakeEmbeddingBackend | None = None) -> None:
        self._backend = backend or self._build_backend()

    def _build_backend(self) -> FakeEmbeddingBackend:
        provider = settings.embeddings_provider.lower()

        if provider != "fake":
            # Defensive: fail soft (warning + fallback) so tests still run
            logger.warning(
                f"Unsupported embeddings_provider '{settings.embeddings_provider}', "
                "falling back to 'fake' backend."
            )

        # You can later parameterize dim via settings if you want.
        return FakeEmbeddingBackend(dim=64)

    @property
    def dimension(self) -> int:
        return int(self._backend.dim)

    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Public API used by routes and similarity service.

        Returns:
            List of vectors (each vector is a List[float]) so that it can be
            serialized easily into JSON by FastAPI.
        """
        if not texts:
            return []

        vectors = self._backend.encode(texts, normalize=normalize)
        return vectors.tolist()
