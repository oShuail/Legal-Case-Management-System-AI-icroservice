from typing import Optional, List, Union
import numpy as np
from app.utils.logger import logger

class _FakeEmbedder:
    """
    Brief: deterministic, fast embedder for dev/tests.
    Turns text into a fixed-size vector using a seeded RNG (based on text hash).
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.device = "cpu"

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            # stable seed from hash -> deterministic across runs
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype("float32")
            if normalize_embeddings:
                n = np.linalg.norm(v) + 1e-9
                v = v / n
            vecs.append(v)
        return np.stack(vecs)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

class ModelManager:
    """
    Brief: singleton-style loader for embedding model.
    In prod, you can swap _FakeEmbedder with a real model (e.g., sentence-transformers).
    """
    _instance: Optional["ModelManager"] = None
    _embedding_model: Optional[object] = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info("Loading FAKE embedder (dev mode).")
            self._embedding_model = _FakeEmbedder(dim=384)
        return self._embedding_model

model_manager = ModelManager()
