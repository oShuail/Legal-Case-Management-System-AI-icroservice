from typing import List, Optional
import numpy as np
from app.core.models import model_manager

class EmbeddingService:
    """provides easy encode APIs used by routes."""
    def __init__(self):
        self.model = model_manager.embedding_model
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        arr = self.model.encode(text, normalize_embeddings=normalize)
        return arr[0] if arr.ndim > 1 else arr

    def encode_batch(self, texts: List[str], normalize: bool = True, batch_size: Optional[int] = None) -> np.ndarray:
        # batch_size unused for fake model, but kept for API compatibility
        return self.model.encode(texts, normalize_embeddings=normalize)
