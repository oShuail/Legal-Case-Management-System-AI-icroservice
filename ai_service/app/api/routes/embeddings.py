from typing import List

from fastapi import APIRouter

from app.api.schemas.requests import EmbedRequest
from app.api.schemas.responses import EmbedResponse
from app.core.embeddings import EmbeddingService

# Router for embedding-related endpoints
router = APIRouter(prefix="/embed", tags=["embeddings"])

# Create a single EmbeddingService instance for this module
_embedding_service = EmbeddingService()


@router.post("/", response_model=EmbedResponse)
def generate_embeddings(payload: EmbedRequest) -> EmbedResponse:
    """
    Generate embeddings for a batch of texts.

    This endpoint is intentionally thin:
    - It receives a validated EmbedRequest (texts + normalize flag).
    - It delegates to EmbeddingService.encode_batch.
    - It wraps the result into the EmbedResponse Pydantic model.
    """
    vectors: List[List[float]] = _embedding_service.encode_batch(
        texts=payload.texts,
        normalize=payload.normalize,
    )

    # Dimension is the length of a single vector (0 if empty)
    dimension = len(vectors[0]) if vectors else 0

    return EmbedResponse(
        embeddings=vectors,
        dimension=dimension,
        count=len(vectors),
    )
