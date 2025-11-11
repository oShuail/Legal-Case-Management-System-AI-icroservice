from fastapi import APIRouter, HTTPException
from app.core.embeddings import EmbeddingService
from app.api.schemas.requests import EmbedRequest
from app.api.schemas.responses import EmbedResponse

router = APIRouter(prefix="/embed", tags=["embeddings"])

@router.post("/", response_model=EmbedResponse)
def generate_embeddings(req: EmbedRequest):
    try:
        svc = EmbeddingService()
        embs = svc.encode_batch(req.texts, normalize=req.normalize)
        return EmbedResponse(
            embeddings=embs.tolist(),
            dimension=svc.dimension,
            count=len(req.texts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
