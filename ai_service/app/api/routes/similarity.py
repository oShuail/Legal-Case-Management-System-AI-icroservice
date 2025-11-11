from fastapi import APIRouter
from app.core.embeddings import EmbeddingService
from app.api.schemas.requests import SimilarityRequest
from app.api.schemas.responses import SimilarityResponse, SimilarityItem
import numpy as np

router = APIRouter(prefix="/similarity", tags=["similarity"])

@router.post("/", response_model=SimilarityResponse)
def semantic_search(req: SimilarityRequest):
    svc = EmbeddingService()
    q = svc.encode_batch(req.queries)     # shape: [Q, D]
    d = svc.encode_batch(req.corpus)      # shape: [N, D]

    # cosine similarity: (QxD @ DxN) / (||Q|| * ||D||)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    sims = qn @ dn.T  # [Q, N]

    results = []
    for i in range(sims.shape[0]):
        row = sims[i]
        idxs = np.argsort(-row)[: req.top_k]  # top_k highest scores
        results.append([SimilarityItem(doc=req.corpus[j], score=float(row[j])) for j in idxs])

    return SimilarityResponse(results=results)
