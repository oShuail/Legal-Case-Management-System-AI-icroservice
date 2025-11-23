from typing import List

from fastapi import APIRouter

from app.api.schemas.requests import SimilarityRequest
from app.api.schemas.responses import SimilarityItem, SimilarityResponse
from app.core.similarity import SimilarityService

router = APIRouter(prefix="/similarity", tags=["similarity"])

# Create a single service instance for this module.
_similarity_service = SimilarityService()


@router.post("/", response_model=SimilarityResponse)
def compute_similarity(payload: SimilarityRequest) -> SimilarityResponse:
    """
    Compute semantic similarity between queries and corpus documents.

    This endpoint is a thin wrapper around the core SimilarityService:
    - It receives a validated SimilarityRequest (queries, corpus, top_k).
    - Delegates the heavy lifting to SimilarityService.rank.
    - Maps the raw (doc, score) tuples into Pydantic response models.
    """
    raw_results: List[List[tuple[str, float]]] = _similarity_service.rank(
        queries=payload.queries,
        corpus=payload.corpus,
        top_k=payload.top_k,
    )

    api_results: List[List[SimilarityItem]] = []
    for query_result in raw_results:
        items = [
            SimilarityItem(doc=doc_text, score=score) for doc_text, score in query_result
        ]
        api_results.append(items)

    return SimilarityResponse(results=api_results)
