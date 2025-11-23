"""
Core similarity logic for the AI microservice.

This module is intentionally independent of FastAPI. It operates purely on
Python types (lists of strings, floats, etc.) so it can be reused from
routes, background workers, or other services.
"""

from typing import List, Tuple

import numpy as np

from app.core.embeddings import EmbeddingService


class SimilarityService:
    """
    Service responsible for computing semantic similarity between pieces of text.

    It delegates:
    - text -> vector: to EmbeddingService
    - vector math: handled here via cosine similarity

    This design makes it easy to swap the underlying embedding model later
    without touching any of the similarity or API code.
    """

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        # Allow injecting a custom EmbeddingService for testing,
        # otherwise use the default one.
        self._embedding_service = embedding_service or EmbeddingService()

    @staticmethod
    def _cosine_similarity_matrix(
        query_vectors: np.ndarray, corpus_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between each query vector and each corpus vector.

        Shapes:
            query_vectors: [num_queries, dim]
            corpus_vectors: [num_docs, dim]

        Returns:
            sim_matrix: [num_queries, num_docs]
                sim_matrix[i, j] = cosine(queries[i], corpus[j])
        """
        if query_vectors.size == 0 or corpus_vectors.size == 0:
            # No queries or no corpus -> all zeros
            return np.zeros((query_vectors.shape[0], corpus_vectors.shape[0]))

        # Normalize rows to unit length to compute cosine similarity via dot product
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8
        corpus_norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-8

        query_unit = query_vectors / query_norms
        corpus_unit = corpus_vectors / corpus_norms

        # Cosine similarity = dot product of normalized vectors
        sim_matrix = np.dot(query_unit, corpus_unit.T)
        return sim_matrix

    def rank(
        self, queries: List[str], corpus: List[str], top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        Compute similarity for each query against all corpus documents.

        Args:
            queries: List of query strings.
            corpus: List of candidate document strings.
            top_k: How many top documents to return per query.

        Returns:
            A list of lists:
                outer list length = len(queries)
                inner list contains (doc, score) tuples sorted by score descending.
        """
        # Edge cases: no queries or no corpus
        if not queries:
            return []
        if not corpus:
            return [[] for _ in queries]

        # 1) Encode queries and corpus into vectors
        query_vecs = np.array(self._embedding_service.encode_batch(queries))
        corpus_vecs = np.array(self._embedding_service.encode_batch(corpus))

        # 2) Compute cosine similarity matrix
        sim_matrix = self._cosine_similarity_matrix(query_vecs, corpus_vecs)

        # Ensure we don't ask for more docs than exist
        top_k = max(1, min(top_k, len(corpus)))

        results: List[List[Tuple[str, float]]] = []
        num_queries = len(queries)

        for i in range(num_queries):
            scores = sim_matrix[i]  # shape: [num_docs]

            # Argsort in descending order of similarity
            # np.argsort is ascending by default, so we reverse it.
            sorted_indices = np.argsort(scores)[::-1][:top_k]

            query_results: List[Tuple[str, float]] = []
            for idx in sorted_indices:
                doc_text = corpus[idx]
                score = float(scores[idx])
                query_results.append((doc_text, score))

            results.append(query_results)

        return results
