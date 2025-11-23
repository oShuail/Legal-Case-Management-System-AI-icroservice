from app.core.similarity import SimilarityService


def test_similarity_service_basic_ranking():
    service = SimilarityService()

    queries = ["The contract was signed yesterday"]
    corpus = [
        "The contract was signed yesterday by both parties.",
        "The weather is sunny today.",
        "Yesterday, both parties executed the agreement.",
    ]

    top_k = 3
    results = service.rank(queries=queries, corpus=corpus, top_k=top_k)

    # We have 1 query, so results should be a list with a single inner list
    assert len(results) == 1

    ranked = results[0]
    # Inner list should have exactly top_k items (or len(corpus) if smaller)
    assert len(ranked) == min(top_k, len(corpus))

    # Each item is (doc, score)
    for doc, score in ranked:
        assert isinstance(doc, str)
        assert isinstance(score, float)

    # Scores should be sorted in descending order
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
