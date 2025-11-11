from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "service" in body
    assert "version" in body

def test_embed_basic():
    payload = {
        "texts": ["قضية عمالية", "commercial contract"],
        "normalize": True
    }
    r = client.post("/embed/", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert isinstance(body["embeddings"], list)
    assert len(body["embeddings"]) == 2
    dim = body["dimension"]
    assert isinstance(dim, int) and dim > 0
    assert all(len(vec) == dim for vec in body["embeddings"])

def test_similarity_basic():
    payload = {
        "queries": ["termination benefits"],
        "corpus": ["labor case", "sales contract", "end of service benefits", "visa rules"],
        "top_k": 3
    }
    r = client.post("/similarity/", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) == 1
    assert len(body["results"][0]) == 3
    first = body["results"][0][0]
    assert "doc" in first and "score" in first
    assert isinstance(first["doc"], str)
    assert isinstance(first["score"], float)
