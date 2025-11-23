# Legal Case Management System – AI Microservice

[![CI](https://github.com/oShuail/Legal-Case-Management-System-AI-Microservice/actions/workflows/ci.yml/badge.svg)](https://github.com/oShuail/Legal-Case-Management-System-AI-Microservice/actions/workflows/ci.yml)

This repository contains the **AI microservice** used by the _Legal Case Management System_ project.

The service exposes a small, clean FastAPI API for:

- **Health checks** – to monitor container / service status  
- **Text embeddings** – generate vector representations for text  
- **Semantic similarity** – compute similarity between queries and a corpus  

The code is structured so that the **AI core (embeddings, similarity)** is isolated from
the API layer and can be swapped later for a real model (e.g., BGE, OpenAI, etc.).

---

## Tech stack

- **Language:** Python 3.12  
- **Web framework:** FastAPI + Uvicorn  
- **Config:** Pydantic v2 (`BaseSettings`) + `.env`  
- **Core logic:** NumPy, custom embedding backend  
- **Logging:** loguru  
- **Testing:** pytest  
- **Containerization:** Docker, docker-compose  
- **CI:** GitHub Actions (runs tests on each push / PR)

---

## Project structure

```text
ai_service/
  app/
    main.py                # FastAPI application entrypoint
    config.py              # Settings (reads from .env)
    api/
      __init__.py
      deps.py              # (reserved for shared dependencies)
      routes/
        __init__.py
        health.py          # /health/
        embeddings.py      # /embed/
        similarity.py      # /similarity/
      schemas/
        __init__.py
        requests.py        # Pydantic request models
        responses.py       # Pydantic response models
    core/
      __init__.py
      embeddings.py        # Embedding backend + service
      similarity.py        # SimilarityService (cosine similarity)
    utils/
      __init__.py
      logger.py            # loguru configuration
    tests/
      __init__.py
      test_api.py          # API-level tests
      test_similarity_core.py  # Core similarity tests

plans/
  ai-microservice-implementation-plan.md  # Planning document (not used by code)

Dockerfile
docker-compose.yml
requirements.txt
pytest.ini
.env.example
.gitignore
