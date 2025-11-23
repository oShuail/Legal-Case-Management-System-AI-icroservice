# 1) Use an official Python base image
FROM python:3.12-slim

# 2) Set the working directory inside the container
WORKDIR /app

# 3) Install system tools (curl is used by the HEALTHCHECK later)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 4) Install Python dependencies
# Copy only requirements.txt first to leverage Docker cache:
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5) Copy your application source code into the image
COPY ai_service ./ai_service

# 6) Make "ai_service" available on the Python path
ENV PYTHONPATH=/app/ai_service
ENV PYTHONUNBUFFERED=1

# 7) Expose the port the app will listen on inside the container
EXPOSE 8000

# 8) Add a simple container-level healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health/ || exit 1

# 9) Run the FastAPI app with Uvicorn when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "ai_service"]
