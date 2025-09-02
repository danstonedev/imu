# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better cache
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default envs; can be overridden
ENV WORKERS=2 \
    LOG_LEVEL=info \
    OPENAPI_ENABLED=false \
    DOCS_ENABLED=false \
    ALLOWED_ORIGINS=*

CMD ["python", "scripts/serve.py"]
