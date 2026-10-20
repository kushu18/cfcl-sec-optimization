# =============================================================================
# CFCL SEC Optimization Engine — Multi-Stage Industrial Dockerfile
# =============================================================================

FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY config/ /app/config/
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/
COPY outputs/models/ /app/outputs/models/
COPY outputs/figures/ /app/outputs/figures/
COPY outputs/reports/ /app/outputs/reports/
COPY dashboard/ /app/dashboard/
COPY setup.py /app/setup.py

RUN pip install --no-cache-dir -e .

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Health check against FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: Launch FastAPI supervisory microservice
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
