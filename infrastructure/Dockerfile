FROM python:3.12-slim

WORKDIR /app

# System dependencies for MNE and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first (for Docker cache)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev, no root package)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy application code
COPY shared/ ./shared/
COPY classifiers/ ./classifiers/
COPY experiments/ ./experiments/

# Expose BentoML default port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')" || exit 1

# Run BentoML service
CMD ["bentoml", "serve", "shared.serving.service:NeuroStateService", "--host", "0.0.0.0", "--port", "3000"]
