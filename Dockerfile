# ==============================================================================
# MarketMind - Production Dockerfile (Render / Cloud / Local)
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    API_PORT=10000 \
    API_HOST=0.0.0.0 \
    PYTHONPATH=/app \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with UID 1000 for standard container security
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies (CPU-optimized PyTorch on Linux)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, data, reports, and dashboard
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY reports/ ./reports/
COPY dashboard/ ./dashboard/

# Ensure non-root user has full ownership to write SQLite databases & artifacts
RUN chown -R user:user /app && \
    chmod -R 775 /app

# Switch to non-root user
USER user

# Render defaults to 10000, local 8000, Hugging Face 7860
EXPOSE 10000 8000 7860

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start FastAPI backend with dynamic port expansion
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"
