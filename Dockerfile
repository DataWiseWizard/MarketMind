# ==============================================================================
# MarketMind - Production & Hugging Face Spaces Dockerfile
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    API_PORT=7860 \
    API_HOST=0.0.0.0 \
    PYTHONPATH=/app \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces expects non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, models, data, reports, and dashboard
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY reports/ ./reports/
COPY dashboard/ ./dashboard/

# Ensure user 1000 has full ownership to write to SQLite databases & artifacts
RUN chown -R user:user /app && \
    chmod -R 775 /app

# Switch to non-root user
USER user

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Health check endpoint
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI backend on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
