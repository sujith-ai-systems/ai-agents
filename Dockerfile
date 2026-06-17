# syntax=docker/dockerfile:1
# Container image for the Options Market Analyzer A2A server (Google Cloud Run).

FROM python:3.12-slim

# Faster, cleaner Python in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY multi-agents ./multi-agents

# Cloud Run sets $PORT (default 8080); the server reads it automatically.
EXPOSE 8080

# Start the A2A server
CMD ["python", "multi-agents/market/a2a_server.py"]
