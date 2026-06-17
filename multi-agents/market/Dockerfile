# syntax=docker/dockerfile:1
# Container image for the Options Market Analyzer A2A server (Google Cloud Run).
#
# Build context is the REPO ROOT so shared files (requirements.txt) are available:
#   docker build -f multi-agents/market/Dockerfile -t IMAGE .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching).
# Uses this agent's requirements.txt if present, else the shared root one.
COPY requirements.txt /tmp/root-requirements.txt
COPY multi-agents/market/requirements.tx[t] /tmp/agent-requirements.txt
RUN if [ -f /tmp/agent-requirements.txt ]; then \
        pip install --no-cache-dir -r /tmp/agent-requirements.txt; \
    else \
        pip install --no-cache-dir -r /tmp/root-requirements.txt; \
    fi

# Copy only this agent's code
COPY multi-agents/market ./multi-agents/market

# Cloud Run injects $PORT (default 8080); the server reads it automatically.
EXPOSE 8080

CMD ["python", "multi-agents/market/a2a_server.py"]
