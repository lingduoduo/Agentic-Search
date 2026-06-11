# --- Stage 1: build the React frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Stage 2: Python application ---
FROM python:3.11-slim AS app

WORKDIR /app

# System deps needed by faiss-cpu and pyserini (Java not included by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir -r requirements.txt

COPY . .
# Bring in the pre-built frontend bundle
COPY --from=frontend-builder /app/web/dist ./web/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["uvicorn", "src.internal.servers.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
