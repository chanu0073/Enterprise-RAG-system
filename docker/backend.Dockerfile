# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    EKIP_DATA_DIR=/data/storage

# CPU wheels by default (small image). For NVIDIA GPUs build with:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_FINETUNE=true

RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch --index-url ${TORCH_INDEX_URL}
COPY requirements.txt requirements-finetune.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_FINETUNE" = "true" ]; then pip install -r requirements-finetune.txt; fi

COPY backend/app ./app

RUN useradd --create-home --uid 1000 ekip && mkdir -p /data && chown -R ekip:ekip /data /app
USER ekip
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# single worker: models, indexes and the job queue live in-process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
