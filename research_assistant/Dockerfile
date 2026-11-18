FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_INDEX_PATH=/data/index_store \
    RAG_INGEST_ROOT=/data/ingest \
    RAG_STATE_DIR=/data/state \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser \
        --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/index_store /data/ingest /data/state /app/artifacts /app/logs \
        /app/memory /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /data /app/artifacts /app/logs /app/memory \
        /home/appuser

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.14.1+cpu"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

EXPOSE 7860

CMD ["uvicorn", "app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
