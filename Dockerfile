FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts

ENV PYTHONPATH=/app/src
RUN chmod +x scripts/*.sh

ENTRYPOINT ["bash"]
