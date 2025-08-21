FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/model \
    SENTENCE_TRANSFORMERS_HOME=/app/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Cacheable deps layer: only invalidates when requirements.txt changes
COPY requirements.txt /app/requirements.txt
# Ставим зависимости из requirements.txt и CPU-версию PyTorch, согласованную с runtime
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.0

# 2) App code (changes here не ломают кэш зависимостей)
COPY analyzer.py /app/analyzer.py
COPY core /app/core
COPY metrics /app/metrics
COPY config.json /app/config.json

# Prefetch disabled to enforce strict offline/local-only usage; models are mounted at runtime

CMD ["python", "analyzer.py"]
