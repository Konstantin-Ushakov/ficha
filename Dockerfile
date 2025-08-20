FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps kept minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    sentence-transformers>=2.2.0 \
    scikit-learn>=1.3.0

# Copy analyzer code
COPY . /app

# Create mount points expected by compose (optional but explicit)
RUN mkdir -p /workspace/user-features /workspace/tests /app/model

CMD ["python", "analyzer.py"]

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
# Сначала ставим CPU-версию PyTorch, затем остальные зависимости (torch не будет обновлен)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.0 && \
    pip install --no-cache-dir -r /app/requirements.txt

# 2) App code (changes here не ломают кэш зависимостей)
COPY analyzer.py /app/analyzer.py
COPY core /app/core
COPY metrics /app/metrics

# Prefetch disabled to enforce strict offline/local-only usage; models are mounted at runtime

CMD ["python", "analyzer.py"]
