FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-dev \
    python3-venv \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    libsox-dev \
    sox \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY scripts ./scripts
COPY wheelhouse ./wheelhouse

RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /opt/fish-speech
RUN python3 /app/scripts/patch_fish_speech.py

RUN --mount=type=cache,target=/tmp/uv-cache \
    chmod +x /app/scripts/setup_engine_venv.sh && \
    mkdir -p /tmp/uv-cache && \
    UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.11 /app/.venv && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 \
    UV_CACHE_DIR=/tmp/uv-cache uv pip install --python /app/.venv/bin/python \
    "fastapi>=0.115.0,<1" \
    "huggingface_hub>=0.31.0,<1" \
    "jinja2>=3.1.0,<4" \
    "numpy>=1.26,<3" \
    "python-multipart>=0.0.9,<1" \
    "soundfile>=0.12,<1" \
    "uvicorn[standard]>=0.30.0,<1" \
    "wyoming>=1.8.0,<2" \
    "httpx>=0.27.0,<1" \
    "pytest>=8.0,<9" && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 /app/scripts/setup_engine_venv.sh chatterbox && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 /app/scripts/setup_engine_venv.sh qwen && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 /app/scripts/setup_engine_venv.sh whisperspeech && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 /app/scripts/setup_engine_venv.sh mms && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 /app/scripts/setup_engine_venv.sh xtts && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 /app/scripts/setup_engine_venv.sh fish

COPY app ./app

RUN --mount=type=cache,target=/tmp/uv-cache \
    chmod +x /app/scripts/install_local_package.sh && \
    UV_CACHE_DIR=/tmp/uv-cache /app/scripts/install_local_package.sh

ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"
ENV HF_HOME="/data/hf-cache"
ENV WYOMING_STATE_DIR="/data/state"
ENV CHATTERBOX_PYTHON="/app/.venv-chatterbox/bin/python"
ENV QWEN_PYTHON="/app/.venv-qwen/bin/python"
ENV WHISPERSPEECH_PYTHON="/app/.venv-whisperspeech/bin/python"
ENV MMS_PYTHON="/app/.venv-mms/bin/python"
ENV XTTS_PYTHON="/app/.venv-xtts/bin/python"
ENV FISH_PYTHON="/app/.venv-fish/bin/python"

EXPOSE 10210 8280

ENTRYPOINT ["python3", "-m", "app"]
