#!/usr/bin/env bash
set -euo pipefail

ENGINE_NAME="${1:-}"

if [[ -z "${ENGINE_NAME}" ]]; then
  echo "Usage: $0 <chatterbox|qwen|xtts|whisperspeech|mms|fish>" >&2
  exit 1
fi

case "${ENGINE_NAME}" in
  chatterbox)
    VENV_PATH=".venv-chatterbox"
    EXTRA_NAME="chatterbox"
    ;;
  qwen)
    VENV_PATH=".venv-qwen"
    EXTRA_NAME="qwen"
    ;;
  xtts)
    VENV_PATH=".venv-xtts"
    EXTRA_NAME="xtts"
    ;;
  whisperspeech)
    VENV_PATH=".venv-whisperspeech"
    EXTRA_NAME="whisperspeech"
    ;;
  mms)
    VENV_PATH=".venv-mms"
    EXTRA_NAME="mms"
    ;;
  fish)
    VENV_PATH=".venv-fish"
    EXTRA_NAME="fish"
    ;;
  *)
    echo "Unsupported engine '${ENGINE_NAME}'" >&2
    exit 1
    ;;
esac

mkdir -p /tmp/uv-cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
FLASH_ATTN_WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-/app/wheelhouse}"

if [[ "${ENGINE_NAME}" == "qwen" ]] && compgen -G "${FLASH_ATTN_WHEEL_DIR}/flash_attn-*.whl" > /dev/null; then
  export UV_FIND_LINKS="${FLASH_ATTN_WHEEL_DIR}"
  echo "Using local flash-attn wheel from ${FLASH_ATTN_WHEEL_DIR}" >&2
fi

uv venv --clear --python 3.11 "${VENV_PATH}"

if [[ "${ENGINE_NAME}" == "fish" ]]; then
  FISH_REPO_DIR="${FISH_SPEECH_REPO_DIR:-/opt/fish-speech}"
  uv pip install --python "${VENV_PATH}/bin/python" "${FISH_REPO_DIR}[cu126]"
else
  case "${ENGINE_NAME}" in
    chatterbox)
      uv pip install --python "${VENV_PATH}/bin/python" \
        "chatterbox-tts>=0.1.2" \
        "torch>=2.6,<2.7" \
        "torchaudio>=2.6,<2.7" \
        "transformers==4.46.3"
      ;;
    qwen)
      uv pip install --python "${VENV_PATH}/bin/python" \
        "flash-attn>=2.8.0,<3" \
        "qwen-tts>=0.0.2" \
        "torch>=2.6,<2.7" \
        "torchaudio>=2.6,<2.7" \
        "transformers==4.57.3"
      ;;
    xtts)
      uv pip install --python "${VENV_PATH}/bin/python" \
        "TTS>=0.22.0,<0.23" \
        "torch>=2.6,<2.7" \
        "torchaudio>=2.6,<2.7" \
        "transformers==4.43.1"
      ;;
    whisperspeech)
      uv pip install --python "${VENV_PATH}/bin/python" \
        "ipython>=9.0,<10" \
        "speechbrain<1.0" \
        "torch>=2.6,<2.7" \
        "torchaudio>=2.6,<2.7" \
        "vocos>=0.1.0,<1" \
        "WhisperSpeech @ git+https://github.com/WhisperSpeech/WhisperSpeech.git"
      ;;
    mms)
      uv pip install --python "${VENV_PATH}/bin/python" \
        "soundfile>=0.12,<1" \
        "torch>=2.6,<2.7" \
        "torchaudio>=2.6,<2.7" \
        "transformers>=4.57.0,<4.58" \
        "sentencepiece>=0.2,<1"
      ;;
  esac
fi

echo "Created ${VENV_PATH} for extra '${EXTRA_NAME}'"
