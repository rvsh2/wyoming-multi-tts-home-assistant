#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "/app" ]]; then
  echo "/app is required" >&2
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

VENV_PATHS=(
  "/app/.venv"
  "/app/.venv-chatterbox"
  "/app/.venv-qwen"
  "/app/.venv-whisperspeech"
  "/app/.venv-mms"
  "/app/.venv-xtts"
  "/app/.venv-fish"
)

for VENV_PATH in "${VENV_PATHS[@]}"; do
  if [[ -x "${VENV_PATH}/bin/python" ]]; then
    uv pip install --python "${VENV_PATH}/bin/python" --no-deps /app
  fi
done
