#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_INSIDE_DOCKER:-0}" != "1" ]]; then
  exec docker compose run --rm -e RUN_INSIDE_DOCKER=1 emo-infer ./scripts/run_infer.sh "$@"
fi

python -m emotion_attention.infer --tokenizer-dir artifacts/tokenizer "$@"
