#!/usr/bin/env bash
set -euo pipefail
python -m emotion_attention.infer --tokenizer-dir artifacts/tokenizer "$@"
