#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=${LOG_FILE:-artifacts/train.log}
mkdir -p "$(dirname "${LOG_FILE}")"

python -m emotion_attention.train \
  --epochs 5 \
  --batch-size 16 \
  --lambda-prop 0.8 \
  --use-distil \
  --log-file "${LOG_FILE}" \
  --freeze-base "$@"
