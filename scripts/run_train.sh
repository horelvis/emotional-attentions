#!/usr/bin/env bash
set -euo pipefail
python -m emotion_attention.train \
  --epochs 5 \
  --batch-size 16 \
  --lambda-prop 0.8 \
  --use-distil \
  --freeze-base "$@"
