#!/usr/bin/env bash
set -euo pipefail

uvicorn emotion_attention.api:app --host 0.0.0.0 --port "${API_PORT:-8000}" --reload
