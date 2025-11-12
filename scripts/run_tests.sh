#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_INSIDE_DOCKER:-0}" != "1" ]]; then
  ARGS=""
  if [[ $# -gt 0 ]]; then
    ARGS=$(printf ' %q' "$@")
  fi
  exec docker compose run --rm -e RUN_INSIDE_DOCKER=1 emo-trainer ./scripts/run_tests.sh${ARGS}
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src"
else
  export PYTHONPATH="${REPO_ROOT}/src"
fi

python3 -m unittest discover -s "${REPO_ROOT}/tests" -p "test_*.py" "$@"
