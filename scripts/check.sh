#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

"$PYTHON" -m ruff check .
"$PYTHON" -m ruff format --check .
"$PYTHON" -m mypy
"$PYTHON" -m pytest -q
SMOKE="$(mktemp -d)"
trap 'rm -rf "$SMOKE"' EXIT
"$PYTHON" -m traffic_rl.train --agent q_learning --episodes 5 --output "$SMOKE"
