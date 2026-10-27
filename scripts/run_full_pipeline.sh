#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-results}"

for agent in q_learning sarsa; do
    "$PYTHON" -m traffic_rl.train --agent "$agent" --output "$OUTPUT"
done
"$PYTHON" -m traffic_rl.evaluate --output "$OUTPUT"
"$PYTHON" -m traffic_rl.demo --output "$OUTPUT"
