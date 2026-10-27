#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-results}"

for agent in q_learning sarsa; do
    "$PYTHON" -m traffic_rl.train --agent "$agent" --output "$OUTPUT"
done
if "$PYTHON" -c "import stable_baselines3" 2>/dev/null; then
    "$PYTHON" -m traffic_rl.train --agent dqn --output "$OUTPUT"
else
    echo "skipping DQN: install the dqn extra with pip install -e \".[dqn]\" to train it"
fi
"$PYTHON" -m traffic_rl.evaluate --output "$OUTPUT"
"$PYTHON" -m traffic_rl.demo --output "$OUTPUT"
