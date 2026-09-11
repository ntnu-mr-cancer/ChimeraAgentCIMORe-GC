#!/usr/bin/env bash
set -euo pipefail

NUM_WORKERS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

trap 'kill -- -$$ 2>/dev/null || true' INT TERM EXIT

i=0

IFS=','
for gpu in $CUDA_DEVICES; do
    echo "Starting worker $i/$NUM_WORKERS on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu \
    python3 -m chimera_agent_baseline.run \
        agent.worker_id=$i \
        agent.num_workers=$NUM_WORKERS \
        "$@" &

    i=$((i + 1))
done

wait