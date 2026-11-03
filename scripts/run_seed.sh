#!/usr/bin/env bash
# Retrain the final configuration with another seed and evaluate it on test. Resumable.
# Usage: bash scripts/run_seed.sh 1
set -euo pipefail
SEED=${1:?seed}
export PYTHONPATH=src PATH=~/venvs/vllm/bin:$PATH
OUT=outputs/qwen3-4b-lora-full-seed$SEED
TAG=qwen3-4b-lora-full-ep2-seed$SEED

[ -d "$OUT/merged" ] || ~/venvs/train/bin/python scripts/train.py --set seed="$SEED" output_dir="$OUT"
[ -f "reports/metrics/${TAG}_test.json" ] ||
  ~/venvs/vllm/bin/python scripts/evaluate.py --tag "$TAG" --model "$OUT/merged" --split test
echo "done: $TAG"
