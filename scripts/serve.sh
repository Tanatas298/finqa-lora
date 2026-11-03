#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-outputs/qwen3-4b-lora-full/merged}"
exec vllm serve "$MODEL" \
  --served-model-name finqa-4b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
