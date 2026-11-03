#!/usr/bin/env bash
# Training phase + test phase. Every choice reads validation metrics only. Re-run to resume.
set -euo pipefail
T=~/venvs/train/bin/python
V=~/venvs/vllm/bin/python
export PYTHONPATH=src PATH=~/venvs/vllm/bin:$PATH
O=outputs
M=reports/metrics

log() { echo "[$(date +%T)] $*"; }
train() {  # train <run name> <k=v ...>
  [ -d "$O/$1/merged" ] || { log "train $1"; $T scripts/train.py --set output_dir="$O/$1" "${@:2}"; }
}
evaluate() {  # evaluate <tag> <model dir> <split> [extra args]
  [ -f "$M/$1_$3.json" ] || { log "eval $1 on $3"; $V scripts/evaluate.py --tag "$1" --model "$2" --split "$3" "${@:4}"; }
}
acc() { python3 -c "import json; print(json.load(open('$M/$1_validation.json'))['exec_acc'])"; }
better() { python3 -c "import sys; sys.exit(0 if $1 >= $2 else 1)"; }

# 1. search: 1,500 examples, 1 epoch, lr 1e-4 vs 2e-4
for LR in 1e-4 2e-4; do
  train "qwen3-4b-lora-n1500-lr$LR" train_limit=1500 epochs=1 learning_rate="${LR/e/.0e}"
  evaluate "qwen3-4b-lora-n1500-lr$LR" "$O/qwen3-4b-lora-n1500-lr$LR/merged" validation
done
if better "$(acc qwen3-4b-lora-n1500-lr1e-4)" "$(acc qwen3-4b-lora-n1500-lr2e-4)"; then LR=1e-4; else LR=2e-4; fi
log "best lr on validation: $LR"

# 2. data-efficiency point: 500 examples, same settings
train qwen3-4b-lora-n500 train_limit=500 epochs=1 learning_rate="${LR/e/.0e}"
evaluate qwen3-4b-lora-n500 "$O/qwen3-4b-lora-n500/merged" validation

# 3. final: all data, 2 epochs, checkpoint per epoch
F="$O/qwen3-4b-lora-full"
train qwen3-4b-lora-full learning_rate="${LR/e/.0e}"
EP1=$(ls -d "$F"/checkpoints/checkpoint-* | sort -V | head -1)
[ -d "$F/merged-ep1" ] || { log "merge $EP1"; $T scripts/merge.py "$EP1" "$F/merged-ep1"; }
evaluate qwen3-4b-lora-full-ep1 "$F/merged-ep1" validation
evaluate qwen3-4b-lora-full-ep2 "$F/merged" validation
if better "$(acc qwen3-4b-lora-full-ep1)" "$(acc qwen3-4b-lora-full-ep2)"; then
  BEST=ep1 BEST_MODEL="$F/merged-ep1" BEST_ADAPTER="$EP1"
else
  BEST=ep2 BEST_MODEL="$F/merged" BEST_ADAPTER="$F/adapter"
fi
log "best epoch on validation: $BEST"

# 4. data-efficiency curve with the final settings (lr and epochs chosen above)
EPOCHS=${BEST#ep}
if [ "$EPOCHS" = 1 ]; then
  N500=qwen3-4b-lora-n500 N1500="qwen3-4b-lora-n1500-lr$LR"
else
  N500=qwen3-4b-lora-n500-ep$EPOCHS N1500=qwen3-4b-lora-n1500-ep$EPOCHS
  for N in 500 1500; do
    TAG=qwen3-4b-lora-n$N-ep$EPOCHS
    train "$TAG" train_limit=$N epochs="$EPOCHS" learning_rate="${LR/e/.0e}"
    evaluate "$TAG" "$O/$TAG/merged" validation
  done
fi
printf '{"500": "%s", "1500": "%s", "6251": "%s"}\n' "$N500" "$N1500" \
  "qwen3-4b-lora-full-$BEST" > reports/curve.json

# 5. test, once
for TAG in qwen3-4b-zeroshot qwen3-4b-fewshot qwen3-8b-fewshot qwen3-14b-awq-fewshot; do
  [ -f "$M/${TAG}_test.json" ] || $V scripts/run_baselines.py --split test --only "$TAG"
done
evaluate "qwen3-4b-lora-full-$BEST" "$BEST_MODEL" test
for TAG in "$N500" "$N1500"; do evaluate "$TAG" "$O/$TAG/merged" test; done
$V scripts/rescore.py
$V scripts/make_tables.py  # error labels: scripts/error_sample.py, then label by hand

# 6. backup to the Windows drive
mkdir -p backup
cp -r "$BEST_ADAPTER" backup/adapter-"$BEST"
cp -r reports backup/
log "done; best: qwen3-4b-lora-full-$BEST"
