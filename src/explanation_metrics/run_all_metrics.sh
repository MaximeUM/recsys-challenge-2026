#!/bin/bash
# Complete explanation-evaluation pipeline (reproducible: seed 2026, truncation=False).
#   1. dev sample (build_dev_responses.py, seed 2026) -> 2. final-pipeline generation on four GPUs
#   3. merge -> 4. literature metrics (reference_free) -> 5. robustness
#   6. dimensions IntRS'25 (userstudy_dimensions.py) x 4 judges -> 7. combine.
# Prerequisite: HF token (~/.cache/huggingface/token) with Gemma and Llama access.
#   bash src/explanation_metrics/run_all_metrics.sh   (logs: logs/run_all_metrics.log)
set -e
cd "$(dirname "$0")/../.." || exit 1   # repository root
mkdir -p logs exp
PY=python    # analyses (reference_free/robustness)
PYV=python   # Model inference (vLLM) in the same environment.
DEV_PARQ=data/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet

echo "[0] remove previous results"
rm -f exp/inference/devset/shqwen8b_respsample_shard*.json \
      exp/inference/devset/shqwen8b_respsample_shard*_convbestofN.json* \
      exp/inference/devset/shqwen8b_respsample_convbestofN.json \
      exp/explanation_metrics.csv exp/explanation_metrics_ref_per_turn.csv
rm -rf exp/intrs_dims exp/intrs_dims_json exp/intrs_dims_qwen3_8b exp/intrs_dims_qwen3_8b_think exp/intrs_dims_llama32_3b

echo "[1] dev sample (seed 2026)"
$PY src/explanation_metrics/build_dev_responses.py --n_sess 50

echo "[2] response generation (convbestof20_vllm.py, 4 GPUs) $(date)"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PYV src/response/convbestof20_vllm.py \
    --input exp/inference/devset/shqwen8b_respsample_shard$i.json \
    --blind_parquet $DEV_PARQ \
    --out exp/inference/devset/shqwen8b_respsample_shard${i}_convbestofN.json \
    --n 20 --gen google/gemma-3n-E4B-it --judge google/gemma-4-E2B-it > logs/devresp_shard$i.log 2>&1 &
done; wait
[ $(ls exp/inference/devset/shqwen8b_respsample_shard*_convbestofN.json | wc -l) -eq 4 ] || { echo "generation FAILED"; exit 1; }

echo "[3] merge"
$PY - <<'EOF'
import json, glob
d = []
for p in sorted(glob.glob('exp/inference/devset/shqwen8b_respsample_shard*_convbestofN.json')):
    if not p.endswith('.cands.json'): d += json.load(open(p))
json.dump(d, open('exp/inference/devset/shqwen8b_respsample_convbestofN.json', 'w'), ensure_ascii=False, indent=1)
print(len(d), 'turns')
EOF

echo "[4] literature metrics $(date)"
$PY src/explanation_metrics/reference_free_metrics.py > logs/metrics175.log 2>&1
echo "[5] robustness $(date)"
CUDA_VISIBLE_DEVICES=0 $PY src/explanation_metrics/robustness_metrics.py > logs/metrics176.log 2>&1

echo "[6] dimensions IntRS x4 judges $(date)"
run_judge() {
  local extra=""; [ "$3" = think ] && extra="--thinking"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PYV src/explanation_metrics/userstudy_dimensions.py --shard $i --nshards 4 --vllm \
      --judge "$1" --tag "$2" $extra > logs/intrs_${2}_shard$i.log 2>&1 &
  done; wait
  [ $(ls exp/intrs_dims_$2/scores_[0-9]*.parquet | wc -l) -eq 4 ] || { echo "judge FAILED $1"; exit 1; }
  echo "  judge $1 ok $(date)"
}
run_judge google/gemma-4-E2B-it json
run_judge Qwen/Qwen3-8B qwen3_8b
run_judge meta-llama/Llama-3.2-3B-Instruct llama32_3b
run_judge Qwen/Qwen3-8B qwen3_8b_think think

echo "[7] combine"
$PY src/explanation_metrics/userstudy_dimensions.py --combine > logs/intrs_combine.log 2>&1
echo "FULL CHAIN DONE $(date)"
