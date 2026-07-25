#!/bin/bash
# FULL-DEV explanation EVALUATION (8,000 turns), fully reproducible:
#   VLLM_BATCH_INVARIANT=1 throughout (verified bitwise determinism), generation seed 2026,
#   direct judges: one run (greedy; verified seed-independent), thinking: three seeds (2026/1337/7).
#   bash src/explanation_metrics/run_fulldev_evals.sh (logs: logs/run_fulldev.log; detachable with setsid)
set -e
cd "$(dirname "$0")/../.." || exit 1   # repository root
mkdir -p logs exp
PY=python
PYV=python
export VLLM_BATCH_INVARIANT=1
DEV_PARQ=data/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet
PFX=shqwen8b_fulldev
MERGED=exp/inference/devset/${PFX}_convbestofN.json

echo "[1] picks full-dev (1000 sessions) $(date)"
$PY src/explanation_metrics/build_dev_responses.py --n_sess 1000 --prefix $PFX

echo "[2] response generation (convbestof20_vllm.py, batch-invariant, 4 GPUs) $(date)"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PYV src/response/convbestof20_vllm.py \
    --input exp/inference/devset/${PFX}_shard$i.json --blind_parquet $DEV_PARQ \
    --out exp/inference/devset/${PFX}_shard${i}_convbestofN.json --n 20 \
    --gen google/gemma-3n-E4B-it --judge google/gemma-4-E2B-it > logs/fulldev_gen_shard$i.log 2>&1 &
done; wait
[ $(ls exp/inference/devset/${PFX}_shard*_convbestofN.json | grep -vc cands) -eq 4 ] || { echo "generation FAILED"; exit 1; }

echo "[3] merge $(date)"
$PYV - <<EOF
import json, glob
d = []
for p in sorted(glob.glob('exp/inference/devset/${PFX}_shard*_convbestofN.json')):
    if not p.endswith('.cands.json'): d += json.load(open(p))
json.dump(d, open('$MERGED', 'w'), ensure_ascii=False, indent=1)
print(len(d), 'turns')
EOF

# The PUBLISHED protocol excludes 91 gold placeholder responses (`Unknown message`),
# yielding 7,909 paired turns with --drop_unknown. Also produce an inclusive 8,000-turn
# diagnostic under a distinct name to prevent overwriting.
echo "[4] literature metrics — PUBLISHED, without placeholders (7909) $(date)"
$PY src/explanation_metrics/reference_free_metrics.py --dev_resp $MERGED --drop_unknown \
  --out exp/explanation_metrics_fulldev_nounknown.csv > logs/fulldev_175_nounknown.log 2>&1
echo "[4b] literature metrics — diagnostic, with placeholders (8000) $(date)"
$PY src/explanation_metrics/reference_free_metrics.py --dev_resp $MERGED \
  --out exp/explanation_metrics_fulldev.csv > logs/fulldev_175.log 2>&1

echo "[5] robustness — PUBLISHED, without placeholders $(date)"
CUDA_VISIBLE_DEVICES=0 $PY src/explanation_metrics/robustness_metrics.py --dev_resp $MERGED \
  --drop_unknown > logs/fulldev_176_nounknown.log 2>&1
echo "[5b] robustness — diagnostic, with placeholders $(date)"
CUDA_VISIBLE_DEVICES=0 $PY src/explanation_metrics/robustness_metrics.py --dev_resp $MERGED \
  > logs/fulldev_176.log 2>&1

echo "[5c] check: the published run must contain exactly 7909 paired turns"
$PY - <<'EOF'
import pandas as pd, sys
df = pd.read_csv('exp/explanation_metrics_fulldev_nounknown.csv')
n = df.loc[df['set'].str.startswith('dev'), 'n']
if not (n == 7909).all():
    sys.exit(f'FAILED: expected 7909 paired turns, got {sorted(set(n))}')
print('OK — 7909 paired turns (91 placeholders excluded)')
EOF

echo "[6] dimensions IntRS (vLLM, batch-invariant) $(date)"
run_judge() {  # $1=judge $2=tag $3=seed $4=think?
  local extra=""; [ "$4" = think ] && extra="--thinking"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PYV src/explanation_metrics/userstudy_dimensions.py --shard $i --nshards 4 --vllm $extra \
      --judge "$1" --tag "$2" --seed $3 --dev_resp $MERGED > logs/fulldev_intrs_${2}_shard$i.log 2>&1 &
  done; wait
  [ $(ls exp/intrs_dims_$2/scores_[0-9]*.parquet 2>/dev/null | wc -l) -eq 4 ] || { echo "judge FAILED $2"; exit 1; }
  echo "  $2 ok $(date)"
}
run_judge google/gemma-4-E2B-it fulldev_gemma4 2026
run_judge Qwen/Qwen3-8B fulldev_qwen3 2026
run_judge meta-llama/Llama-3.2-3B-Instruct fulldev_llama 2026
for s in 2026 1337 7; do run_judge google/gemma-4-E2B-it fulldev_gemma4_think_s$s $s think; done
for s in 2026 1337 7; do run_judge Qwen/Qwen3-8B fulldev_qwen3_think_s$s $s think; done

echo "[7] dimension aggregation $(date)"
$PY - <<'EOF'
import glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
DIMS={'A':'Transparency','B':'Transparency','C':'Effectiveness','D':'Effectiveness','E':'Persuasion','F':'Trust','G':'Satisfaction'}
ORDER=['Transparency','Effectiveness','Persuasion','Trust','Satisfaction']
for d in sorted(glob.glob('exp/intrs_dims_fulldev_*')):
    fs=sorted(glob.glob(f'{d}/scores_[0-9]*.parquet'))
    if len(fs)<4: continue
    df=pd.concat([pd.read_parquet(f) for f in fs]); ok=df[df['score']>0].assign(dim=lambda x:x['item'].map(DIMS))
    piv=ok.pivot_table(index=['session_id','turn','cond'],columns='dim',values='score').reset_index()
    print(f'\n=== {d} ({len(df)} scores, {(df["score"]==0).sum()} failures) ===')
    for dim in ORDER:
        w=piv.pivot_table(index=['session_id','turn'],columns='cond',values=dim).dropna()
        st,p=wilcoxon(w['gen'],w['gold'])
        print(f'  {dim:14s} gen {w["gen"].mean():.3f}  gold {w["gold"].mean():.3f}  delta {w["gen"].mean()-w["gold"].mean():+.3f}  p={p:.1e} (n={len(w)})')
EOF
echo "[8] generated/gold failure breakdown by judge $(date)"
$PY src/explanation_metrics/judge_failure_breakdown.py
echo "FULLDEV DONE $(date)"
