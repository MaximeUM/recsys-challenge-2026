#!/bin/bash
# =============================================================================
# End-to-end smoke test: the WHOLE reproduce_paper.sh chain, our own models and
# hyperparameters, but on a mini dataset - so the full run's shape is exercised
# in a fraction of its time.
#
# What it proves: every stage runs, and each one hands the next the artifact it
# expects (retriever -> pool -> descriptors -> reranker datasets -> four LoRA
# trainings -> sharded dev prediction -> nDCG table), on the exact base models,
# tokenizers, markers and context lengths the paper used. What it does NOT
# prove: any number. The metrics come from ~40 dev sessions, a 4k-track catalog
# and a handful of training steps - they are noise by construction.
#
# Nothing is substituted or scaled down except the data itself, plus a step cap
# on the two top-200 scoring heads (see SCOREHEAD_ARGS below). This therefore
# needs the same hardware as the real run: 4 GPUs. On less, it refuses to start
# rather than silently testing something else.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/smoke_test.sh              # everything
#   bash scripts/smoke_test.sh retrieval    # one stage (same names as reproduce_paper.sh)
#
# Knobs (all optional):
#   GPU_IDS=0,1,2,3  which cards to use (default: the first four visible)
#   MINI=data_mini   where the mini dataset lives (rebuilt if absent)
#   SANDBOX=.smoke   working tree for the run
#   REBUILD=1        rebuild the mini dataset even if it is already there
#
# Everything the run writes - mini data, LoRA weights, caches, pools, picks -
# stays inside $SANDBOX and $MINI, both git-ignored. Your real data/, models/
# and exp/ are never touched: the sandbox has its own models/, cache/, exp/ and
# logs/, and its data/ points at the mini dataset.
# =============================================================================
set -e
cd "$(dirname "$0")/.."                       # repo root

STAGE="${1:-all}"
MINI="${MINI:-data_mini}"
SANDBOX="${SANDBOX:-.smoke}"
NEED_GPUS=4                                   # the configuration the paper was run on

# --- 1. hardware -------------------------------------------------------------
# The 8B scoring head at 14,336 tokens does not fit on fewer cards, and running
# the chain on a different model or a shorter context would no longer be a test
# of what we actually ship.
if [ -z "$GPU_IDS" ]; then
  VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  [ "$VISIBLE" -ge "$NEED_GPUS" ] || {
    echo "This smoke test runs the published configuration: $NEED_GPUS GPUs required, $VISIBLE visible." >&2
    echo "Run it on the training box, or pass GPU_IDS explicitly if nvidia-smi is not available." >&2
    exit 1; }
  GPU_IDS=$(seq -s, 0 $((NEED_GPUS - 1)))
fi
IFS=',' read -r -a IDS <<< "$GPU_IDS"
[ "${#IDS[@]}" -eq "$NEED_GPUS" ] || {
  echo "GPU_IDS must list exactly $NEED_GPUS cards, got '${GPU_IDS}'." >&2; exit 1; }

# --- 2. mini dataset ---------------------------------------------------------
if [ -n "$REBUILD" ] || [ ! -f "$MINI/MINI_INFO.json" ]; then
  echo "### [mini] Building $MINI/ from data/"
  python scripts/make_mini_dataset.py --src data --out "$MINI"
else
  echo "### [mini] Reusing $MINI/ (REBUILD=1 to rebuild)"
fi

# --- 3. sandbox --------------------------------------------------------------
# reproduce_paper.sh does `cd "$(dirname "$0")/.."`, and bash resolves that
# logically, so invoking it through $SANDBOX/scripts/ makes it treat $SANDBOX as
# the repo root: every relative path it uses (data/, models/, cache/, exp/)
# lands in the sandbox, while src/ and scripts/ are the real ones.
echo "### [sandbox] $SANDBOX/ (src, scripts -> repo; data -> $MINI)"
mkdir -p "$SANDBOX"/{models,cache,exp,logs}
ln -sfn ../src     "$SANDBOX/src"
ln -sfn ../scripts "$SANDBOX/scripts"
case "$MINI" in /*) ln -sfn "$MINI" "$SANDBOX/data";;   # absolute stays absolute
                 *) ln -sfn "../$MINI" "$SANDBOX/data";; esac

# --- 4. run ------------------------------------------------------------------
export GPU_IDS
export NGPU="${#IDS[@]}"
export GPU="${IDS[0]}" SFT_GPU="${IDS[0]}"    # single-process builders pick one card
export EXPECT_ROWS=0                          # the mini dev split is not the 8,000-turn one
# The only training shortcut: the two top-200 scoring heads see 64 records for
# one epoch instead of the full set for up to four. Same model, same context.
export SCOREHEAD_ARGS="--max_records 64 --max_epochs 1 --eval_steps 20 --val_eval_size 16"

echo "### [config] GPUs $GPU_IDS | published base models and context lengths"
START=$SECONDS
bash "$SANDBOX/scripts/reproduce_paper.sh" "$STAGE"
echo "### SMOKE TEST OK - stage '$STAGE' in $(( (SECONDS - START) / 60 ))m$(( (SECONDS - START) % 60 ))s"
echo "### Artifacts under $SANDBOX/ - the numbers are meaningless by construction."
