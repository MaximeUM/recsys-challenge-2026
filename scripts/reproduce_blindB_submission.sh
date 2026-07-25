#!/bin/bash
# =============================================================================
# Reproduce our FINAL Blind-B submission (leaderboard composite 0.45, 9th/18).
#
#   Blind-B parquet  ->  RRF pool  ->  Qwen3-8B scoring-head rerank  ->
#   conversation best-of-20 responses  ->  predictions.json
#
# This is the "Stage 1 - Inference" entry point requested by the organizers:
#   Blind-Dataset-B -> pipeline -> predictions.json
#
# Requirements (see models/README.md and data/README.md):
#   models/qwen3_ft_dualencoder_4b_ctx1024   (dense retriever)
#   models/qwen3_8b_scorehead_top200         (listwise scoring-head reranker)
#   google/gemma-3n-E4B-it, google/gemma-4-E2B-it   (HF cache, gated)
#   data/TalkPlayData-Challenge-Blind-B/, -Track-Metadata/, -Track-Embeddings/, -Dataset/
#
# Runtime: ~30-50 min (pool ~15 min, rerank ~10 min, responses ~15-25 min).
# The rerank step needs ~17 GB of VRAM; on smaller cards run it with
# DEVICE_MAP=auto to shard across GPUs. Track ids are not bit-reproducible
# (bf16 greedy).
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/reproduce_blindB_submission.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."                       # repo root
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

B=data/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet
POOL=exp/combined_pool_ctx1024_blindB.parquet
PICKS=exp/inference/blindset_B/firstpos_scorehead_qwen_blindB.json
FINAL=exp/inference/blindset_B/firstpos_scorehead_qwen_blindB_convbestofN.json

# --- preflight ---------------------------------------------------------------
# Fail here with a readable message rather than deep inside SentenceTransformers,
# which would otherwise treat a missing local models/... path as a Hugging Face
# repo id and report an opaque HTTP 404.
missing=0
for f in "$B" \
         data/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet \
         data/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet \
         models/qwen3_ft_dualencoder_4b_ctx1024 \
         models/qwen3_8b_scorehead_top200 \
         cache/content_desc_full.json ; do
  [ -e "$f" ] || { echo "MISSING: $f" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  cat >&2 <<'EOF'

Preflight failed. Depending on what is missing:
  datasets                  python scripts/download_data.py --only inference
  fine-tuned weights        see models/README.md
  cache/content_desc_full.json
                            rebuild it (GPU, ~25 min, no training data needed):
                              python src/reranking/build_content_descriptors.py
                              python src/reranking/build_full_descriptors.py --desc_only
EOF
  exit 1
fi
echo "preflight OK - all required data, weights and caches present"

echo "[1/4] Hybrid retrieval pool (BM25 + dense + item-CF + artist, RRF, top-200)"
python src/retrieval/build_pool_blind.py --blind_parquet "$B" --out "$POOL"

echo "[2/4] Listwise scoring-head rerank (Qwen3-8B, top-200 -> top-20)"
python src/reranking/predict_scorehead_blind.py           # reads $POOL, writes $PICKS

echo "[3/4] Conversation best-of-20 response generation (gemma-3n gen, gemma-4 judge, clean+diverse)"
python src/response/convbestof20_diverse.py --input "$PICKS" --blind_parquet "$B" --out "$FINAL" --n 20

echo "[4/4] Validate against the official scorer format"
python scripts/validate_submission.py "$FINAL"

echo "DONE -> submission file: $FINAL"