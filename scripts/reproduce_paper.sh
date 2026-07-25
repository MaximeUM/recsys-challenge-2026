#!/bin/bash
# =============================================================================
# Train the pipeline FROM SCRATCH and regenerate the paper's reranking table.
# This is the "Stage 2 - Reproducibility" entry point requested by the
# organizers. It is a long, multi-GPU job: run stages individually.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/reproduce_paper.sh <stage>
#
# Stages, in dependency order:
#   retrieval    train the dense encoder, build the dev pool, print recall@k
#   descriptors  distil audio neighbours into cache/content_desc.json
#   data         build the raw reranker datasets (top-50 and top-200)
#   enrich       add {sound|themes} to every candidate; also writes the
#                cache/content_desc_full.json that inference consumes
#   firstpos     train the generative single-pick reranker (table row 2)
#   scorehead    train the three listwise scoring heads (table rows 3-5)
#   eval         dev nDCG@20 for every row of the table
#   all          every stage above, in order
#
# The reranking table has five rows; each is produced here:
#
#   row 1  no reranker, retrieval pool order   eval_pool_ndcg.py
#   row 2  generative firstpos (intent+mm)     train_firstpos_intent_mm.py
#   row 3  scoring head, Llama-3.2-3B, top-50  train_scorehead_top50.py
#   row 4  scoring head, Llama-3.2-3B, top-200 train_scorehead.py
#   row 5  scoring head, Qwen3-8B, top-200     train_scorehead.py --base Qwen/Qwen3-8B
#
# Note on row 2: the published generative reference is the *intent+mm* variant,
# trained on the ENRICHED top-50 candidates with an intent-emphasising prompt.
# An earlier, non-enriched firstpos recipe exists in the project history and
# scores lower; it is not what the table reports.
#
# Artifact chain (each stage consumes the previous stage's exact output):
#
#   descriptors -> cache/content_desc.json
#   data        -> _sft_dataset_cache/sft_ctx1024_60000_top{50,200}_firstpos.parquet
#   enrich      -> cache/content_desc_full.json
#                  _sft_dataset_cache/sft_ctx1024_enriched_firstpos.parquet
#                  _sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet
#
# Run across several machines; see the README for per-stage VRAM requirements
# rather than a machine list. Expect several hours per training stage.
# Reranker training uses LoRA rank 16 with early stopping (scoring heads
# overfit from epoch 2).
#
# Smoke test: to check that the whole chain still runs without waiting for a
# full training round, use scripts/smoke_test.sh. It drives this same script,
# with these same models and context lengths, over the mini dataset built by
# scripts/make_mini_dataset.py, through the GPU_IDS / SCOREHEAD_ARGS /
# EXPECT_ROWS variables defined below. Unset, they leave every command exactly
# as it was run for the paper.
# =============================================================================
set -e
cd "$(dirname "$0")/.."                       # repo root
STAGE="${1:-all}"

NGPU="${NGPU:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"     # dev prediction shards, one per GPU

# Extra flags for the two top-200 scoring-head trainings, appended after the
# published ones so they win (argparse keeps the last occurrence). Empty by
# default: the commands below then run exactly as they did for the paper.
# scripts/smoke_test.sh uses this to cap the step count on a mini dataset.
SCOREHEAD_ARGS="${SCOREHEAD_ARGS:-}"
# Strictness of the dev combiners: 0 disables the row check (mini dev splits).
EXPECT_ROWS="${EXPECT_ROWS:-8000}"
EXPECT="--expect_shards ${#GPU_LIST[@]} --expect_rows $EXPECT_ROWS"

launch() {  # accelerate rejects --multi_gpu on a single process
  if [ "$NGPU" -gt 1 ]; then
    accelerate launch --multi_gpu --num_processes "$NGPU" --gpu_ids "$GPU_IDS" "$@"
  else
    accelerate launch --num_processes 1 --gpu_ids "$GPU_IDS" "$@"
  fi
}

CACHE_DIR=models/_sft_dataset_cache
RAW_TOP50=$CACHE_DIR/sft_ctx1024_60000_top50_firstpos.parquet
RAW_TOP200=$CACHE_DIR/sft_ctx1024_60000_top200_firstpos.parquet
ENR_TOP50=$CACHE_DIR/sft_ctx1024_enriched_firstpos.parquet
ENR_TOP200=$CACHE_DIR/sft_ctx1024_enriched_top200_firstpos.parquet

require() {  # fail early with a readable message instead of a remote-model 404
  for f in "$@"; do
    [ -e "$f" ] || { echo "MISSING required artifact: $f" >&2
                     echo "Run the earlier stage that produces it (see header)." >&2
                     exit 1; }
  done
}

shards_ok() {  # $1=name - refuse to combine an incomplete or stale sharded run
  local n want=${#GPU_LIST[@]}
  n=$(ls exp/picks/picks_"$1"_*.parquet 2>/dev/null | wc -l)
  [ "$n" -eq "$want" ] || { echo "EXPECTED $want shards for '$1', found $n." >&2
                            echo "Clear exp/picks/ and re-run the prediction step." >&2
                            exit 1; }
}

run_retrieval() {
  echo "### [retrieval] Train the dense dual-encoder (Qwen3-Embedding-4B, ctx 1024, left-truncated)"
  launch src/retrieval/train_dense_encoder.py

  echo "### [retrieval] Build the dev pool (RRF over 4 channels, top-200)"
  # Also prints the paper's recall@K comparison: BM25+dense baseline vs the
  # four-channel combined pool, on the full 8000-turn dev split.
  python src/retrieval/build_pool_dev.py
}

run_descriptors() {
  echo "### [descriptors] Audio-neighbour consensus tags -> cache/content_desc.json"
  python src/reranking/build_content_descriptors.py
  require cache/content_desc.json
}

run_data() {
  echo "### [data] Raw reranker dataset, top-50"
  python src/reranking/build_reranker_data.py --n_rag_top 50  --out "$RAW_TOP50"
  echo "### [data] Raw reranker dataset, top-200"
  python src/reranking/build_reranker_data.py --n_rag_top 200 --out "$RAW_TOP200"
  require "$RAW_TOP50" "$RAW_TOP200"
}

run_enrich() {
  require cache/content_desc.json "$RAW_TOP50" "$RAW_TOP200"
  echo "### [enrich] Lyrical themes -> cache/content_desc_full.json + enriched top-50"
  python src/reranking/build_full_descriptors.py
  require cache/content_desc_full.json "$ENR_TOP50"

  echo "### [enrich] Inject {sound|themes} into the top-200 candidates"
  python src/reranking/enrich_reranker_data.py
  require "$ENR_TOP200"
}

run_firstpos() {
  require "$ENR_TOP50"
  echo "### [row 2] Generative single-pick reranker, intent+mm (enriched top-50)"
  launch src/reranking/train_firstpos_intent_mm.py \
      --cache "$ENR_TOP50" --output models/llama32_3b_intent_mm_ctx1024 \
      --n_cand 50 --max_seq_len 3584
}

run_scorehead() {
  require "$ENR_TOP50" "$ENR_TOP200"
  echo "### [row 3] Scoring head - Llama-3.2-3B, top-50 pool"
  launch src/reranking/train_scorehead_top50.py \
      --cache "$ENR_TOP50" --output models/llama32_3b_scorehead_ctx1024 \
      --max_seq_len 4096

  echo "### [row 4] Scoring head - Llama-3.2-3B, top-200 pool"
  launch src/reranking/train_scorehead.py \
      --cache "$ENR_TOP200" --output models/llama32_3b_scorehead_top200 \
      --max_seq_len 14336 --n_cand 200 $SCOREHEAD_ARGS

  echo "### [row 5] Scoring head - Qwen3-8B, top-200 pool (the submitted reranker)"
  launch src/reranking/train_scorehead.py \
      --cache "$ENR_TOP200" --output models/qwen3_8b_scorehead_top200 \
      --base Qwen/Qwen3-8B --mark_str '<|box_end|>' \
      --max_seq_len 14336 --n_cand 200 $SCOREHEAD_ARGS
}

# $1=script  $2=model dir  $3=run name  $4=n_cand  $5=maxlen
# One shard per GPU listed in GPU_IDS; with a single GPU the shards run in sequence.
predict_sharded() {
  require "$2"
  local n=${#GPU_LIST[@]} s
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$s]}" python "$1" --path "$2" --name "$3" \
        --shard "$s" --nshards "$n" --n_cand "$4" --maxlen "$5" &
  done
  wait
  shards_ok "$3"
}

run_eval() {
  mkdir -p exp/picks
  echo "### [row 1] No reranker - retrieval pool order"
  python src/retrieval/eval_pool_ndcg.py

  echo "### [row 2] Generative firstpos (intent+mm)"
  predict_sharded src/reranking/predict_firstpos_dev.py \
      models/llama32_3b_intent_mm_ctx1024 intentmm 50 3584
  python src/reranking/eval_firstpos_ndcg.py --names intentmm $EXPECT

  echo "### [row 3] Scoring head - Llama-3.2-3B, top-50"
  predict_sharded src/reranking/predict_scorehead_dev.py \
      models/llama32_3b_scorehead_ctx1024 sh_llama_top50 50 4096
  python src/reranking/eval_scorehead_ndcg.py --name sh_llama_top50 $EXPECT

  echo "### [row 4] Scoring head - Llama-3.2-3B, top-200"
  predict_sharded src/reranking/predict_scorehead_dev.py \
      models/llama32_3b_scorehead_top200 sh_llama_top200 200 14336
  python src/reranking/eval_scorehead_ndcg.py --name sh_llama_top200 $EXPECT

  echo "### [row 5] Scoring head - Qwen3-8B, top-200"
  predict_sharded src/reranking/predict_scorehead_dev.py \
      models/qwen3_8b_scorehead_top200 shqwen8b 200 14336
  python src/reranking/eval_scorehead_ndcg.py --name shqwen8b $EXPECT

  echo "### Table 1 summary -> exp/table1_<today>.csv"
  python src/reranking/table1_summary.py $EXPECT
}

case "$STAGE" in
  retrieval)   run_retrieval ;;
  descriptors) run_descriptors ;;
  data)        run_data ;;
  enrich)      run_enrich ;;
  firstpos)    run_firstpos ;;
  scorehead)   run_scorehead ;;
  eval)        run_eval ;;
  all)         run_retrieval; run_descriptors; run_data; run_enrich
               run_firstpos; run_scorehead; run_eval ;;
  *) echo "Unknown stage: $STAGE (see header for the list)"; exit 1 ;;
esac

echo "DONE - stage: $STAGE"
