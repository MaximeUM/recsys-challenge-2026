# 2. LLM reranking

This is where the paper's central finding lives: **picking and ranking are
distinct skills**.

Given the top of the RRF pool (top-50 or top-200 depending on configuration; the
retrieval itself is evaluated to depth 500), an LLM conditioned on the full
conversation reranks the candidates. We compare two training objectives.

## The candidate representation

Each candidate is one line of text:

```
12. Thunderstruck by AC/DC [rock, 80s, classic rock] {sound: hard rock, metal | themes: rebellion, anger}
```

- **title, artist** — from the catalog metadata
- **`[...]`** — up to five community tags
- **`{sound: ...}`** — the consensus community tags of the track's nearest
  neighbours in the **audio** (CLAP) embedding space
- **`{themes: ...}`** — its closest entries in a 22-item lyrical-topic lexicon,
  in the **lyrics** embedding space

The `sound` and `themes` fields are derived from the challenge's provided
embeddings, but passed as **words**, not vectors. This is deliberate: the
reranker is a language model, and as the [ablations](05_ablations_negative_results.md)
show, the multimodal signal transfers only once rendered as text. Fusing the raw
embeddings as extra retrieval channels *underperformed* the text-only pool.

The system prompt weighs the current request over the history and penalises
candidates whose sound or themes contradict an explicit rejection; the
conversation always ends with an explicit `user (REQUEST):` marker.

For the listwise variant, each candidate line ends with a **single-token
marker** whose hidden state the scoring head reads. The target index never
appears in the text — it only enters the loss.

## Objective A — generative single-pick (firstpos)

A 3B LLM (LoRA) is trained to generate the index of the single best candidate,
with the **loss on the first generated position only**.

The motivation is that the task has one ground truth per turn, so
`nDCG@20 = 1/log2(rank_GT + 1)` — only the rank of the correct track matters.
Training to emit one confident pick therefore matches both the task and the
scoring.

| Training target | Dev nDCG@20 |
|---|---|
| Full 20-position list | 0.1386 |
| **Single confident pick** | **0.1604** |

The stage lifts the retrieval pool from 0.1431 to **0.1681 (+17%)**, and the
gain is present at every turn depth — the reranker never degrades on average.

## Objective B — listwise scoring head (scorehead)

Instead of generating, we append a marker token after each candidate, read its
hidden state from a **single forward pass**, and map each marker to a score with
a linear head, trained with listwise cross-entropy.

This ranks the whole pool in one pass at comparable training cost, and it
underlies our best Blind-B submission.

## The comparison

| Reranker | Training pool | TOP-1 | TOP-20 |
|---|---|---|---|
| Generative firstpos (reference) | top-50 | 0.172 | 0.172 |
| Scoring head — Llama-3.2-3B | top-50 | 0.175 | 0.202 |
| Scoring head — Llama-3.2-3B | top-200 | 0.178 | 0.216 |
| Scoring head — Qwen3-8B | top-200 | 0.177 | **0.219** |

- **TOP-1** = the single best pick, remaining candidates left in pool order.
- **TOP-20** = the full learned ranking, which is what the leaderboard scores.

The objectives **pick** equally well (0.172–0.178). The scoring head's **ranking**
is far stronger (0.219 vs 0.172), because it learns to order positions 2–20
rather than leaving them in retrieval order. That difference is the primary
lever behind the leaderboard's nDCG@20.

## Why training on a deeper pool helps

Training the scoring head on top-200 rather than top-50 moves TOP-20 from 0.202
to 0.216. Decomposing `nDCG = P(GT in pool) x conditional ranking quality`:

| Training pool | GT-in-pool | Conditional | Net TOP-20 |
|---|---|---|---|
| top-50 | 0.467 | 0.433 | 0.202 |
| top-200 | 0.625 | 0.345 | 0.216 |

The conditional term falls not because the model regresses, but by composition:
the "GT in pool" set now includes the hard cases (ground truth at retrieval
ranks 50–200, weakly connected) that we rank poorly. Recall rises faster than it
dilutes, so the net moves up.

## Training discipline

- **LoRA rank 16** (alpha 32) on a frozen backbone. Verified against the
  trainers and the repository history; earlier drafts said 64, which came from
  an abandoned catalog-pretraining branch, not from these rerankers.
- **Session-level train/validation split** of the *train* set — every turn of a
  conversation stays on the same side, otherwise the validation signal leaks.
- **Early stopping.** The loss curves show a clean overfit from **epoch 2**
  (validation rises while training falls); early stopping keeps the ~2-epoch
  checkpoint. Without it, 4 epochs would have degraded the model.
- The dev split is the held-out test set and is **never** used for training or
  model selection.

Full fine-tuning instead of LoRA *degrades* TOP-20 (0.204 vs 0.216) — see the
[ablations](05_ablations_negative_results.md).

## Scripts

| Script | Role |
|---|---|
| `build_content_descriptors.py` | audio-neighbour consensus tags -> `cache/content_desc.json` |
| `build_reranker_data.py` | builds the raw training set: conversation + N candidates + target index |
| `build_full_descriptors.py` | adds lyrical themes, writes `cache/content_desc_full.json` (the artifact inference reads) and the enriched top-50 set |
| `enrich_reranker_data.py` | injects `{sound \| themes}` into the top-200 training set |
| `train_firstpos.py` | trains the generative single-pick reranker |
| `train_scorehead.py` | trains the listwise scoring head (LoRA, val monitoring, early stopping, merges the best adapter) |
| `predict_scorehead_dev.py` | scores the dev pool, shardable across GPUs |
| `eval_scorehead_ndcg.py` | combines shards into dev nDCG@20 (TOP-1 and TOP-20) |
| `predict_scorehead_blind.py` | Blind-B inference with the submitted Qwen3-8B head |

## Commands

The artifact chain matters — each step consumes the previous step's exact
output, and `scripts/reproduce_paper.sh` fails early if one is missing:

```
build_content_descriptors  ->  cache/content_desc.json
build_reranker_data        ->  sft_ctx1024_60000_top{50,200}_firstpos.parquet
build_full_descriptors     ->  cache/content_desc_full.json  (+ enriched top-50)
enrich_reranker_data       ->  sft_ctx1024_enriched_top200_firstpos.parquet
train_scorehead            ->  models/qwen3_8b_scorehead_top200
```

```bash
# 1. audio-neighbour descriptors (GPU, ~20 min)
python src/reranking/build_content_descriptors.py

# 2. raw training sets
python src/reranking/build_reranker_data.py --n_rag_top 50 \
    --out models/_sft_dataset_cache/sft_ctx1024_60000_top50_firstpos.parquet
python src/reranking/build_reranker_data.py --n_rag_top 200 \
    --out models/_sft_dataset_cache/sft_ctx1024_60000_top200_firstpos.parquet

# 3. lyrical themes -> content_desc_full.json, then enrich the top-200 set
python src/reranking/build_full_descriptors.py     # GPU
python src/reranking/enrich_reranker_data.py       # CPU only

# 3a. train the scoring head - Llama-3.2-3B
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \
    src/reranking/train_scorehead.py \
    --cache models/_sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet \
    --output models/llama32_3b_scorehead_top200 \
    --max_seq_len 14336 --n_cand 200

# 3b. train the scoring head - Qwen3-8B (the submitted reranker)
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \
    src/reranking/train_scorehead.py \
    --cache models/_sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet \
    --output models/qwen3_8b_scorehead_top200 \
    --base Qwen/Qwen3-8B --mark_str '<|box_end|>' \
    --max_seq_len 14336 --n_cand 200

# 4. evaluate on dev (4-way sharded)
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python src/reranking/predict_scorehead_dev.py \
      --path models/qwen3_8b_scorehead_top200 --name shqwen8b \
      --shard $s --nshards 4 --n_cand 200 --maxlen 14336 &
done; wait
python src/reranking/eval_scorehead_ndcg.py --name shqwen8b
```

The marker token is backbone-specific: `<|reserved_special_token_5|>` for Llama,
`<|box_end|>` for Qwen. It must be a **single** token, which is why community
mirrors of the base models are not interchangeable here.

`--no_goal` on the dev prediction script masks `conversation_goal`, simulating
the Blind-B condition where the goal field is null.

Next: [3. Response generation](03_response_generation.md).
