# 1. Hybrid retrieval

With a ~47k-track catalog, no LLM can hold every track in context. This stage
narrows the catalog to a high-recall candidate pool that the expensive LLM
stages then rerank.

## Channels

For each turn we build a pool by **Reciprocal Rank Fusion (RRF)** over four
channels:

| Channel | Signal | Notes |
|---|---|---|
| **BM25** | lexical | over track name, artist, album and community tags |
| **Dense** | semantic | dual-encoder fine-tuned from Qwen3-Embedding-4B |
| **Item-CF** | behavioural | item-based collaborative filtering from session-level co-occurrence |
| **Artist expansion** | behavioural | promotes artists already played in the session, ranked by dense similarity to the query |

Artist expansion is worth its own channel because the ground truth shares an
artist with an already-played track in **46.0%** of turns that have one, **40.3%**
overall (turn 1 has none). Recompute it with
`python src/retrieval/artist_overlap_stat.py`.

*"Shares an artist"* means a non-empty intersection between the two tracks' sets
of credited artists. The definition matters: 11.2% of the catalog credits more
than one artist, and on those rows `artist_name` and `artist_id` are not even the
same length, so taking a single element is not well defined. Two weaker readings
give 43.0% / 37.6% (first `artist_name`) and 40.9% / 35.8% (the whole `artist_id`
array compared as a string); `--compare` prints all three.

> **Note.** The channel itself applies the third, most conservative reading:
> `build_pool_dev.py` keys tracks by `str(artist_id)`, so a feature counts as a
> different artist from the soloist and expands to a narrower candidate set. The
> published pools and every recall figure below were produced that way.
> `--artist_match shared` switches the channel to the intersection rule quoted
> above; it is not the default precisely because the published pools would no
> longer be reproducible from this repository.

Already-played tracks are excluded twice: by catalog index, and by a
`name||artist` key that also catches re-releases of the same recording.

RRF beat both naive union and round-robin merging. A fifth channel scoring the
challenge's provided **BPR user–track latent factors** degraded the fusion and
was dropped.

## The context-length fix

The dense encoder originally ran at `seq=256, truncation='right'`. Because the
current user request is placed **last** in the query text, that setting threw
the request away on any long conversation — 76% of dev queries exceed 256
tokens (median 671).

Switching to `seq=1024, truncation_side='left'` keeps the recent turns and the
final request. Effect on dev nDCG@20, broken down by conversation length:

| Preceding turns | 256 / right | 1024 / left |
|---|---|---|
| 0 (cold start) | 0.1595 | 0.1611 (no change) |
| 1–2 | 0.1616 | 0.1753 (+0.014) |
| 3–5 | 0.1157 | 0.1343 (+0.019) |
| 6+ | 0.0992 | 0.1153 (+0.016) |
| **Overall** | **0.1285** | **0.1431** |

The fix does nothing for cold start (turn 1 has a short query that was never
truncated) and everything for warm, long conversations — exactly the predicted
signature.

> **Important:** the encoding/pool script must set the *same* `max_seq_length`
> and `truncation_side` as training, or train/inference mismatch silently costs
> most of the gain.

## Results

| Pool | recall@50 | recall@500 |
|---|---|---|
| BM25 + dense only (RRF) | 44.3% | 70.4% |
| all four channels | **46.7%** | **72.2%** |

recall@500 is the ceiling a perfect reranker could reach. Both rows come from a
single `build_pool_dev.py` run over the 8000 dev turns, with the **same** shipped
4B ctx-1024 dense channel on each — so the delta isolates item-CF and artist
expansion. Full combined-pool curve:

| K | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| BM25 + dense | 31.1% | 44.3% | 52.7% | 60.5% | 70.4% |
| all four channels | **33.9%** | **46.7%** | **55.0%** | **62.5%** | **72.2%** |

Two consistency checks: recall@50 = 0.467 and recall@200 = 0.625 are exactly the
GT-in-pool figures that [the reranking section](02_reranking.md) decomposes, and
the curve keeps climbing past the pool sizes we train on — the basis for the
paper's conclusion that the recoverable headroom is in **ranking**, not in
widening the pool.

## Scripts

| Script | Role |
|---|---|
| `train_dense_encoder.py` | fine-tunes Qwen3-Embedding-4B (LoRA on attention, `MultipleNegativesRankingLoss` with gather, no hard negatives) at seq 1024 / left truncation |
| `build_pool_dev.py` | builds the dev pool (all four channels + RRF, played tracks excluded) |
| `build_pool_blind.py` | same for a blind split, parameterised by `--blind_parquet` / `--out` |
| `recall_breakdown.py` | per-channel and per-turn-count recall breakdown |

## Commands

```bash
# train the dense retriever (multi-GPU)
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \
    src/retrieval/train_dense_encoder.py
# -> models/qwen3_ft_dualencoder_4b_ctx1024

# dev pool (top-200 per turn)
python src/retrieval/build_pool_dev.py
# -> exp/combined_pool_ctx1024_dev.parquet

# blind pool
python src/retrieval/build_pool_blind.py \
    --blind_parquet data/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet \
    --out exp/combined_pool_ctx1024_blindB.parquet

# recall analysis
python src/retrieval/recall_breakdown.py
```

Pool construction takes ~15–20 min on one GPU (it encodes the 47k catalog plus
every query). Output columns: `session_id`, `turn`, `pool` (JSON list of
candidate track ids, top-200).

Pool construction **is** bit-reproducible: re-running gives identical pools.

Next: [2. LLM reranking](02_reranking.md).
