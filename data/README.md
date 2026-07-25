# Data

The pipeline reads everything from this `data/` directory. Nothing here is
versioned in git — all files come from the official challenge datasets on
Hugging Face (`talkpl-ai`). We use **no external resource** beyond these.

The artifacts *we* release — fine-tuned weights, precomputed descriptors and RRF
pools — are not challenge data and live elsewhere: see
[`models/README.md`](../models/README.md) and the Hugging Face collection
**[RecSys Challenge 2026 — FPMs_UMONS](https://huggingface.co/collections/MaximeM/recsys-challenge-2026-fpms-umons-6a636e612bd9187457225c86)**.

## Download

```bash
source .venv/bin/activate

python scripts/download_data.py                       # all seven datasets (~875 MB)
python scripts/download_data.py --only inference      # only what Blind-B inference needs
python scripts/download_data.py --only blind-b track-metadata
python scripts/download_data.py --check               # verify without re-fetching
python scripts/download_data.py --evaluator           # also clone the official evaluator
```

The script downloads from the `talkpl-ai` organisation, then verifies that every
expected parquet file is present and exits non-zero if any is missing. It is
safe to re-run: transfers resume and already-complete files are skipped.

Sizes: Track-Embeddings dominates at 756 MB; everything else together is under
120 MB.

| Short name | Repository | Size |
|---|---|---|
| `dataset` | TalkPlayData-Challenge-Dataset | 91 MB |
| `blind-a` | TalkPlayData-Challenge-Blind-A | 476 KB |
| `blind-b` | TalkPlayData-Challenge-Blind-B | 132 KB |
| `track-metadata` | TalkPlayData-Challenge-Track-Metadata | 20 MB |
| `track-embeddings` | TalkPlayData-Challenge-Track-Embeddings | 756 MB |
| `user-metadata` | TalkPlayData-Challenge-User-Metadata | 384 KB |
| `user-embeddings` | TalkPlayData-Challenge-User-Embeddings | 7.5 MB |

`--only inference` selects `dataset`, `blind-b`, `track-metadata` and
`track-embeddings` — the four needed by
`scripts/reproduce_blindB_submission.sh`.

Everything downloaded here is git-ignored; only this README is versioned.

## Expected layout

```
data/
├── TalkPlayData-Challenge-Dataset/data/
│   ├── train-00000-of-00001.parquet          # training conversations
│   └── test-00000-of-00001.parquet           # dev split: 1000 sessions x 8 turns, GT known
├── TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet
├── TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet
├── TalkPlayData-Challenge-Track-Metadata/data/
│   ├── all_tracks-00000-of-00001.parquet     # catalog: 47,071 tracks
│   └── test_tracks-00000-of-00001.parquet
├── TalkPlayData-Challenge-Track-Embeddings/data/
│   ├── all_tracks-0000{0..3}-of-00004.parquet
│   └── test_tracks-00000-of-00001.parquet
├── TalkPlayData-Challenge-User-Metadata/data/
└── TalkPlayData-Challenge-User-Embeddings/data/
```

## What each split is used for

| Split | Role in the paper |
|---|---|
| `Dataset/train` | reranker training data; item-CF co-occurrence channel |
| `Dataset/test` (**dev**) | our held-out test set: all ablations and Table 2. Never used for training or model selection. |
| `Blind-A` | first blind phase (best composite 0.5295) |
| `Blind-B` | final phase, 3 submissions allowed (best composite **0.45**, 9th/18 academic teams) |

The **dev split** is the official `test` split of the challenge dataset: its
ground truth is distributed, which is why we can use it locally as a test set.
The train/validation split used for reranker early stopping is carved out of
`Dataset/train` **at session level** (all turns of a conversation stay on the
same side) to avoid leakage.

## Track embeddings

Provided per track, ~99% coverage over the 47k catalog:

| Field | Dim | Used? |
|---|---|---|
| `audio-laion_clap` | 512 | yes — distilled into the `sound:` text descriptor |
| `lyrics-qwen3_embedding_0.6b` | 1024 | yes — distilled into the `themes:` text descriptor |
| `image-siglip2` | 768 | no (negative result: unusable without raw cover images) |
| `cf-bpr` | 128 | no (degraded RRF fusion, dropped) |
| `attributes-qwen3...`, `metadata-qwen3...` | 1024 | not used in the final pipeline |

See [`docs/05_ablations_negative_results.md`](../docs/05_ablations_negative_results.md)
for why raw embeddings hurt and only their **text distillation** helps.

## Official evaluator (optional)

Some scripts reuse the challenge's own metric implementations. Clone the
official evaluator at the repo root — `download_data.py --evaluator` does this
for you, or manually:

```bash
git clone https://github.com/nlp4musa/music-crs-evaluator
```

It is needed by `src/common/score_submission.py`, and by the explanation-quality
evaluation suite in `src/explanation_metrics/`, which imports
`compute_lexical_diversity` from it.
