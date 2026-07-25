# Models

All model weights live in this directory. Nothing here is versioned in git.
There are two kinds: **base models** pulled from Hugging Face, and **our
fine-tuned weights**, released separately on Hugging Face.

## 1. Base models (Hugging Face, some gated)

| Model | Role |
|---|---|
| `Qwen/Qwen3-Embedding-4B` | backbone of the dense retriever |
| `meta-llama/Llama-3.2-3B-Instruct` | backbone of the 3B rerankers |
| `Qwen/Qwen3-8B` | backbone of the submitted scoring-head reranker |
| `google/gemma-3n-E4B-it` | response generator (best-of-N) |
| `google/gemma-4-E2B-it` | local judge that selects among the N responses |
| `BAAI/bge-reranker-v2-m3` | cross-encoder (negative ablation only) |

Llama and Gemma are **gated**: accept the licence on the model page, then

```bash
huggingface-cli login        # or: export HF_TOKEN=...
```

Use the official repositories — do not substitute a community mirror, the
tokenizer marker tokens the scoring head relies on must match exactly.

## 2. Our fine-tuned weights

All five checkpoints and the derived-artifact dataset are grouped in one Hugging
Face collection:
**[RecSys Challenge 2026 — FPMs_UMONS](https://huggingface.co/collections/MaximeM/recsys-challenge-2026-fpms-umons-6a636e612bd9187457225c86)**.

```bash
python scripts/download_models.py                  # ~40 GB, into models/ and cache/
python scripts/download_models.py --only inference # only what Blind-B reads (~22 GB)
python scripts/download_models.py --no_artifacts   # weights only, no artifacts
python scripts/download_models.py --check          # verify an existing download
python scripts/download_models.py --check --checksums   # ... and hash every file
```

`--only inference` means the retriever, the submitted scoring head, **and** the
derived artifacts that the Blind-B path reads (`cache/content_desc_full.json`
and the Blind-B pool). It excludes the dev pool, which only the paper's dev
tables need. Combine it with `--no_artifacts` to fetch the two checkpoints alone
and rebuild the artifacts locally.

Every transfer is pinned to the revision recorded in
[`scripts/artifact_manifest.json`](../scripts/artifact_manifest.json) and checked
against the SHA-256 values stored there, so the weights you download are the
exact ones behind the published numbers. `--check` verifies presence and file
size; `--checksums` additionally hashes every file. Use `--no_pin` to bypass the
pin, and `--write_manifest` to re-record revisions after publishing new weights.

| Hugging Face repo | Local directory | Size | Role |
|---|---|---|---|
| [`Qwen3-Embedding-4B-recsys-challenge-2026-retriever-ctx1024`](https://huggingface.co/MaximeM/Qwen3-Embedding-4B-recsys-challenge-2026-retriever-ctx1024) | `qwen3_ft_dualencoder_4b_ctx1024` | 7.6G | **dense retriever** (seq 1024, left truncation) — the dense channel of the RRF pool |
| [`Qwen3-8B-recsys-challenge-2026-scorehead-top200`](https://huggingface.co/MaximeM/Qwen3-8B-recsys-challenge-2026-scorehead-top200) | `qwen3_8b_scorehead_top200` | 15G | **submitted Blind-B reranker** — composite **0.45** |
| [`Llama-3.2-3B-recsys-challenge-2026-scorehead-top200`](https://huggingface.co/MaximeM/Llama-3.2-3B-recsys-challenge-2026-scorehead-top200) | `llama32_3b_scorehead_top200` | 6.1G | scoring head, top-200 pool |
| [`Llama-3.2-3B-recsys-challenge-2026-scorehead-top50`](https://huggingface.co/MaximeM/Llama-3.2-3B-recsys-challenge-2026-scorehead-top50) | `llama32_3b_scorehead_ctx1024` | 6.1G | scoring head, top-50 pool |
| [`Llama-3.2-3B-recsys-challenge-2026-firstpos-intent-mm`](https://huggingface.co/MaximeM/Llama-3.2-3B-recsys-challenge-2026-firstpos-intent-mm) | `llama32_3b_intent_mm_ctx1024` | 6.1G | generative single-pick reranker |

Naming follows `<base model>-recsys-challenge-2026-<role>`. The two Qwen derivatives are
**Apache-2.0**; the three Llama derivatives are governed by the **Llama 3.2
Community License**, which is why their names begin with `Llama` and why
`LICENSE.txt` and `USE_POLICY.md` ship with those weights.

Together they cover every row of the paper's reranking table:

| Table row | Model |
|---|---|
| No reranker (pool order) | — computed by `src/retrieval/eval_pool_ndcg.py` |
| Generative firstpos (ref.) | `Llama-3.2-3B-recsys-challenge-2026-firstpos-intent-mm` |
| Scoring head — Llama-3.2-3B, top-50 | `Llama-3.2-3B-recsys-challenge-2026-scorehead-top50` |
| Scoring head — Llama-3.2-3B, top-200 | `Llama-3.2-3B-recsys-challenge-2026-scorehead-top200` |
| Scoring head — Qwen3-8B, top-200 | `Qwen3-8B-recsys-challenge-2026-scorehead-top200` |

> **The scoring heads are not standard checkpoints.** Their `config.json`
> declares `Qwen3Model` / `LlamaModel` — a backbone with no LM head — and the
> head itself lives in `score_head.pt`, which also records the marker token id.
> Downloading only the safetensors gives a silently unusable model, which is why
> `download_models.py --check` verifies that file by name.

### Which model produced which submission

| Submission | Reranker | Responses | Composite |
|---|---|---|---|
| Blind-B **final** | `Qwen3-8B-recsys-challenge-2026-scorehead-top200` | best-of-20 clean+diverse | **0.45** |
| Blind-B fallback | earlier firstpos recipe (not released) | best-of-20 clean | 0.43 |
| Blind-A best | earlier firstpos recipe (not released) | best-of-N (N=6) | 0.5295 |

All use the retriever above. The *earlier firstpos recipe* is a non-enriched
variant that predates the intent+mm reranker and scores lower on dev (0.168 vs
0.172). Its weights are not released — `train_firstpos.py` and
`predict_firstpos.py` retrain and run it from scratch — and the generative row
of the table is the intent+mm model.

## 3. Auxiliary artifact

The reranker renders each candidate with distilled `{sound: ... | themes: ...}`
descriptors. They are precomputed once into:

```
cache/content_desc_full.json
```

`download_models.py` fetches it, together with the precomputed RRF pools, from
[`MaximeM/recsys-challenge-2026-artifacts`](https://huggingface.co/datasets/MaximeM/recsys-challenge-2026-artifacts).

Or rebuild it from your own copy of the challenge data (~25 min on one GPU, no
training data required):

```bash
python src/reranking/build_content_descriptors.py
python src/reranking/build_full_descriptors.py --desc_only
```

## Note on reproducibility

Reranker inference runs in bf16 with greedy decoding, which is **not
bit-reproducible** on GPU: re-running the same model on the same pool changes
roughly 40% of the picks when the model is undecided, moving dev nDCG@20 by
about ±0.008. Retrieval pools, by contrast, are exactly reproducible. Expect
submission files to differ slightly from ours run-to-run; the scores do not.
