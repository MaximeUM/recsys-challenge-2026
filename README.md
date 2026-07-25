# Picking is Not Ranking, and Explanation Quality Has Many Dimensions

Team **FPMs_UMONS** — RecSys Challenge 2026 (conversational music recommendation).

This repository contains the full pipeline behind our submission and the
ablations and negative results reported in the paper. It is meant to be re-run
end to end: `Blind-Dataset-B -> pipeline -> predictions.json`, and to retrain
every component from scratch. The explanation-quality evaluation suite behind
the paper's "Beyond the Judge Score" section is included, in
[`src/explanation_metrics/`](docs/04_explanation_quality.md).

> Maxime Manderlier and Fabian Lecron. *Picking is Not Ranking, and Explanation
> Quality Has Many Dimensions: Lessons for Conversational Music Recommendation.*
> RecSys Challenge 2026.
> Department of Technological Innovation Management, Faculty of Engineering,
> University of Mons (UMONS), Belgium.

---

## Results

Final Blind-B submission: **composite 0.45**, 9th of 18 academic teams.

| Component | Weight | Our score |
|---|---|---|
| nDCG@20 | 0.50 | 0.3587 |
| LLM-judge (personalization + explanation quality) | 0.30 | 3.50 / 5 |
| Catalog diversity | 0.10 | 0.0314 |
| Lexical diversity (distinct-2) | 0.10 | 0.8097 |
| **Composite** | | **0.4510** |

Everything runs on small **open-weight** models: a Qwen3-8B reranker, a
Qwen3-Embedding-4B retriever, and Gemma models for response generation and
local judging. No proprietary API is called anywhere in the pipeline.

### The central finding

A reranker does two different jobs: **picking** the track to recommend, and
**ranking** the remaining candidates. They turn out to be distinct skills.

| Reranker | Training pool | TOP-1 | TOP-20 |
|---|---|---|---|
| Generative single-pick (firstpos) | top-50 | 0.172 | 0.172 |
| Scoring head — Llama-3.2-3B | top-50 | 0.175 | 0.202 |
| Scoring head — Llama-3.2-3B | top-200 | 0.178 | 0.216 |
| Scoring head — Qwen3-8B | top-200 | 0.177 | **0.219** |

*Dev nDCG@20. TOP-1 = single best pick, rest left in pool order. TOP-20 = the
full learned ranking, which is what the leaderboard scores.*

The two objectives pick equally well (0.172–0.178), but the listwise scoring
head ranks positions 2–20 far better (0.219 vs 0.172). Ranking, not picking,
was where the headroom was.

---

## Pipeline

```
Conversation
  (history + current turn)
        |
        v
 [1] Hybrid retrieval  ........  BM25 + Dense (Qwen3-Emb-4B) + Item-CF + Artist expansion
        |                        fused by Reciprocal Rank Fusion, already-played excluded
        v
 Candidate pool (top-200)
        |
        v
 [2] LLM reranking  ..........  listwise scoring head (submitted)
        |                        vs generative single-pick (comparison arm)
        v
 Ranked list (top-20)
        |
        v
 [3] Response generation  ....  N=20 candidates (gemma-3n-E4B)
        |                        -> judge select (gemma-4-E2B) + clean & diversity filters
        v
 Track + response
```

Each candidate is rendered as one line of text, including two fields distilled
from the challenge's audio and lyrics embeddings:

```
12. Thunderstruck by AC/DC [rock, 80s] {sound: hard rock, metal | themes: rebellion, anger}
```

`sound:` is the consensus community tags of the track's nearest neighbours in
the **audio** embedding space; `themes:` its closest entries in a lyrical-topic
lexicon in the **lyrics** embedding space. Passing these as *words* rather than
injecting raw embedding vectors is what makes the multimodal signal transfer —
see [the ablations](docs/05_ablations_negative_results.md).

---

## Quickstart

### 1. Environment

```bash
git clone https://github.com/MaximeUM/recsys-challenge-2026
cd recsys-challenge-2026

bash setup_env.sh
source .venv/bin/activate
```

One environment covers everything: retrieval, reranker training, response
generation, vLLM inference and the explanation-quality metrics. Versions are
pinned in `requirements.txt` to the ones that produced the paper's numbers; the
default PyPI wheels are CUDA-enabled, so no custom index URL is needed.

Everything is run **from the repository root** (`python src/...`), which is
what the relative `data/` and `models/` paths assume.

### 2. Data and weights

```bash
# all seven challenge datasets into data/ (~875 MB)
python scripts/download_data.py

# or just what Blind-B inference needs (~867 MB), plus the official evaluator
python scripts/download_data.py --only inference --evaluator

# verify an existing download without re-fetching
python scripts/download_data.py --check
```

`download_data.py` pulls the official datasets from the `talkpl-ai` Hugging Face
organisation, verifies every expected parquet file afterwards, and is safe to
re-run (downloads resume and already-present files are skipped). See
[`data/README.md`](data/README.md) for the layout and what each split is used
for. Everything it writes is git-ignored.

```bash
# our fine-tuned weights + derived artifacts (~40 GB, into models/ and cache/)
python scripts/download_models.py

# only what Blind-B reads: retriever, scoring head, descriptors, Blind-B pool (~22 GB)
python scripts/download_models.py --only inference

# verify an existing download (add --checksums to hash every file)
python scripts/download_models.py --check --checksums
```

Every transfer is pinned to the revision recorded in
[`scripts/artifact_manifest.json`](scripts/artifact_manifest.json) and verified
against the SHA-256 values in the same file, so you get bit-identical inputs to
the published results rather than whatever the branch head happens to be.
`--no_pin` bypasses the pin; `--write_manifest` re-records it.

Base models are gated (Llama, Gemma): accept the licences on their model pages,
then `huggingface-cli login`. See [`models/README.md`](models/README.md) for what
each weight does and which submission it produced.

All released artifacts are grouped in one Hugging Face collection:
**[RecSys Challenge 2026 — FPMs_UMONS](https://huggingface.co/collections/MaximeM/recsys-challenge-2026-fpms-umons-6a636e612bd9187457225c86)**.
Everything can also be retrained from scratch with `scripts/reproduce_paper.sh`
— no released weight is required.

### 3. Stage 1 — inference (Blind-B -> predictions.json)

```bash
bash scripts/reproduce_blindB_submission.sh
```

Runs pool construction, scoring-head reranking, best-of-20 response generation,
and validates the output against the official scorer's format constraints.
~30–50 min end to end. The scoring-head step is the memory-hungry one; on cards
smaller than ~17 GB, run it with `DEVICE_MAP=auto`. Output:
`exp/inference/blindset_B/firstpos_scorehead_qwen_blindB_convbestofN.json`.


### 4. Stage 2 — training from scratch

```bash
bash scripts/reproduce_paper.sh retrieval     # dense encoder + pool + recall@k
bash scripts/reproduce_paper.sh descriptors   # audio neighbours -> content_desc.json
bash scripts/reproduce_paper.sh data          # raw reranker sets (top-50, top-200)
bash scripts/reproduce_paper.sh enrich        # {sound|themes} -> content_desc_full.json
bash scripts/reproduce_paper.sh firstpos      # generative reranker (table row 2)
bash scripts/reproduce_paper.sh scorehead     # the three scoring heads (rows 3-5)
bash scripts/reproduce_paper.sh eval          # dev nDCG@20 for every row of the table
```

Every stage declares its inputs and outputs, and fails early with a readable
message if an upstream artifact is missing rather than letting Transformers
reinterpret a local `models/...` path as a remote Hugging Face id. The full
artifact chain is documented in the script header and in
[`docs/02_reranking.md`](docs/02_reranking.md).

**Hardware requirements.** Training stages were run across several machines, so
rather than a machine list, here is what each stage actually needs:

| Stage | VRAM | Notes |
|---|---|---|
| dense retriever | ~10 GB per process | multi-GPU via `accelerate`; lower `ENC_BATCH_SIZE` if tight |
| scoring head (8B, 14336 tokens) | ~17 GB+ | set `DEVICE_MAP=auto` to shard across smaller cards |
| scoring head (3B) | ~8 GB | |
| response generation | ~10 GB | generator and judge loaded in two sequential phases |

Expect several hours per training stage.

**Smoke test.** Before committing to a full round, run the same chain on a mini
dataset:

```bash
bash scripts/smoke_test.sh              # every stage, mini data
bash scripts/smoke_test.sh retrieval    # or one stage
```

[`scripts/make_mini_dataset.py`](scripts/make_mini_dataset.py) samples whole
sessions out of `data/`, keeps every track those sessions reference plus random
distractors, and filters the track embeddings to match;
[`scripts/smoke_test.sh`](scripts/smoke_test.sh) then runs
`reproduce_paper.sh` against it inside a `.smoke/` sandbox, leaving your real
`data/`, `models/` and `exp/` untouched. Same base models, same context lengths,
same four-GPU sharding as the published run — only the data is small, and the
two top-200 scoring heads are capped at 64 records. It therefore requires the
same 4 GPUs and refuses to start on fewer. What it checks is that every stage
runs and feeds the next; the metrics it prints are noise by construction.

> **On exact reproducibility.** Retrieval pools are bit-reproducible. Reranker
> inference (bf16, greedy) is **not**: re-running changes ~40% of the picks
> where the model is undecided, moving dev nDCG@20 by ±0.008. Submission files
> will differ slightly from ours; the scores do not.

---

## Repository map

| Path | Contents |
|---|---|
| [`src/retrieval/`](docs/01_retrieval.md) | dense encoder training, four-channel RRF pool, recall analysis |
| [`src/reranking/`](docs/02_reranking.md) | content descriptors, training data, firstpos + scoring head, dev evaluation |
| [`src/response/`](docs/03_response_generation.md) | best-of-N generation, local judge, clean & diversity filters |
| [`src/explanation_metrics/`](docs/04_explanation_quality.md) | explanation-quality suite: reference-free, robustness and user-study metrics |
| [`src/ablations/`](docs/05_ablations_negative_results.md) | every negative result in the paper |
| `src/common/` | submission scoring against the composite, shared retriever text rendering |
| `scripts/` | end-to-end reproduction, submission validation, pinned artifact manifest |
| `tests/` | fast checks that need no data or GPU (`python tests/test_retriever_text_parity.py`) |

Each `docs/` page documents the exact commands, inputs, outputs, and the
numbers that section of the paper reports — including material cut from the
paper for space.

---

## Detailed results

### Retrieval

Adding the two behavioural channels (item-CF, artist expansion) to the standard
sparse+dense recipe:

| Pool | recall@50 | recall@500 |
|---|---|---|
| BM25 + dense only (RRF) | 44.3% | 70.4% |
| + item-CF + artist expansion | **46.7%** | **72.2%** |

`build_pool_dev.py` prints both rows side by side, with the **same** 4B ctx-1024
dense encoder on each — so the delta isolates the two behavioural channels.
recall@50 = 0.467 and recall@200 = 0.625 are exactly the GT-in-pool figures the
reranking section decomposes.

The context fix matters on its own: moving the dense encoder from a 256-token
right-truncated context to a **1024-token left-truncated** one is worth
**+0.015 dev nDCG@20**, because the final user request sits last in the text and
was being truncated away on long conversations. The gain is concentrated where
you would expect — nothing at cold start, +0.019 on turns 3–5.

### Reranking

Within single-pick training, emitting **one confident pick** beats generating
the full 20-position list: **0.1604 vs 0.1386** dev nDCG@20. The stage lifts the
retrieval pool from 0.1431 to **0.1681 (+17%)**.

Training the scoring head on a **deeper pool** helps the deep ranking:
top-50 -> top-200 moves TOP-20 from 0.202 to 0.216. Decomposing
`nDCG = P(GT in pool) x conditional ranking quality` explains why: recall rises
0.467 -> 0.625 while the conditional term falls 0.433 -> 0.345, because the
larger pool admits harder cases. Recall rises faster than it dilutes.

### Response generation

The composite sets judge score against lexical diversity, and we measured the
trade-off on the real judge (Blind-A):

| Configuration | LLM-judge | Lexical diversity |
|---|---|---|
| Distilled single-shot | 4.10 | 0.70 |
| Best-of-N | 3.65 | 0.78–0.80 |
| **Our submission** | **3.50** | **0.81** |
| Forced diversity | 3.25 | 0.835 |

Our 3.50 is a chosen operating point on that frontier: the composite weighs the
judge at 0.30 and lexical diversity at 0.10.

Isolating the two selection filters over the same 20 generations per turn
(re-judged locally):

| Selection | Local judge | Lexical div. | Distinct openings | Hallucinations |
|---|---|---|---|---|
| Best-judge only | 7.86 | 0.796 | 48/80 | 7 |
| + diversity | 7.54 | 0.818 | 75/80 | 7 |
| + clean | 7.72 | 0.791 | 48/80 | **0** |
| + clean + diversity | 7.47 | 0.815 | 75/80 | **0** |

The **clean** filter is nearly free (−0.14 judge) and removes every detected
hallucination; **diversity** costs −0.25 judge for +27 distinct openings.
Inspection of the 7 flagged cases: 6 were downstream symptoms of a reranker
mis-pick (the generator refuses to endorse the track), 1 was a false positive of
the checker. Response–track incoherence is a *ranking* symptom, not a generation
failure.

Conditioning **both** generation and judging on the full conversation is the
single largest response-quality lever: local judge 6.9 -> 7.97/10, and on the
server, judge 4.00 at composite 0.5295 (our best Blind-A submission).

### Explanation quality vs the dataset's own gold replies

Beyond the leaderboard's single judge score, we evaluate our explanations with
established metrics from the explainable-recommendation and CRS literature. We
regenerate responses for the full 8000-turn dev split and compare, paired per
turn, against the dataset's gold replies — themselves written by Gemini 2.5
Flash, the judge's own model family. 91 turns whose gold reply is the
placeholder `Unknown message` are excluded.

| Metric | Better | Ours | Gold |
|---|---|---|---|
| Length (words) | — | 43.9 | 45.1 |
| Distinct-1 | higher | **0.0721** | 0.0719 |
| Distinct-2 | higher | **0.346** | 0.317 |
| Unique responses (USR) | higher | **1.000** | **1.000** |
| Distinct openings | higher | **0.519** | 0.187 |
| Track / artist mention | higher | **0.957 / 0.893** | 0.944 / 0.871 |
| FMR (feature mention rate) | higher | **0.540** | 0.444 |
| FCR (feature coverage rate) | higher | **0.239** | 0.143 |
| Hallucination rate | lower | 0.001 | **0.000** |
| Self-BLEU | lower | **0.609** | 0.662 |
| Intra-session repetition | lower | **0.043** | 0.052 |
| GPT-2 perplexity | lower | 42.9 | **35.2** |

We match or beat the gold on every diversity and content metric — notably +22%
relative FMR and 2.8x the distinct openings. GPT-2 perplexity is the one metric
favouring the gold, and only after removing the placeholders (whose
`Unknown message` text, mean perplexity above 2000, otherwise inflates the gold
to 66.8).

On the 662 turns where our pick matches the ground truth, BLEU-4 more than
doubles (0.085 vs 0.036) and ROUGE-2 nearly does (0.141 vs 0.079), while
BERTScore barely moves (0.890 vs 0.875): the gap to the reference is *which
track was chosen*, not how the explanation is written. The full BLEU-1/2/4,
ROUGE-1/2/L and BERTScore breakdown is in
[`docs/04_explanation_quality.md`](docs/04_explanation_quality.md).

**What the proprietary judge rewards.** Across our seven blind-set variants with
known judge scores (2.85–4.10), the judge correlates with none of the metrics
above. The only sizeable Spearman correlations are with **length (+0.68)** and
**artist mention (+0.63)**; the correlation with Distinct-2 is negative (−0.45).
Directional only, n=7. MAUVE between our responses and the gold is **0.031** —
the two corpora are nearly separable, i.e. we do not imitate the Gemini house
style.

**User-study dimensions.** We administer the seven-item questionnaire from our
prior user study on recommendation explanations to a panel of open-weight LLM
judges standing in for human raters. Each response and gold reply is scored
independently under neutral labels, so the comparison is paired. *This table was
cut from the paper for space.*

| Dimension | gemma-4-E2B direct | gemma-4-E2B think | Qwen3-8B direct | Qwen3-8B think | Llama-3.2-3B |
|---|---|---|---|---|---|
| Transparency | **3.85** / 3.58 | **3.90** / 3.73 | **3.97** / 3.87 | **3.57** / 3.43 | **3.69** / 3.55 |
| Effectiveness | **4.11** / 3.89 | **4.79** / 4.51 | **4.46** / 4.36 | **4.35** / 4.01 | **4.72** / 4.45 |
| Persuasion | **4.25** / 3.97 | **4.61** / 4.37 | **4.93** / 4.82 | **4.76** / 4.53 | **4.54** / 4.23 |
| Trust | **3.96** / 3.73 | **4.09** / 3.85 | **4.60** / 4.30 | **4.26** / 4.03 | **4.76** / 4.32 |
| Satisfaction | **4.39** / 4.05 | **4.83** / 4.54 | **4.96** / 4.86 | **4.82** / 4.56 | **4.87** / 4.64 |

*Ours (bold) / gold. "think" = 3-seed average of the thinking mode (spread ≤ 0.01).*

The gold replies are genuinely good (dimension means 3.4–4.9), but ours score
higher on **every dimension under every judge** (+0.1 to +0.4 per dimension).
All 25 deltas favour our responses (paired Wilcoxon signed-rank, p < 10⁻⁷⁵).
Thinking mode de-saturates the compressed scales (Qwen3's deltas triple),
confirming ceiling effects rather than disagreement.

Protocol, per-judge tables and commands:
[`docs/04_explanation_quality.md`](docs/04_explanation_quality.md).

### Negative results

Reported in full in [`docs/05_ablations_negative_results.md`](docs/05_ablations_negative_results.md):

- **Full fine-tuning underperforms LoRA** (TOP-20 0.204 vs 0.216): it overfits
  within one epoch. Backbone scale is marginal too — Qwen3-8B beats
  Llama-3.2-3B by 0.003 despite 2.6x the parameters.
- **A cross-encoder wins on dev and fails on the blind set**: tied on dev, 13%
  *below* the generative reranker on Blind-A (0.386 vs 0.442). We select on
  dev-to-blind robustness, not dev score.
- **Scaling self-filtered distillation amplifies judge bias**: distilling on
  36,314 responses the local judge rated 10/10 scored **2.85** on the official
  judge — worse than an 800-example distillation (4.10) and below best-of-N (4.00).
- **LLM judges cannot do exact verification**: asked to flag responses citing
  absent tracks, gemma-4-E4B flagged 37/80 with **36 false positives**, while
  the deterministic clean filter found the single true violation with none.
  Letting the LLM rewrite flagged responses *degraded* the judge (7.72 -> 7.28).
- **Multimodal channels fail; distilled descriptors work**: on the dev turns
  explicitly targeting sound or lyrics, injecting an embedding as a raw
  retrieval signal *hurt* nDCG@20, while distilling it into the
  `sound:`/`themes:` descriptors beat the text-only pool.

  | Turn subset | Text pool | Extra RRF channel | Channel alone | Distilled descriptors |
  |---|---|---|---|---|
  | Sound (5940 turns) | 0.134 | 0.113 | 0.059 | **0.188** |
  | Lyrics (641 turns) | 0.158 | 0.134 | 0.072 | **0.222** |

- **Part of the remaining error is task under-determination**: where we fail,
  the ground truth often sits outside the top-500 of *every* channel. Many
  requests are open-ended, and the logged track need not even match the request
  — one dev user, delighted with Led Zeppelin's *Ramble On*, asks for other
  artists, yet the ground truth is the band's own *Heartbreaker*.

---

## Known gaps in this repository

We list these explicitly rather than leave them to be discovered. None of them
affects the correctness of the reported numbers, but each is a place where the
repository does not yet regenerate a result end to end.

- **Retriever training and inference do not render text identically.** The
  trainer leaves the catalog's title/artist/album fields as one-element arrays
  and labels history turns `assistant`, while the pool builders unpack those
  fields and label them `assistant_played`. Context length and truncation side
  do match. This is what produced the published retriever, so the numbers stand.
  `train_dense_encoder.py --text_format aligned` now trains on the pool
  builders' exact rendering instead, via the shared
  [`src/common/conv_render.py`](src/common/conv_render.py);
  `tests/test_retriever_text_parity.py` asserts the two are byte-identical. The
  default stays `published` so the released retriever remains reproducible.
- **Already-played tracks are excluded in the blind path only.**
  `build_pool_blind.py` filters them by catalog index and by a `name || artist`
  key, as described in the paper. `build_pool_dev.py` and
  `build_reranker_data.py` do not. The dev pools therefore carry distractors the
  submission pool does not, which makes the dev figures slightly conservative
  rather than optimistic. `build_pool_dev.py --exclude_played` applies the blind
  path's rule to dev; the default reproduces the published pool.
- **The artist channel is stricter than the statistic that motivates it.** The
  paper reports that the ground truth shares an artist with an already-played
  track in 46.0% of turns that have one — a non-empty intersection between the
  two tracks' credited-artist sets, reproducible with
  `src/retrieval/artist_overlap_stat.py`. The channel itself keys tracks by
  `str(artist_id)`, so a feature counts as a different artist from the soloist
  (40.9% under that reading). The published pools and the recall figures above
  were produced that way; `build_pool_dev.py --artist_match shared` applies the
  paper's rule instead. See [`docs/01_retrieval.md`](docs/01_retrieval.md).
- **`predict_firstpos.py` renders candidates without descriptors.** It serves the
  earlier, non-enriched firstpos recipe kept for reference. The published
  generative row uses `predict_firstpos_dev.py`, which does read
  `cache/content_desc_full.json`.

## Scope and limitations

- Ablations use the local dev split; Blind-B reported a higher nDCG@20 (0.3587)
  than our dev estimate (0.219) — a distributional difference.
- The local judge is a proxy on an undocumented scale; its deltas are directional.
- In the explanation-quality section, LLM judges replace human raters, and the
  gold replies are synthetic: "beats gold" means beating the authoring pipeline.
- The system is specific to English, TalkPlayData-style conversations.
- The challenge ships embeddings only, and we used **no external resources**.
  With raw audio, lyrics or cover images, different designs would open up.

## Acknowledgements

The present research benefited from computational resources made available on
Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by
the Walloon Region under the grant agreement n°1910247.

## License

Released under the MIT License (see `LICENSE`). The TalkPlayData-Challenge
datasets and the base models remain under their own respective licences.
