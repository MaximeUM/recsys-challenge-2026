# 4. Explanation quality beyond the judge score

The leaderboard summarises explanation quality as one number — 3.50/5 from a
proprietary, undocumented Gemini judge. The organizers disclose the judge as
Gemini-family but publish neither the exact model and version nor the scoring
prompt.

To situate that number, we evaluate our explanations with **established metrics
from the explainable-recommendation and CRS literature**, against a reference
the challenge itself provides: the dataset's **gold replies**, written by the
TalkPlayData 2 pipeline running Gemini 2.5 Flash — the judge's own model family.

## Protocol

- Responses are regenerated for the **full 8000-turn dev split** with the
  submitted pipeline.
- Every metric is **paired**: our response vs the same turn's gold reply.
- **91 turns** whose gold reply is the placeholder `Unknown message` are
  excluded, leaving **7909** turns. This matters: that placeholder's mean
  perplexity is above 2000 and otherwise inflates the gold's GPT-2 perplexity
  from 35.2 to 66.8.
- Generation and judging run under `VLLM_BATCH_INVARIANT=1` with seed 2026, so
  the whole suite is bit-reproducible. Thinking-mode judges are run with three
  seeds (2026 / 1337 / 7).

## Reference-free metrics

Every metric `reference_free_metrics.py` computes, not only those the paper
quotes:

| Metric | Better | Ours | Gold |
|---|---|---|---|
| Length (words) | — | 43.9 | 45.1 |
| Distinct-1 | higher | **0.0721** | 0.0719 |
| Distinct-2 (the challenge's own lexical diversity) | higher | **0.346** | 0.317 |
| Unique responses (USR) | higher | **1.000** | **1.000** |
| Distinct openings | higher | **0.519** | 0.187 |
| Track mention | higher | **0.957** | 0.944 |
| Artist mention | higher | **0.893** | 0.871 |
| FMR — feature mention rate | higher | **0.540** | 0.444 |
| FCR — feature coverage rate | higher | **0.239** | 0.143 |
| Hallucination rate | lower | 0.001 | **0.000** |

- **Distinct-n** is the share of distinct word n-grams.
- **USR** (unique responses) and **distinct openings** follow the NETE protocol
  and a distinct-n variant over response openings.
- **FMR** is the share of explanations mentioning at least one feature of the
  item they justify; features are adapted from PETER — here the track's top-5
  community tags, which were also available to the gold-authoring pipeline.
  **FCR** is the share of the feature vocabulary covered.
- Mention and hallucination rates reuse the deterministic checks from
  [response generation](03_response_generation.md).

We match or beat the gold on **every** diversity and content metric — notably
+22% relative FMR and 2.8x the distinct openings.

## Degeneration and robustness

| Metric | Better | Ours | Gold |
|---|---|---|---|
| Self-BLEU (high = mode collapse) | lower | **0.609** | 0.662 |
| Intra-session repetition | lower | **0.043** | 0.052 |
| GPT-2 perplexity (fluency) | lower | 42.9 | **35.2** |

Intra-session repetition is the bigram overlap between a session's eight
responses. Both degeneration measures favour us: the system repeats itself
*less* across a conversation than the dataset's own replies, and per-turn
quality is flat from turn 1 to turn 8.

GPT-2 perplexity is the one metric favouring the gold — a gap visible only after
removing the 91 placeholders.

## Reference-based metrics

BLEU counts word sequences shared with the gold reply, ROUGE is its
recall-oriented counterpart, and BERTScore compares meaning through contextual
embeddings. Standard CRS practice, with a known caveat: the reference talks
about the **gold** track, so a different — even valid — pick mechanically caps
surface overlap.

Every reference-based metric the script computes, split by whether our pick
matches the ground truth (662 of the 7909 turns):

| | All turns | Pick matches GT | Pick differs |
|---|---|---|---|
| BLEU-1 | 0.283 | **0.326** | 0.280 |
| BLEU-2 | 0.140 | **0.196** | 0.135 |
| BLEU-4 | 0.040 | **0.085** | 0.036 |
| ROUGE-1 | 0.324 | **0.376** | 0.319 |
| ROUGE-2 | 0.084 | **0.141** | 0.079 |
| ROUGE-L | 0.196 | **0.238** | 0.192 |
| BERTScore | 0.875 | 0.890 | 0.875 |

BLEU-4 more than doubles and ROUGE-2 nearly does, while BERTScore barely moves.
The gap to the reference is *which track was chosen*, not how the explanation is
written — mirroring the task under-determination discussed in the
[ablations](05_ablations_negative_results.md).

## What does the proprietary judge reward?

Across our seven blind-set variants with known judge scores (2.85–4.10), the
judge correlates with **none** of the metrics above. The only sizeable Spearman
correlations:

| Against | Spearman |
|---|---|
| Response length | **+0.68** |
| Artist mention | **+0.63** |
| Distinct-2 | **−0.45** |

Directional only (n=7). The underlying per-variant measurements are what
`reference_free_metrics.py` writes.

**MAUVE** between our responses and the gold replies is **0.031**: the two
corpora are nearly separable — we do not imitate the Gemini house style. Given
LLM evaluators' self-preference, this plausibly accounts for part of the 3.50 as
stylistic distance from the judge's own family. Not proof: the server judge
cannot be queried on the gold replies.

## User-study dimensions, via a judge panel

We administer the questionnaire from our prior user study on recommendation
explanations — seven Likert items (1–5) covering transparency, effectiveness,
persuasion, trust and satisfaction — to a panel of open-weight LLM judges
standing in for human raters. Each response and gold reply is scored
independently under neutral labels, making the comparison paired.

**This table was cut from the paper for space.** Ours (bold) / gold:

| Dimension | gemma-4-E2B direct | gemma-4-E2B think | Qwen3-8B direct | Qwen3-8B think | Llama-3.2-3B |
|---|---|---|---|---|---|
| Transparency | **3.85** / 3.58 | **3.90** / 3.73 | **3.97** / 3.87 | **3.57** / 3.43 | **3.69** / 3.55 |
| Effectiveness | **4.11** / 3.89 | **4.79** / 4.51 | **4.46** / 4.36 | **4.35** / 4.01 | **4.72** / 4.45 |
| Persuasion | **4.25** / 3.97 | **4.61** / 4.37 | **4.93** / 4.82 | **4.76** / 4.53 | **4.54** / 4.23 |
| Trust | **3.96** / 3.73 | **4.09** / 3.85 | **4.60** / 4.30 | **4.26** / 4.03 | **4.76** / 4.32 |
| Satisfaction | **4.39** / 4.05 | **4.83** / 4.54 | **4.96** / 4.86 | **4.82** / 4.56 | **4.87** / 4.64 |

*"think" columns are the 3-seed average of the thinking mode (spread ≤ 0.02).
Every cell is regenerated from the raw judge outputs by
`src/explanation_metrics/aggregate_userstudy_table.py`, which applies the
published protocol (7909 turns) and writes
`exp/userstudy_dimensions_fulldev.csv`.*

Findings:

- The gold replies are **genuinely good** (dimension means 3.4–4.9). This is not
  a weak baseline.
- Ours score higher on **every dimension under every judge**, a consistent +0.1
  to +0.4 margin.
- **All 25 deltas** favour our responses (paired Wilcoxon signed-rank,
  p < 10⁻⁷⁵), spanning +0.09 to +0.44. Not the quirk of one judge.
- Both sides face the same judge biases, so the **paired differences** are the
  robust quantity, not the absolute levels.
- **Thinking mode de-saturates compressed scales** (Qwen3's deltas triple),
  confirming ceiling effects rather than disagreement.

Judge reliability is tracked separately: across all nine judge runs
(8000 responses + 8000 gold each), **zero** final parse failures, with at most 5
retries in any run — recomputed by `judge_failure_breakdown.py`.

## Scripts

| Script | Role |
|---|---|
| `build_dev_responses.py` | samples dev sessions, reads the reranker picks, shards for parallel generation |
| `reference_free_metrics.py` | distinct-n, USR, openings, mention rates, FMR/FCR, BLEU/ROUGE/BERTScore, MAUVE |
| `robustness_metrics.py` | self-BLEU, intra-session repetition, GPT-2 perplexity, per-turn stability |
| `userstudy_dimensions.py` | the seven Likert items, administered by one judge (shardable, `--vllm`, `--thinking`) |
| `aggregate_userstudy_table.py` | applies the published protocol and regenerates the table above from raw judge outputs |
| `judge_failure_breakdown.py` | parse-failure and retry accounting across judges |
| `run_all_metrics.sh` | the sampled-subset chain |
| `run_fulldev_evals.sh` | **the full 8000-turn chain that produced the tables above** |

`reference_free_metrics.py` imports `compute_lexical_diversity` from the
official evaluator, so clone it at the repo root first (see
[`data/README.md`](../data/README.md)).

## Commands

**Prerequisite:** this chain reads the reranker picks
(`exp/picks/picks_shqwen8b_*.parquet`), it does not produce them. Run the dev
prediction step of [reranking](02_reranking.md) first — otherwise
`build_dev_responses.py` exits immediately.

**Hugging Face access:** some of the models pulled by the chain are gated —
public, but requiring their licence to be accepted first. Export `HF_TOKEN` (or
`hf auth login`) for an account that has access to all of them. Otherwise
generation fails.

```bash
# the whole full-dev suite (long: generation on 4 GPUs, then 9 judge runs)
bash src/explanation_metrics/run_fulldev_evals.sh

# or step by step
python src/explanation_metrics/build_dev_responses.py --n_sess 1000 --prefix shqwen8b_fulldev

# --drop_unknown is part of the published protocol: without it these two run on
# 8000 turns instead of 7909 and the numbers will not match *_nounknown.csv
python src/explanation_metrics/reference_free_metrics.py \
    --drop_unknown \
    --dev_resp exp/inference/devset/shqwen8b_fulldev_convbestofN.json \
    --out exp/explanation_metrics_fulldev_nounknown.csv

python src/explanation_metrics/robustness_metrics.py \
    --drop_unknown \
    --dev_resp exp/inference/devset/shqwen8b_fulldev_convbestofN.json

# one judge, 4-way sharded. VLLM_BATCH_INVARIANT=1 is what makes this
# bit-reproducible; run_fulldev_evals.sh exports it, a manual run must too.
export VLLM_BATCH_INVARIANT=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python src/explanation_metrics/userstudy_dimensions.py \
      --shard $i --nshards 4 --vllm --judge google/gemma-4-E2B-it \
      --tag fulldev_gemma4 --seed 2026 \
      --dev_resp exp/inference/devset/shqwen8b_fulldev_convbestofN.json &
done; wait

# regenerate the published table + its CSV (applies the 7909-turn protocol)
python src/explanation_metrics/aggregate_userstudy_table.py
```

`run_fulldev_evals.sh` produces **both** variants: the published one with
`--drop_unknown` (7909 turns, written to `*_nounknown.csv`) and an inclusive
diagnostic (8000 turns), under distinct filenames so neither can overwrite the
other. It asserts the published run contains exactly 7909 paired turns.
Use `--thinking` for the thinking-mode judge conditions.

## Why this matters

The suite applies verbatim to other teams' responses, enabling cross-team
comparison on explicit, published criteria rather than one opaque score. And
since every team tuned against the judge, judge scores partly reflect
optimization pressure — a property of any optimized-against metric, not a
deficiency of the judge.

Next: [5. Ablations and negative results](05_ablations_negative_results.md).
