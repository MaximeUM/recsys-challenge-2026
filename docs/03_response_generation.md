# 3. Response generation

Every turn also requires a natural-language response introducing the
recommendation. It is scored by two of the composite's four components: the
**LLM judge** (0.30) and **lexical diversity** (0.10).

## Method: conversation-grounded best-of-N

For each turn we generate **N=20** candidate responses with **gemma-3n-E4B**,
then select one with a local judge (**gemma-4-E2B**). Both the generator and the
judge are conditioned on the **full conversation**, not just the current turn.

That last point is the single largest response-quality lever we found:

| Conditioning | Local judge |
|---|---|
| Current turn only | 6.9 / 10 |
| **Full conversation** | **7.97 / 10** |

On the server judge this configuration scored **4.00**, giving our best Blind-A
composite of **0.5295**.

## Two selection filters

Best-of-N with a pure best-judge criterion converges on a monotone style (37/80
distinct openings, the tic "Okay, I understand..." appearing 16 times), which
costs lexical diversity. Two filters run on top of the judge's ranking:

- **clean** — the response must name the recommended track and must not cite any
  track absent from the conversation. A deterministic string-match check, not an
  LLM.
- **diversity-aware** — among the clean candidates, prefer one whose opening has
  not been used elsewhere.

Isolated over the same 20 generations per turn, re-judged locally:

| Selection | Local judge | Lexical div. | Distinct openings | Hallucinations |
|---|---|---|---|---|
| Best-judge only | 7.86 | 0.796 | 48/80 | 7 |
| + diversity | 7.54 | 0.818 | 75/80 | 7 |
| + clean | 7.72 | 0.791 | 48/80 | **0** |
| **+ clean + diversity** | **7.47** | **0.815** | **75/80** | **0** |

The **clean** filter is nearly free: −0.14 judge, and it removes every detected
hallucination. **Diversity** is the expensive one: −0.25 judge for +27 distinct
openings (+0.024 lexical diversity).

Selecting a clean candidate among N is far better than trying to constrain or
rewrite the generation — the LLM rewriting pass *degraded* the judge from 7.72
to 7.28 (see [ablations](05_ablations_negative_results.md)).

## Residual hallucinations are a ranking symptom

Inspecting the 7 flagged responses: **6 of 7** were genuine mismatches caused by
a **bad reranker pick** (multi-turn requests containing a negation). The
generator refuses to endorse the track it was handed and recommends something
else or apologises. The 7th was a false positive of the checker (a track title
containing parentheses).

Response–track incoherence is therefore a downstream symptom of the *ranking*
stage, not a generation failure.

## The judge / diversity trade-off

The composite pits the judge against lexical diversity, and they trade off
consistently on the real judge (Blind-A):

| Configuration | LLM-judge | Lexical diversity |
|---|---|---|
| Distilled single-shot | 4.10 | 0.70 |
| Best-of-N | 3.65 | 0.78–0.80 |
| **Our submission** | **3.50** | **0.81** |
| Forced diversity | 3.25 | 0.835 |

Our operating point follows from the composite's weights (judge 0.30 vs lexical
0.10). The trade-off measures at roughly **−0.27 local judge per +0.06 lexical
diversity**.

## A note on model choice

The organizers disclose the judge as Gemini-family. Generating with the judge's
own family is permitted and, given LLM evaluators' self-preference, would likely
have scored higher. We deliberately ran only small **open-weight** models: it is
reproducible (pinned weights, no silent API updates), auditable, cheap at scale,
and deployable without sending user conversations to a third party.

## Scripts

| Script | Role |
|---|---|
| `convbestofn_judge.py` | best-of-N generation + local judge selection (the Blind-A configuration, N=6) |
| `convbestof20_clean.py` | N=20 with the **clean** filter |
| `convbestof20_diverse.py` | N=20 with **clean + diversity** — the submitted configuration |
| `convbestof20_vllm.py` | vLLM port of the above (same prompts, same filters), for full-dev scale |
| `reselect_compare.py` | the 2x2 filter ablation table above, re-selecting over cached generations |

All of them take the reranker's output JSON (track ids, empty responses) and
fill in `predicted_response`. Track ids are never modified.

## Commands

```bash
# submitted configuration (Blind-B): N=20, clean + diversity
python src/response/convbestof20_diverse.py \
    --input  exp/inference/blindset_B/firstpos_scorehead_qwen_blindB.json \
    --blind_parquet data/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet \
    --out    exp/inference/blindset_B/firstpos_scorehead_qwen_blindB_convbestofN.json \
    --n 20

# Blind-A configuration: N=6
python src/response/convbestofn_judge.py \
    --input exp/inference/blindset_A/firstpos_top50_ctx1024_blindA.json \
    --gen google/gemma-3n-E4B-it --n 6

# reproduce the 2x2 selection table
python src/response/reselect_compare.py
```

Generation takes ~15–25 min for 80 turns on one GPU (the script loads the
generator and the judge in two sequential phases to fit in memory). Defaults:
`--gen google/gemma-3n-E4B-it`, `--judge google/gemma-4-E2B-it`.

For the full 8000-turn dev split, use the vLLM port `convbestof20_vllm.py`,
which is deterministic under `VLLM_BATCH_INVARIANT=1` with a fixed seed.

## Known edge cases

- **Safety refusals.** gemma-3n occasionally refuses to write about explicitly
  tagged tracks, and all 20 candidates come back as refusals. This happened once
  in 80 turns on the submitted run.
- **Titles with parentheses**, and artist first names shorter than four
  characters, can defeat the mention checker's regex. These are false positives
  of the *checker*, not real hallucinations.

The explanation-quality evaluation suite behind the paper's "Beyond the Judge
Score" section is documented in
[4. Explanation quality](04_explanation_quality.md).
