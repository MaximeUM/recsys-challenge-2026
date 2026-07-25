# 5. Ablations and negative results

Everything in this page is a thing we tried that did **not** work, or worked
only under a condition worth naming. It is a substantial part of the paper's
contribution, and all of it is reproducible from `src/ablations/`.

---

## 1. Full fine-tuning underperforms LoRA

Fully fine-tuning the scoring head **degrades** TOP-20:

| Training | Dev TOP-20 (top-200 pool) |
|---|---|
| LoRA rank 16 | **0.216** |
| Full fine-tuning | 0.204 |

Full fine-tuning overfits within a single epoch, while LoRA keeps generalising.

Backbone scale is similarly marginal: Qwen3-8B beats Llama-3.2-3B by only
**0.003** TOP-20 despite ~2.6x the parameters (8.2B vs 3.2B). Neither more
capacity nor more freedom is the bottleneck.

**Script:** `train_scorehead_fullft.py` (same data and monitoring as the LoRA
trainer, full-parameter updates).

---

## 2. A cross-encoder wins on dev and fails on the blind set

As a third option beside firstpos and the scoring head, we tried a
**cross-encoder**: a fully fine-tuned bge-reranker-v2-m3 scoring each
conversation–candidate pair *jointly*, unlike the retrieval dual-encoder which
encodes them separately.

| Evaluation | Cross-encoder | Generative reranker |
|---|---|---|
| Full dev | tied | tied |
| 200-sample dev | +7% | — |
| **Blind-A (held out)** | **0.386** | **0.442** |

It scored **13% below** the generative reranker on the blind set. It had fit
regularities shared by train and dev but absent from the blind distribution
(cold users, truncated conversations) — the same overfitting signature as full
vs LoRA above.

**Lesson:** select on **dev-to-blind robustness**, not on dev score. This was
the single most useful methodological correction of the project, and it is why
we kept the scoring head (which did transfer: 0.45 > 0.43 on Blind-B) rather
than trusting a dev win.

**Scripts:** `crossencoder_train.py`, `crossencoder_predict.py`,
`crossencoder_blind.py`.

---

## 3. Scaling self-filtered distillation amplifies judge bias

Best-of-N is expensive at inference. The obvious remedy is rejection
fine-tuning: distil a single-shot generator on generations the local judge rates
10/10.

We generated at scale (32 shards), kept 44,602 responses rated ≥8 by the local
judge, of which **36,314 were rated 10/10**, and fine-tuned a Llama-3.2-3B
student on them.

| Generator | Official judge |
|---|---|
| Distilled on **800** examples | **4.10** |
| Online best-of-N | 4.00 |
| Distilled on **36,314** examples | **2.85** |

More data made it **worse**. The local judge is lenient (81% of responses rated
10/10), so it selected a style the official judge dislikes, and scaling the
auto-filtered data amplified that bias rather than averaging it out — volume
without the solution diversity that makes rejection fine-tuning scale.

**Lesson:** an online signal beat offline distillation here, and auto-filtering
with a weak judge amplifies the weak judge.

**Script:** `rft_distillation_generate.py`.

---

## 4. LLM judges are good selectors and bad verifiers

Judge-based **selection** works (that is the whole response stage). Judge-based
**verification** fails.

Prompted to flag responses citing tracks absent from the conversation:

| Detector | Flagged | False positives | True violations found |
|---|---|---|---|
| gemma-4-E2B | 38% of picks | over-flags | — |
| gemma-4-E4B (bigger, full conversation) | 37/80 | **36** | 1 |
| **Deterministic string match** | 1/80 | **0** | **1** |

The bigger model was **worse**. It flagged even the recommended track itself as
an "invented memory". And letting the LLM rewrite the flagged responses
*degraded* the judge score from **7.72 to 7.28** — it was rewriting correct
responses.

**Lesson:** exact membership and grounding checks belong to deterministic code.
An LLM reasons by surface form; asking it to detect a hallucination introduces
its own.

**Scripts:** `hallucination_detector_llm.py`, `hallucination_detector_e4b.py`.

---

## 5. Multimodal channels fail; distilled descriptors work

This is the most thoroughly tested negative result. The challenge ships
per-track embeddings (audio CLAP 512, image SigLIP2 768, lyrics Qwen3 1024,
CF-BPR 128) with ~99% catalog coverage. We tested them as retrieval signals in
five ways, and as reranking signals in two.

### As retrieval (all negative)

**a. Extra RRF channel** — nearest neighbours of the history mean. Unique
contribution ≈ 0. *Script:* `multimodal_rrf_channel.py`

**b. Two-tower projection** query -> concatenated multimodal space:

| Variant | Dev nDCG@20 |
|---|---|
| Frozen MLP head | 0.047 |
| Fully trained 4B query encoder | 0.0986 |
| Text dense retriever | **0.159** |

Giving the projection full capacity doubled the score and still left it 38%
below text. The frozen multimodal space is simply inferior for this signal,
whatever the encoder capacity. *Scripts:* `twotower_frozen_proj.py`,
`twotower_trained_proj.py`

**c. Intent routing** — route each query to the channel of its modality, and
measure on the target subset, which is the most favourable possible test:

| Query type | Specialised channel R@500 | Text R@500 |
|---|---|---|
| Cover art (301 queries) | 8.3% | **59.1%** |
| Lyrics (635 queries) | 18.4% | **72.3%** |
| Sound / mood (1135 queries) | 38% | **71%** |

The channels are genuinely aligned (8–18x random; SigLIP v1 gives 1.0%, i.e.
chance, confirming the SigLIP2 family is the right one) but they are crushed by
text **on their own target subset**. *Scripts:* `cover_channel.py`,
`cover_channel_siglipv1.py`, `lyrics_channel.py`, `audio_channel.py`.
`cover_channel_querytransform.py` tests one further variant: an LLM rewrites the
request as a short visual caption in SigLIP's native form before matching. It
does not close the gap either.

**d. Alignment trained by us** rather than off-the-shelf — an MLP mapping our
fine-tuned query encoder into each modality space, trained contrastively:

| Modality | Off-the-shelf | Aligned by us | Text |
|---|---|---|---|
| Cover art | 8.3% | 34.2% (x4) | **59.1%** |
| Lyrics | 18.4% | 27.7% | **72.3%** |

Alignment was indeed the missing piece in the off-the-shelf tests, but the
aligned channel still sits below text, and RRF fusion **degrades** (53.8 < 59.1;
68.3 < 72.3): the channels are not complementary. *Script:*
`aligned_modality_channel.py`

**e. Pure-intent routing** — an LLM classifies cover-art queries as "purely
visual" (216) vs "mixed" (85). Even on the pure ones, text wins (54.6 vs 30.6).
A "pure cover art" request is not context-free: the conversation (tracks played,
co-occurrence, artist) still carries the signal that localises the ground truth.
*Script:* `pure_intent_routing.py`

### As a query-dependent reranking score (also negative)

Distinct from the above: a per-candidate, per-query score
`cos(query in modality space, candidate modality embedding)`, used to re-rank the
top-200 pool, measured on the target subset.

| Turn subset | Text pool | Match alone | RRF fusion | Text scoring head |
|---|---|---|---|---|
| Lyrics (641 turns) | 0.158 | 0.072 | 0.134 | **0.222** |
| Sound (5940 turns) | 0.134 | 0.059 | 0.113 | **0.188** |

Query-dependent matching is worse than the raw pool order *and* worse than the
text reranker, even on its own target subset. The cause is an asymmetry: the
query is a short description while the embedding encodes the entire content —
plus under-determination (a thousand tracks are "about loss").

A surprise: lyrics queries are not even a weak spot for the text reranker
(0.222 > 0.202 overall), and sound queries are barely below (0.188).

*Scripts:* `lyric_match_rerank.py`, `sound_match_rerank.py`

### What actually works: distillation into text

Enriching each candidate with descriptors **derived** from the embeddings —
`sound` = consensus tags of the audio-space nearest neighbours, `themes` =
lyrical topics — and retraining the reranker on the enriched candidates:

| Turn subset | Text pool | Extra RRF channel | Channel alone | **Distilled descriptors** |
|---|---|---|---|---|
| Sound | 0.134 | 0.113 | 0.059 | **0.188** |
| Lyrics | 0.158 | 0.134 | 0.072 | **0.222** |

Overall dev effect on the single-pick reranker: 0.1681 -> 0.1718 (+2.2%), and a
vote between the enriched and original rerankers reaches 0.1747, so they are
complementary.

**The multimodal signal transfers as text, not as raw vectors** — and only in
the reranker, not in retrieval.

**Cover art remains unusable.** `image-siglip2` is a pure vision feature with no
paired text encoder in that space, and we do not have the raw images (or URLs)
to caption with a VLM. Retrieving covers via ISRC or album id from an external
database would be out-of-rules.

*Script:* `build_content_descriptors.py` (in `src/reranking/`).

---

## 6. Part of the remaining error is task under-determination

Where we fail, the ground truth is often outside the top-500 of **every**
channel — text and audio alike. Not a model fault: the single logged ground
truth is a weak label, in two distinct ways.

**Open-ended requests.** "Suggest something else that's popular" is satisfied by
thousands of tracks, but the log keeps only the one played next. A
valid-but-different pick scores zero.

**The logged track need not match the request.** One dev user, delighted with
Led Zeppelin's *Ramble On*, explicitly asks for **other** artists — and the
ground truth is the band's own *Heartbreaker*. Another asks to "play the 1990
remaster of *Ramble On*", a precise metadata request, and the ground truth is
*When The Levee Breaks*: simply what the assistant played next.

Either way, single-ground-truth nDCG under-credits plausible recommendations and
adds noise near the ceiling. This also means every multimodal measurement above
**understates** channel value: a channel that retrieves a valid but different
track is scored as a miss.

---

## 7. Retriever fallback when the pick looks wrong

Idea: detect a bad reranker pick and fall back to the retriever's top-1.

| Detector | Firing rate | Effect |
|---|---|---|
| Explanation does not name the track | 14/1500 | wash (helps 1, hurts 1) |
| gemma-4 "is this appropriate?" | 566/1500 | **degrades** (0.1595 vs 0.1703) |

The reranker beats the retriever even on the hard cases. The genuinely odd picks
— multi-turn requests containing a negation ("I do *not* want X") — are rare,
and the retriever is no better on them.

---

## Cross-cutting lessons

1. **Dev does not predict blind.** The 200-sample over-promises, full dev
   over-promises less but still over-promises. Robustness to distribution shift
   beats a small dev gain. The cross-encoder is the flagrant counter-example;
   the RRF ensemble is a milder one (dev 0.2260 > 0.2186, blind 0.44 < 0.45).
2. **nDCG is won in ranking, not recall.** Enriching retrieval raises the number
   of *reachable* ground truths without separating them better at the top.
3. **LLMs do fuzzy semantic judgement, not exact lookup.** Pick the tool to fit
   the task.
4. **Auto-filtering with a weak judge amplifies its bias.**
5. **The judge/diversity tension in the composite is real**: optimizing one
   costs the other.
6. **bf16 greedy decoding is not bit-reproducible.** The reranker's picks vary
   ~40% run to run, moving nDCG@20 by ±0.008. Worth knowing before chasing a
   0.005 "improvement".

## Limitations of the multimodal conclusion

We concluded "multimodal is negative", but took shortcuts that can fairly be
contested:

- The alignment MLP was trained on a **frozen** query encoder, not a full
  fine-tune. Our aligned numbers (34% cover art) are a **floor**.
- The projection was trained **generally**, not specialised per intent.
- The modality was tested as a retrieval/RRF channel and as a query-dependent
  rerank score, but **never as a learned feature of the reranker**, where the
  model could learn *when* to trust audio or image depending on intent. That is
  the strongest remaining formulation, and it is untested.
- Single-ground-truth evaluation penalises a channel that retrieves a valid but
  different track.

The defensible claim is therefore: **multimodal does not beat text+context in
the configurations we tested**, and the one credible remaining angle is
modality-as-reranker-feature with a full fine-tune.
