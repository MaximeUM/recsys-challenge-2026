"""Literature-based explanation robustness and degeneration metrics.

1. Per turn (1..8, dev 400): does quality hold late in a session? (multi-turn
   degeneration and cross-turn repetition following Welleck et al., ICLR 2020)
2. Self-BLEU (Zhu et al., Texygen, SIGIR 2018): mode collapse - lower is more diverse.
3. MAUVE (Pillutla et al., NeurIPS 2021): distributional gap between generated and gold responses.
4. GPT-2-large perplexity (fluency; standard CRS practice in ReDial and UniCRS).
5. Distribution shift: the same grounding metrics on in-distribution dev versus cold-start Blind-B.

    python src/explanation_metrics/robustness_metrics.py            # GPU recommended for MAUVE/PPL
"""
import os, glob, json, re, warnings, argparse, random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd, torch
from pathlib import Path
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--dev_resp', default='exp/inference/devset/shqwen8b_respsample_convbestofN.json')
ap.add_argument('--max_selfbleu_refs', type=int, default=0,
                help='0 = EXACT self-BLEU (all references, multiprocessing); otherwise number of sampled references (seed 2026)')
ap.add_argument('--drop_unknown', action='store_true', default=False,
                help='exclude turns whose gold reply is "Unknown message" (dataset placeholder)')
a = ap.parse_args()

DATA = Path('data')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name', 'artist_name']:
    tm[c] = tm[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x).astype(str)
lk = tm.set_index('track_id')
norm = lambda s: re.sub(r'[\W_]+', ' ', str(s).lower(), flags=re.UNICODE).strip()  # Unicode-aware (Cyrillic, CJK, accents); treat _ as a separator.
dev = pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
sess = {s['session_id']: s for _, s in dev.iterrows()}

def top_tags(t, k=5):
    if t not in lk.index: return []
    tl = lk.loc[t, 'tag_list']
    return [str(x) for x in list(tl)[:k]] if isinstance(tl, (list, np.ndarray)) and len(tl) > 0 else []

def bigrams(s):
    w = norm(s).split(); return set(zip(w, w[1:]))

def distinct2(R):
    ng, tot = set(), 0
    for r in R:
        t = r.lower().split()
        for i in range(len(t)-1): ng.add((t[i], t[i+1])); tot += 1
    return len(ng)/max(1, tot)

gen = json.load(open(a.dev_resp))
gold = []
for e in gen:
    cs = sess[e['session_id']]['conversations']; t = int(e['turn_number'])
    mus = next(x['content'] for x in cs if x['role'] == 'music' and int(x['turn_number']) == t)
    g = next(x['content'] for x in cs if x['role'] == 'assistant' and int(x['turn_number']) == t)
    gold.append({'session_id': e['session_id'], 'turn_number': t, 'predicted_track_ids': [mus], 'predicted_response': g})

if a.drop_unknown:
    keep = [str(gld['predicted_response']).strip() != 'Unknown message' for gld in gold]
    n0 = len(gen)
    gen  = [e for e, k in zip(gen, keep) if k]
    gold = [gld for gld, k in zip(gold, keep) if k]
    print(f'[drop_unknown] removed {n0 - len(gen)} "Unknown message" turns -> {len(gen)} paired turns', flush=True)

# 1. Per-turn metrics (1..8): generated versus gold.
def perturn(entries, label):
    rows = []
    for t in range(1, 9):
        E = [e for e in entries if int(e['turn_number']) == t]
        R = [e['predicted_response'] for e in E]
        m_tr = np.mean([norm(lk.loc[e['predicted_track_ids'][0], 'track_name']) in norm(e['predicted_response']) for e in E])
        fmr = np.mean([any(f' {norm(tg)} ' in f" {norm(e['predicted_response'])} " for tg in top_tags(e['predicted_track_ids'][0]) if len(norm(tg)) >= 3) for e in E])
        rows.append({'turn': t, 'n': len(E), 'words': np.mean([len(r.split()) for r in R]),
                     'distinct2': distinct2(R), 'mention_track': m_tr, 'FMR': fmr})
    df = pd.DataFrame(rows)
    print(f'\n=== Per turn - {label} ===')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return df

pt_gen = perturn(gen, 'dev generated (final pipeline)')
pt_gold = perturn(gold, 'dev gold')

# Cross-turn repetition within a session (mean bigram Jaccard over turn pairs).
def intra_session_rep(entries):
    bysess = {}
    for e in entries: bysess.setdefault(e['session_id'], []).append(e)
    sims = []
    for es in bysess.values():
        B = [bigrams(e['predicted_response']) for e in es]
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                u = B[i] | B[j]
                if u: sims.append(len(B[i] & B[j])/len(u))
    return float(np.mean(sims))

print(f"\nCross-turn repetition (within-session bigram Jaccard; lower is less repetitive)")
print(f"  generated: {intra_session_rep(gen):.4f} | gold: {intra_session_rep(gold):.4f}")

# -- 2. Self-BLEU (mode collapse) on each set --------------------------------------
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import multiprocessing as mp
sm = SmoothingFunction().method1
_TOKS = None
def _sb_one(i):
    refs = _TOKS[:i] + _TOKS[i+1:]
    K = a.max_selfbleu_refs
    if K and len(refs) > K:
        refs = random.Random(2026 + i).sample(refs, K)
    return sentence_bleu(refs, _TOKS[i], weights=(.25,)*4, smoothing_function=sm)
def self_bleu(R):
    """Exact by default using all references, parallelized across CPU cores."""
    global _TOKS
    _TOKS = [norm(r).split() for r in R]
    with mp.Pool(min(32, os.cpu_count() or 8)) as p:
        vals = p.map(_sb_one, range(len(_TOKS)), chunksize=16)
    return float(np.mean(vals))

SETS = [
    ('Blind-B SUBMITTED (judge 3.50)', 'exp/inference/blindset_B/firstpos_scorehead_qwen_blindB_convbestofN.json'),
    ('blindA best-of-N v2 (4.00)', 'exp/inference/blindset_A/firstpos_top50_blindA_convbestofN.json'),
    ('Blind-A distilled 800ex (4.10)', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_sftresp.json'),
    ('Blind-A distilled RFT 36k (2.85)', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_rftresp.json'),
]
print('\n=== Self-BLEU (SIGIR 2018; LOWER = more diverse, high means mode collapse) ===')
for label, p in SETS:
    if not os.path.exists(p):
        print(f'!! missing: {p}'); continue
    R = [e['predicted_response'] for e in json.load(open(p))]
    print(f'  {label:34s} {self_bleu(R):.4f} (n={len(R)})')
Rgen = [e['predicted_response'] for e in gen]; Rgold = [e['predicted_response'] for e in gold]
print(f'  {"dev generated final pipeline":34s} {self_bleu(Rgen):.4f} (n={len(Rgen)})')
print(f'  {"dev gold (same turns)":34s} {self_bleu(Rgold):.4f} (n={len(Rgold)})')

# 3. MAUVE generated versus gold (dev).
import mauve
mv = mauve.compute_mauve(p_text=Rgold, q_text=Rgen, featurize_model_name='gpt2-large',
                         device_id=0 if torch.cuda.is_available() else -1, verbose=False, seed=2026)
print(f'\n=== MAUVE (NeurIPS 2021) generated vs gold: {mv.mauve:.3f} (1.0 = indistinguishable distributions)')

# 4. GPT-2-large perplexity.
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
dev_dv = 'cuda:0' if torch.cuda.is_available() else 'cpu'
gt = GPT2TokenizerFast.from_pretrained('gpt2-large')
gm = GPT2LMHeadModel.from_pretrained('gpt2-large', torch_dtype=torch.float32).to(dev_dv).eval()
@torch.no_grad()
def ppl(R):
    vals = []
    for r in R:
        ids = gt(r, return_tensors='pt', truncation=False).input_ids.to(dev_dv)
        if ids.shape[1] < 2: continue
        vals.append(float(torch.exp(gm(ids, labels=ids).loss)))
    return float(np.mean(vals))
print('\n=== GPT-2-large perplexity (fluency; lower is more fluent) ===')
print(f'  dev generated: {ppl(Rgen):.1f} | dev gold: {ppl(Rgold):.1f}')
for label, p in SETS:
    if not os.path.exists(p):
        print(f'!! missing: {p}'); continue
    R = [e['predicted_response'] for e in json.load(open(p))]
    print(f'  {label:34s} {ppl(R):.1f}')
