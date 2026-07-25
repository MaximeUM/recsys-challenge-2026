"""Literature-based explanation metrics computed on generated responses.

Reference-free (for each Blind-A/B and dev response set):
  - Distinct-1/2 (Li et al., NAACL 2016), using the official challenge D-2 implementation.
  - USR (Unique Sentence Ratio) and distinct openings (Li et al., CIKM 2020 / ACL 2021).
  - Recommended track/artist mention (structural grounding; see the clean filter).
  - Adapted FMR/FCR (PETER, ACL 2021): features are the recommended track's top five community tags.
  - Hallucinations: quoted titles absent from the conversation and recommendation.
Reference-based (dev only, against the gold assistant response):
  - BLEU-1/2/4 (sentence, smoothing1), ROUGE-1/2/L F1, BERTScore F1 (roberta-large).
  - Split pick==GT / pick!=GT (the reference discusses the gold track, so an incorrect pick
    mechanically limits overlap).

    python src/explanation_metrics/reference_free_metrics.py [--dev_resp exp/inference/devset/shqwen8b_respsample_convbestofN.json]
"""
import os, argparse, glob, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'music-crs-evaluator')
from metrics import compute_lexical_diversity

ap = argparse.ArgumentParser()
ap.add_argument('--dev_resp', default='exp/inference/devset/shqwen8b_respsample_convbestofN.json')
ap.add_argument('--out', default='exp/explanation_metrics.csv')
ap.add_argument('--bertscore', action='store_true', default=True)
ap.add_argument('--no_bertscore', dest='bertscore', action='store_false')
ap.add_argument('--drop_unknown', action='store_true', default=False,
                help='exclude turns whose gold reply is "Unknown message" (dataset placeholder)')
a = ap.parse_args()

DATA = Path('data')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name', 'artist_name']:
    tm[c] = tm[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x).astype(str)
lk = tm.set_index('track_id')
norm = lambda s: re.sub(r'[\W_]+', ' ', str(s).lower(), flags=re.UNICODE).strip()  # Unicode-aware (Cyrillic, CJK, accents); treat _ as a separator.

PARQ = {
    'A': DATA/'TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet',
    'B': DATA/'TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet',
    'dev': DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet',
}
SESS = {k: {s['session_id']: s for _, s in pd.read_parquet(p).iterrows()} for k, p in PARQ.items()}

def nm(t):
    return f"{lk.loc[t,'track_name']} by {lk.loc[t,'artist_name']}" if t in lk.index else str(t)

def top_tags(t, k=5):
    if t not in lk.index: return []
    tl = lk.loc[t, 'tag_list']
    return [str(x) for x in list(tl)[:k]] if isinstance(tl, (list, np.ndarray)) and len(tl) > 0 else []

QUOTE = re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”')

def conv_ctx_norm(split, e):
    """Normalized conversation text through the target turn, plus the recommended track."""
    cs = SESS[split][e['session_id']]['conversations']; tt = int(e['turn_number'])
    P = []
    for t in cs:
        if int(t['turn_number']) > tt: break
        P.append(nm(t['content']) if t['role'] == 'music' else str(t['content']))
    P.append(nm(e['predicted_track_ids'][0]))
    return norm(' || '.join(P))

def hallucinated(resp, ctx):
    for m in QUOTE.finditer(resp):
        q = norm(m.group(1) or m.group(2))
        if len(q) < 3: continue
        if not (q in ctx or any(w in ctx for w in q.split() if len(w) >= 5)):
            return True
    return False

def word_in(needle, hay):
    return f' {needle} ' in f' {hay} '

def opening(c): return ' '.join(norm(c).split()[:3])

def reffree(entries, split):
    """entries: list of dicts {session_id, turn_number, predicted_track_ids, predicted_response}."""
    R = [e['predicted_response'] for e in entries]
    n = len(R)
    row = {'n': n,
           'len_words': float(np.mean([len(r.split()) for r in R])),
           'distinct1': compute_lexical_diversity(R, n=1),
           'distinct2': compute_lexical_diversity(R, n=2),
           'USR': len(set(R)) / n,
           'openings': len(set(opening(r) for r in R)) / n}
    m_track = m_artist = m_fmr = m_hall = 0
    matched_feats, all_feats = set(), set()
    for e in entries:
        top = e['predicted_track_ids'][0]; r = norm(e['predicted_response'])
        tn, ar = norm(lk.loc[top, 'track_name']), norm(lk.loc[top, 'artist_name'])
        if tn and (tn in r or (len(tn.split()) >= 3 and ' '.join(tn.split()[:3]) in r)): m_track += 1
        if ar and ar in r: m_artist += 1
        feats = [norm(t) for t in top_tags(top) if len(norm(t)) >= 3]
        all_feats.update(feats)
        hit = [f for f in feats if word_in(f, r)]
        matched_feats.update(hit)
        if hit: m_fmr += 1
        if hallucinated(e['predicted_response'], conv_ctx_norm(split, e)): m_hall += 1
    row.update({'mention_track': m_track / n, 'mention_artist': m_artist / n,
                'FMR': m_fmr / n, 'FCR': len(matched_feats) / max(1, len(all_feats)),
                'halluc': m_hall / n})
    return row

# Response sets: label, path, split, and actual Gemini judge score when known.
SETS = [
    ('blindB scorehead-qwen best-of-20+clean+div (SUBMITTED)', 'exp/inference/blindset_B/firstpos_scorehead_qwen_blindB_convbestofN.json', 'B', 3.50),
    ('blindA best-of-N v2 (composite 0.5295)', 'exp/inference/blindset_A/firstpos_top50_blindA_convbestofN.json', 'A', 4.00),
    ('blindA best-of-N v2 ctx1024', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_convbestofN.json', 'A', 3.65),
    ('Blind-A distilled Llama 800ex (single-shot)', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_sftresp.json', 'A', 4.10),
    ('Blind-A distilled RFT 36k', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_rftresp.json', 'A', 2.85),
    ('blindA gemma-4-E4B best-of-N', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_gemma4resp.json', 'A', 3.65),
    ('Blind-A gemma-4-E4B diversified (temp 1.0)', 'exp/inference/blindset_A/firstpos_top50_ctx1024_blindA_gemma4resp_div.json', 'A', 3.25),
]

rows = []
for label, path, split, judge in SETS:
    if not os.path.exists(path):
        print(f'!! missing: {path}'); continue
    entries = json.load(open(path))
    r = {'set': label, 'real_judge': judge}; r.update(reffree(entries, split))
    rows.append(r); print(f'ok {label} (n={r["n"]})')

# Dev: generated final-pipeline responses + gold on the same turns.
def gold_entries(keys):
    out = []
    for sid, t in keys:
        cs = SESS['dev'][sid]['conversations']
        mus = next(x['content'] for x in cs if x['role'] == 'music' and int(x['turn_number']) == t)
        gold = next(x['content'] for x in cs if x['role'] == 'assistant' and int(x['turn_number']) == t)
        out.append({'session_id': sid, 'turn_number': t, 'predicted_track_ids': [mus], 'predicted_response': gold})
    return out

dev_gen = []
for p in sorted(glob.glob(a.dev_resp)) or []:
    dev_gen += json.load(open(p))
if not dev_gen:
    shards = sorted(glob.glob('exp/inference/devset/shqwen8b_respsample_shard*_convbestofN.json'))
    for p in shards: dev_gen += json.load(open(p))

if dev_gen:
    keys = [(e['session_id'], int(e['turn_number'])) for e in dev_gen]
    gold = gold_entries(keys)
    gold_by_key = {(g['session_id'], g['turn_number']): g for g in gold}

    if a.drop_unknown:
        n0 = len(dev_gen)
        dev_gen = [e for e in dev_gen
                   if gold_by_key[(e['session_id'], int(e['turn_number']))]['predicted_response'].strip() != 'Unknown message']
        keys = [(e['session_id'], int(e['turn_number'])) for e in dev_gen]
        gold = gold_entries(keys)
        gold_by_key = {(g['session_id'], g['turn_number']): g for g in gold}
        print(f'[drop_unknown] removed {n0 - len(dev_gen)} "Unknown message" turns -> {len(dev_gen)} paired turns')

    r = {'set': f'dev generated by final pipeline (n={len(dev_gen)})', 'real_judge': None}
    r.update(reffree(dev_gen, 'dev')); rows.append(r)
    r = {'set': 'dev GOLD (same turns)', 'real_judge': None}
    r.update(reffree(gold, 'dev')); rows.append(r)

    # Reference-based metrics.
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    sm = SmoothingFunction().method1
    rs = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    picks = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('exp/picks/picks_shqwen8b_*.parquet'))]).set_index('key')

    recs = []
    for e in dev_gen:
        k = (e['session_id'], int(e['turn_number']))
        ref, hyp = gold_by_key[k]['predicted_response'], e['predicted_response']
        rt, ht = norm(ref).split(), norm(hyp).split()
        sc = rs.score(ref, hyp)
        pk = picks.loc[f"{e['session_id']}|{e['turn_number']}"]
        recs.append({
            'correct': bool(pk['pick'] == pk['gt_pos']),
            'bleu1': sentence_bleu([rt], ht, weights=(1,), smoothing_function=sm),
            'bleu2': sentence_bleu([rt], ht, weights=(.5, .5), smoothing_function=sm),
            'bleu4': sentence_bleu([rt], ht, weights=(.25,)*4, smoothing_function=sm),
            'rouge1': sc['rouge1'].fmeasure, 'rouge2': sc['rouge2'].fmeasure, 'rougeL': sc['rougeL'].fmeasure,
            'ref': ref, 'hyp': hyp})
    rdf = pd.DataFrame(recs)

    if a.bertscore:
        from bert_score import score as bscore
        _, _, F = bscore(rdf['hyp'].tolist(), rdf['ref'].tolist(), lang='en',
                         model_type='roberta-large', device='cpu', batch_size=32, verbose=False)
        rdf['bertscore'] = F.numpy()

    cols = [c for c in ['bleu1', 'bleu2', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'bertscore'] if c in rdf]
    print('\n=== Reference-based metrics (dev, versus gold assistant response) ===')
    print('all turns       :', {c: round(rdf[c].mean(), 4) for c in cols}, f"(n={len(rdf)})")
    for v, lab in [(True, 'pick == GT     '), (False, 'pick != GT     ')]:
        s = rdf[rdf['correct'] == v]
        if len(s): print(f'{lab}:', {c: round(s[c].mean(), 4) for c in cols}, f'(n={len(s)})')
    rdf.drop(columns=['ref', 'hyp']).to_csv(a.out.replace('.csv', '_ref_per_turn.csv'), index=False)
else:
    print('\n(no generated dev responses yet; skipping reference-based metrics)')

df = pd.DataFrame(rows)
print('\n=== Reference-free metrics ===')
print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
df.to_csv(a.out, index=False)
print(f'\nSaved -> {a.out}')

# Correlation between metrics and actual Gemini judge scores on known blind sets.
j = df[df['real_judge'].notna()]
if len(j) >= 4:
    from scipy.stats import spearmanr
    print('\n=== Spearman correlation with actual Gemini judge score (n=%d sets) ===' % len(j))
    for c in ['distinct1', 'distinct2', 'USR', 'openings', 'mention_track', 'mention_artist', 'FMR', 'FCR', 'len_words']:
        rho, p = spearmanr(j['real_judge'], j[c])
        print(f'  {c:16s} rho={rho:+.2f} (p={p:.2f})')
