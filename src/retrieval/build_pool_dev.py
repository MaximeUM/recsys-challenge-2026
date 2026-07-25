"""Part A - combined RRF pool (BM25 + dense FT + co-occ + ranked artist).

Validate the recall@K gain on dev (8,000 items) against the current RRF
(BM25+dense), then save the dev top-500 pool for later SFT and reranker evaluation.

Channels:
  - BM25 (ranked top 500)
  - Dense FT v1 (ranked top 500)
  - Session-KNN co-occurrence: candidates co-listened with the history, ranked by count
  - Ranked artist: tracks by previously played artists, ranked by dense similarity to the query
Fusion: RRF over the ranks from each available channel.

Already-played tracks: build_pool_blind.py drops them from the submitted pool
(by catalog index and by a `name || artist` key), this script does not. That is
what produced the published dev pool, so it stays the default; the dev pools
therefore carry distractors the submission pool does not, which makes the dev
figures slightly conservative rather than optimistic. Pass --exclude_played to
build a dev pool under the blind path's exact rule instead.

Run (detached):
    setsid nohup python src/retrieval/build_pool_dev.py \\
        > logs/combined_pool.log 2>&1 < /dev/null &
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GPU', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys, warnings, json, argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import bm25s
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

DATA         = Path('data')
MODEL_FT_DIR = Path('models/qwen3_ft_dualencoder_4b_ctx1024')
DEVICE       = 'cuda'
TASK_QWEN    = 'Given a music chat conversation, retrieve the track the user wants to listen to next'
K_EVAL       = [20, 50, 100, 200, 500]
N_CHAN       = 500
RRF_K        = 60
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--exclude_played', action='store_true',
                 help="drop already-played tracks from the pool, exactly as "
                      "build_pool_blind.py does (off by default: the published "
                      "dev pool keeps them)")
_ap.add_argument('--artist_match', choices=['array', 'shared'], default='array',
                 help="how the artist channel decides two tracks share an artist. "
                      "'array' (default): exact equality of the stringified "
                      "artist_id array - what produced the published pools and "
                      "the published recall figures. 'shared': any credited "
                      "artist in common, which is the definition the paper's "
                      "motivating statistic uses (see artist_overlap_stat.py)")
_ap.add_argument('--out', default='exp/combined_pool_ctx1024_dev.parquet')
_args = _ap.parse_args()
EXCLUDE_PLAYED = _args.exclude_played
# Extra depth to retrieve before filtering, so the pool still reaches its target
# size once played tracks are removed. Mirrors build_pool_blind.py.
FILTER_MARGIN = 200

OUT_POOL     = Path(_args.out)
OUT_POOL.parent.mkdir(parents=True, exist_ok=True)

print('Loading data...', flush=True)
train      = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
dev        = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
track_meta = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for col in ['track_name', 'artist_name', 'album_name']:
    track_meta[col] = track_meta[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x).astype(str)
track_ids   = track_meta['track_id'].tolist()
track_index = {tid: i for i, tid in enumerate(track_ids)}
track_lookup = track_meta.set_index('track_id')
ARTIST_MATCH = _args.artist_match
# A track credits a LIST of artists (11.2% of the catalog credits more than one),
# and artist_name / artist_id are not even the same length on those rows. 'shared'
# keys each track by every credited artist, so a feature matches the featured
# artist's own catalog; 'array' keys it by the whole array rendered as a string,
# so "['A' 'B']" and "['A']" are treated as different artists entirely.
if ARTIST_MATCH == 'shared':
    artist_keys = [list(x) if isinstance(x, (list, np.ndarray)) else [x]
                   for x in track_meta['artist_id']]
else:
    artist_keys = [[str(x)] for x in track_meta['artist_id'].astype(str).values]
# `name || artist` duplicate key, same definition as build_pool_blind.py
track_key = (track_meta['track_name'].str.lower().str.strip() + ' || '
             + track_meta['artist_name'].str.lower().str.strip()).values
N = len(track_ids)
print(f'tracks={N}', flush=True)

artist_to_tracks = defaultdict(list)
for i, keys in enumerate(artist_keys):
    for a in keys:
        artist_to_tracks[a].append(i)
print(f'artist channel: match={ARTIST_MATCH}, {len(artist_to_tracks)} distinct artist keys', flush=True)

def track_id_to_short(tid):
    if tid not in track_lookup.index: return tid
    r = track_lookup.loc[tid]; return f"{r['track_name']} - {r['artist_name']}"

def conv_text(convs, target_turn):
    lines = []
    for t in convs:
        if t['turn_number'] >= target_turn: break
        role, content = t['role'], t['content']
        if role == 'music': role, content = 'assistant_played', track_id_to_short(content)
        lines.append(f'{role}: {content}')
    for t in convs:
        if t['turn_number'] == target_turn and t['role'] == 'user':
            lines.append(f"user (REQUEST): {t['content']}"); break
    return '\n'.join(lines)

def dense_text(row):
    parts = [f"track_name: {row['track_name']}", f"artist_name: {row['artist_name']}", f"album_name: {row['album_name']}"]
    if isinstance(row['tag_list'], (list, np.ndarray)) and len(row['tag_list']) > 0:
        parts.append(f"tags: {', '.join(row['tag_list'])}")
    return '\n'.join(parts)

# -- Co-occurrence (from the train split) -------------------------------------
print('Building co-occurrence from train...', flush=True)
cooc = defaultdict(Counter)
for _, s in train.iterrows():
    seq = [track_index[t['content']] for t in s['conversations']
           if t['role'] == 'music' and t['content'] in track_index]
    u = set(seq)
    for a in u:
        for b in u:
            if a != b: cooc[a][b] += 1

def cooc_rank(history, topn=N_CHAN):
    agg = Counter()
    for h in history:
        if h in cooc: agg.update(cooc[h])
    for h in history: agg.pop(h, None)
    return [i for i, _ in agg.most_common(topn)]

# ── Dev items ─────────────────────────────────────────────────────────────────
dev_items = []
for _, s in dev.iterrows():
    convs = s['conversations']
    gtbt = {t['turn_number']: t['content'] for t in convs if t['role'] == 'music'}
    for tn in range(1, 9):
        gt = gtbt.get(tn)
        if gt is None or gt not in track_index: continue
        hist = [track_index[t['content']] for t in convs
                if t['role'] == 'music' and t['turn_number'] < tn and t['content'] in track_index]
        played_artists = {a for h in hist for a in artist_keys[h]}
        dev_items.append({'session_id': s['session_id'], 'turn': tn, 'q': conv_text(convs, tn),
                          'gt': track_index[gt], 'hist': hist, 'played_artists': played_artists})
Q = len(dev_items)
print(f'{Q} dev items', flush=True)

# ── BM25 + Dense ──────────────────────────────────────────────────────────────
print('Building BM25...', flush=True)
corpus = []
for _, row in track_meta.iterrows():
    parts = [f"track_name: {row['track_name']}", f"artist_name: {row['artist_name']}", f"album_name: {row['album_name']}"]
    if isinstance(row['tag_list'], (list, np.ndarray)): parts.append(f"tags: {', '.join(row['tag_list'])}")
    corpus.append('\n'.join(parts))
retr = bm25s.BM25(); retr.index(bm25s.tokenize(corpus, show_progress=False))

print('Dense FT encode...', flush=True)
ft = SentenceTransformer(str(MODEL_FT_DIR), device=DEVICE, model_kwargs={'torch_dtype': torch.bfloat16})
ft.max_seq_length = 1024; ft.tokenizer.truncation_side='left'; ft.eval()
ft_emb = ft.encode([dense_text(r) for _, r in track_meta.iterrows()], batch_size=16, show_progress_bar=False,
                   normalize_embeddings=True, device=DEVICE, convert_to_tensor=True).to(torch.bfloat16)
queries = [it['q'] for it in dev_items]
bm = retr.retrieve(bm25s.tokenize([q.lower() for q in queries], show_progress=False), k=N_CHAN, return_as='tuple')
bm25_top = [np.asarray(bm.documents[i], dtype=np.int64) for i in range(Q)]
q_emb = ft.encode([f'Instruct: {TASK_QWEN}\nQuery: {q.lower()}' for q in queries], batch_size=16,
                  show_progress_bar=False, normalize_embeddings=True, device=DEVICE, convert_to_tensor=True).to(torch.bfloat16)
dense_top = []
for s in range(0, Q, 256):
    sc = (q_emb[s:min(s+256,Q)] @ ft_emb.T).float()
    dense_top.extend(torch.topk(sc, N_CHAN, dim=1).indices.cpu().numpy())

def rrf(channels, k=max(K_EVAL)):
    sc = np.zeros(N, dtype=np.float32)
    for lst in channels:
        for r, idx in enumerate(lst): sc[idx] += 1.0 / (RRF_K + r + 1)
    return np.argsort(-sc)[:k]

def drop_played(order, hist):
    """build_pool_blind.py's filter: by catalog index and by `name || artist`."""
    phist = set(hist); pkeys = {track_key[h] for h in hist}
    return np.array([c for c in order
                     if c not in phist and track_key[c] not in pkeys],
                    dtype=np.int64)

def recall(order_per_q):
    return {k: float(np.mean([dev_items[i]['gt'] in order_per_q[i][:k] for i in range(Q)])) for k in K_EVAL}

print('Building channels + fusing...', flush=True)
base_order, comb_order, comb_pool = [], [], []
for i, it in enumerate(dev_items):
    bm_l, dn_l = bm25_top[i], dense_top[i]
    base_order.append(rrf([bm_l, dn_l]))
    chans = [bm_l, dn_l]
    if it['hist']:
        co = cooc_rank(it['hist'])
        if co: chans.append(np.array(co, dtype=np.int64))
        # Artist channel ranked by dense similarity.
        art = sorted({t for a in it['played_artists'] for t in artist_to_tracks.get(a, [])})
        if art:
            art = np.array(art, dtype=np.int64)
            asc = (q_emb[i] @ ft_emb[art].T).float().cpu().numpy()
            chans.append(art[np.argsort(-asc)])
    if EXCLUDE_PLAYED and it['hist']:
        full = drop_played(rrf(chans, max(K_EVAL) + FILTER_MARGIN), it['hist'])[:max(K_EVAL)]
    else:
        full = rrf(chans)
    comb_order.append(full)
    comb_pool.append(full[:N_CHAN])

mb, mc = recall(base_order), recall(comb_order)
print('\n' + '=' * 60)
print(f'Recall@K (n={Q})')
print(f'played tracks: {"excluded (blind-path rule)" if EXCLUDE_PLAYED else "kept (published dev pool)"}')
print(f'{"K":>6} {"RRF(BM25+dense)":>16} {"+ co-occ + artist":>20}')
for k in K_EVAL:
    print(f'{k:>6} {mb[k]*100:>15.1f}% {mc[k]*100:>19.1f}%')
print('=' * 60)

# Save the combined dev pool.
pd.DataFrame({
    'session_id': [it['session_id'] for it in dev_items],
    'turn': [it['turn'] for it in dev_items],
    'gt_idx': [it['gt'] for it in dev_items],
    'pool': [json.dumps([int(x) for x in p]) for p in comb_pool],
}).to_parquet(OUT_POOL)
print(f'Combined dev pool saved -> {OUT_POOL}', flush=True)
