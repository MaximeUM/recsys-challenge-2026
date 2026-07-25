"""single-pick SFT data over the 4B pool.

Changelog 2026-06-28: add `session_id` and `turn_number` columns. session_id supports a train/val
split BY SESSION (leakage prevention: all turns from one conversation stay in the same set), while
turn_number supports turn-balance checks. Existing top-50 artifacts are unchanged; regenerate them
to add these fields. Used for the top-200 build.

Diff vs the top-50 build:
- dense channel = 4B retriever (models/qwen3_ft_dualencoder_4b_ctx1024)
- target = THE SINGLE best candidate: target = JSON [GT_position].
  Motivation: a single GT per turn => nDCG@20 = 1/log2(GT_rank+1), so only the GT rank
  matters. Train the model to output the best candidate (loss on this single pick),
  not to rank 20 positions where 19 merely reproduce noisy pool order.
- --n_rag_top: size of the candidate context (50 or 200).

Run (detached, GPU 1):
    setsid nohup python src/reranking/build_reranker_data.py --n_rag_top 50 \\
        > logs/build_sft_4b_top50.log 2>&1 < /dev/null &
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('SFT_GPU', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys, json, random, warnings, argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import bm25s
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--n_rag_top', type=int, default=50)
ap.add_argument('--out', default=None)
args = ap.parse_args()

DATA          = Path('data')
MODEL_FT_DIR  = Path('models/qwen3_ft_dualencoder_4b_ctx1024')
CACHE_DIR     = Path('models/_sft_dataset_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
N_RAG_TOP     = args.n_rag_top
OUT           = Path(args.out) if args.out else CACHE_DIR / f'sft_ctx1024_60000_top{N_RAG_TOP}_firstpos.parquet'
N_SAMPLES     = 60000
RRF_K         = 60
SEED          = 42
DEVICE        = 'cuda'
TASK_QWEN     = 'Given a music chat conversation, retrieve the track the user wants to listen to next'

def format_user_profile(up):
    if up is None: return ''
    return ', '.join(f'{k}={up.get(k)}' for k in
                     ['age_group','country_name','gender','preferred_language','preferred_musical_culture'] if up.get(k))
def format_goal(g):
    if g is None: return ''
    return ', '.join(f'{k}={g.get(k)}' for k in ['category','specificity','listener_goal'] if g.get(k))

print(f'n_rag_top={N_RAG_TOP} -> {OUT}', flush=True)
print('Loading data...', flush=True)
train      = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
track_meta = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for col in ['track_name', 'artist_name', 'album_name']:
    track_meta[col] = track_meta[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x).astype(str)
track_ids   = track_meta['track_id'].tolist()
track_index = {tid: i for i, tid in enumerate(track_ids)}
track_lookup = track_meta.set_index('track_id')
artist_id   = track_meta['artist_id'].astype(str).values
N = len(track_ids)
artist_to_tracks = defaultdict(list)
for i, a in enumerate(artist_id): artist_to_tracks[a].append(i)

def track_id_to_short(tid):
    if tid not in track_lookup.index: return tid
    r = track_lookup.loc[tid]; return f"{r['track_name']} - {r['artist_name']}"
def track_text_compact(row):
    parts = [f"{row['track_name']} by {row['artist_name']}"]
    if isinstance(row['tag_list'], (list, np.ndarray)) and len(row['tag_list']) > 0:
        parts.append(f"[{', '.join(list(row['tag_list'])[:5])}]")
    return ' '.join(parts)
def build_conv_text(convs, target_turn):
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

print('Building co-occurrence...', flush=True)
cooc = defaultdict(Counter)
session_tracks = []
for _, s in train.iterrows():
    seq = [track_index[t['content']] for t in s['conversations']
           if t['role'] == 'music' and t['content'] in track_index]
    u = set(seq); session_tracks.append(u)
    for a in u:
        for b in u:
            if a != b: cooc[a][b] += 1

print('Building train items...', flush=True)
items = []
for sidx, s in train.iterrows():
    convs = s['conversations']
    gtbt = {t['turn_number']: t['content'] for t in convs if t['role'] == 'music'}
    for tn in range(1, 9):
        gt = gtbt.get(tn)
        if gt is None or gt not in track_index: continue
        items.append((sidx, s, tn, track_index[gt]))
rng = random.Random(SEED)
sampled = rng.sample(items, k=min(N_SAMPLES, len(items)))
print(f'{len(items):,} items, sample {len(sampled):,}', flush=True)

print('BM25 index...', flush=True)
corpus = []
for _, row in track_meta.iterrows():
    parts = [f"track_name: {row['track_name']}", f"artist_name: {row['artist_name']}", f"album_name: {row['album_name']}"]
    if isinstance(row['tag_list'], (list, np.ndarray)): parts.append(f"tags: {', '.join(row['tag_list'])}")
    corpus.append('\n'.join(parts))
retr = bm25s.BM25(); retr.index(bm25s.tokenize(corpus, show_progress=False))

print('Dense encode catalog (4B)...', flush=True)
ft = SentenceTransformer(str(MODEL_FT_DIR), device=DEVICE, model_kwargs={'torch_dtype': torch.bfloat16})
ft.max_seq_length = 1024; ft.tokenizer.truncation_side='left'; ft.eval()
ft_track = ft.encode([dense_text(r) for _, r in track_meta.iterrows()], batch_size=16, show_progress_bar=False,
                     normalize_embeddings=True, device=DEVICE, convert_to_tensor=True).to(torch.bfloat16)

raw_q = [build_conv_text(s['conversations'], tn) for _, s, tn, _ in sampled]
print('BM25 retrieve...', flush=True)
bm = retr.retrieve(bm25s.tokenize([q.lower() for q in raw_q], show_progress=False), k=200, return_as='tuple')
bm25_top = [np.asarray(bm.documents[i], dtype=np.int64) for i in range(len(sampled))]
print('Dense encode queries...', flush=True)
q_emb = ft.encode([f'Instruct: {TASK_QWEN}\nQuery: {q.lower()}' for q in raw_q], batch_size=16,
                  show_progress_bar=False, normalize_embeddings=True, device=DEVICE, convert_to_tensor=True).to(torch.bfloat16)
dense_top = []
for s in range(0, len(sampled), 256):
    sc = (q_emb[s:min(s+256,len(sampled))] @ ft_track.T).float()
    dense_top.extend(torch.topk(sc, 200, dim=1).indices.cpu().numpy())

def rrf_topk(channels, k):
    sc = defaultdict(float)
    for lst in channels:
        for r, idx in enumerate(lst): sc[int(idx)] += 1.0 / (RRF_K + r + 1)
    return [idx for idx, _ in sorted(sc.items(), key=lambda x: -x[1])][:k]

print('Fusing combined pool + formatting (LOO co-occ, target=single best)...', flush=True)
records = []; kept = dropped = 0
for i, (sidx, session, tn, gt_idx) in enumerate(sampled):
    hist = [track_index[t['content']] for t in session['conversations']
            if t['role'] == 'music' and t['turn_number'] < tn and t['content'] in track_index]
    chans = [bm25_top[i], dense_top[i]]
    if hist:
        agg = Counter()
        for h in hist:
            if h in cooc: agg.update(cooc[h])
        sess = session_tracks[sidx]
        for c in list(agg.keys()):
            if c in sess: agg[c] -= len(hist)
        for h in hist: agg.pop(h, None)
        co = [idx for idx, v in agg.most_common() if v > 0][:200]
        if co: chans.append(np.array(co, dtype=np.int64))
        art = sorted({t for h in hist for t in artist_to_tracks.get(artist_id[h], [])})
        if art:
            art = np.array(art, dtype=np.int64)
            asc = (q_emb[i] @ ft_track[art].T).float().cpu().numpy()
            chans.append(art[np.argsort(-asc)][:200])
    cand_list = rrf_topk(chans, N_RAG_TOP)
    if gt_idx not in cand_list:
        dropped += 1; continue
    kept += 1
    gt_pos = cand_list.index(gt_idx)
    cand_lines = [f'{k}. {track_text_compact(track_meta.iloc[idx])}' for k, idx in enumerate(cand_list, 1)]
    records.append({
        'session_id':        int(sidx),                   # Train/val split BY SESSION (leakage prevention).
        'turn_number':       int(tn),                     # Check turn balance.
        'user_profile':      format_user_profile(session.get('user_profile')),
        'conversation_goal': format_goal(session.get('conversation_goal')),
        'conversation':      build_conv_text(session['conversations'], tn),
        'candidates':        '\n'.join(cand_lines),
        'target':            json.dumps([gt_pos + 1]),   # SINGLE best pick (loss-on-first)
    })
    if (i + 1) % 10000 == 0:
        print(f'  {i+1}/{len(sampled)} | kept={kept} dropped={dropped}', flush=True)

print(f'Kept {kept:,}/{kept+dropped:,} ({100*kept/(kept+dropped):.1f}% GT in top-{N_RAG_TOP})', flush=True)
pd.DataFrame(records).to_parquet(OUT)
print(f'Saved -> {OUT}', flush=True)
