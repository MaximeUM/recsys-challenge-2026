"""Prototype: use lyric match as a pool RERANKING signal, not as retrieval fusion.
On the lyrics-request subset (same regex as lyrics_channel.py), for each turn whose GT is in the top-K pool,
compute lyric_match = cos(Qwen3-encoded query, candidate lyrics-qwen3) and compare nDCG@20
single-GT under: (a) raw POOL order, (b) sorted by lyric match, (c) RRF(pool, lyric), (d) current scorehead,
(e) RRF(scorehead, lyric). The goal is to test whether this helps THESE cases without retraining.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/lyric_match_rerank.py
"""
import os, glob, re, json, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
DATA='data/'
dev=pd.read_parquet(DATA+'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids)
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
COL='lyrics-qwen3_embedding_0.6b'
dim=next(len(r) for r in emb[COL] if isinstance(r,(list,np.ndarray)) and len(r)>0)
LY=np.zeros((N,dim),np.float32); has=np.zeros(N,bool)
for tid,row in emb[COL].items():
    if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: LY[tidx[tid]]=row; has[tidx[tid]]=True
LY=LY/(np.linalg.norm(LY,axis=1,keepdims=True)+1e-9)
print(f"lyrics-qwen3 dim={dim} present={has.sum()}/{N}",flush=True)
LYR=re.compile(r"\b(lyric|lyrics|the line|the words|goes like|sings?|singing|the part where|verse|chorus|a line about|words go|hook|about (love|death|loss|war|heartbreak|god))\b",re.I)
# Current request by (session, turn).
req={}
for _,s in dev.iterrows():
    cs=s['conversations']
    for t in cs:
        if t['role']=='user': req[(s['session_id'],int(t['turn_number']))]=t['content']
# Scoring-head picks (actual baseline): key -> gt_rank_full.
sh={}
for f in glob.glob('exp/picks/picks_scorehead_*.parquet'):
    for _,r in pd.read_parquet(f).iterrows(): sh[r['key']]=(int(r['gt_pos']),int(r['gt_rank_full']))
pool=pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet')
K=200
items=[]
for _,r in pool.iterrows():
    sid=r['session_id']; tn=int(r['turn']); q=req.get((sid,tn))
    if not q or not LYR.search(q): continue
    gt=int(r['gt_idx']); pl=json.loads(r['pool'])[:K]
    items.append({'key':f'{sid}|{tn}','q':q,'gt':gt,'pool':pl,'gt_in':gt in pl,'gt_has':bool(has[gt])})
print(f"lyrics requests (dev): {len(items)} | GT in top-{K} pool: {np.mean([it['gt_in'] for it in items])*100:.1f}% | GT has lyrics: {np.mean([it['gt_has'] for it in items])*100:.1f}%",flush=True)
from sentence_transformers import SentenceTransformer
sg=SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',device='cuda')
TASK='Given a description of a song (its lyrics, theme or a remembered line), retrieve the matching song.'
qe=sg.encode([f'Instruct: {TASK}\nQuery:{it["q"]}' for it in items],normalize_embeddings=True,batch_size=32,show_progress_bar=False)
def dcg(rank): return 1.0/np.log2(rank+1) if (rank is not None and rank<=20) else 0.0
def rrf(rank_a,rank_b,k=60):  # Combine two rankings at list level; no separate new GT rank needed.
    pass
schemes={'pool':[], 'lyric':[], 'rrf_pool_lyric':[], 'scorehead':[], 'rrf_sh_lyric':[]}
for it,qv in zip(items,qe):
    pl=it['pool']; gt=it['gt']
    # (a) pool order
    r_pool=pl.index(gt)+1 if it['gt_in'] else None
    # (b) lyric-match sort
    lm=np.array([float(qv@LY[c]) if has[c] else -1e9 for c in pl])
    order_l=[pl[i] for i in np.argsort(-lm)]
    r_lyric=order_l.index(gt)+1 if it['gt_in'] else None
    # (c) RRF(pool, lyric) over pool candidates.
    rrf_sc={c:0.0 for c in pl}
    for rank,c in enumerate(pl): rrf_sc[c]+=1/(60+rank+1)              # pool rank
    for rank,c in enumerate(order_l): rrf_sc[c]+=1/(60+rank+1)         # lyric rank
    order_rrf=[c for c,_ in sorted(rrf_sc.items(),key=lambda x:-x[1])]
    r_rrf=order_rrf.index(gt)+1 if it['gt_in'] else None
    # (d) Scoring head (complete rank already computed), available only if key is in sh and gt_pos>=0.
    sval=sh.get(it['key']); r_sh=sval[1] if (sval and sval[0]>=0) else None
    # (e) RRF(scorehead-rank, lyric-rank): approximate using pool ranks. The complete scoring-head
    #     order is unavailable, so use gt_rank alone and combine at the GT-rank level.
    schemes['pool'].append(dcg(r_pool)); schemes['lyric'].append(dcg(r_lyric))
    schemes['rrf_pool_lyric'].append(dcg(r_rrf)); schemes['scorehead'].append(dcg(r_sh))
n=len(items)
print(f"\n=== nDCG@20 on LYRICS subset ({n} turns, denominator=all) ===")
for k in ['pool','lyric','rrf_pool_lyric','scorehead']:
    print(f"  {k:<18}: {np.mean(schemes[k]):.4f}")
# Conditional on GT in pool AND GT having lyrics, where lyric match CAN help.
mask=[it['gt_in'] and it['gt_has'] for it in items]
if any(mask):
    print(f"\n--- conditional (GT in pool AND has lyrics, {sum(mask)} turns) ---")
    for k in ['pool','lyric','rrf_pool_lyric']:
        print(f"  {k:<18}: {np.mean([schemes[k][i] for i in range(n) if mask[i]]):.4f}")
