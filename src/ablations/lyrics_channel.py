"""Dev prototype: intent-conditioned LYRICS channel.
lyrics-qwen3_embedding_0.6b contains Qwen3-Embedding-0.6B lyrics embeddings (known model, guaranteed alignment).
Detect lyrics/line/words requests and encode the query with Qwen3-Embedding-0.6B ->
cosine similarity against the lyrics embeddings -> GT recall. Compare with the current text pool + RRF.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/lyrics_channel.py
"""
import os, json, re, warnings, glob
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
print(f"lyrics-qwen3: dim={dim}, present={has.sum()}/{N}", flush=True)
LYR=re.compile(r"\b(lyric|lyrics|the line|the words|goes like|sings?|singing|the part where|verse|chorus|a line about|words go|hook)\b", re.I)
items=[]
for _,s in dev.iterrows():
    cs=s['conversations']; gtbt={int(t['turn_number']):t['content'] for t in cs if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx: continue
        lu=[t['content'] for t in cs if t['role']=='user' and int(t['turn_number'])==tn]
        if lu and LYR.search(lu[0]): items.append({'q':lu[0],'gt':tidx[gt],'sid':s['session_id'],'tn':tn})
print(f"lyrics requests detected on dev: {len(items)}", flush=True)
if not items: raise SystemExit
from sentence_transformers import SentenceTransformer
sg=SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',device='cuda')
# Qwen3-Embedding is ASYMMETRIC: the query needs an instruction; the lyrics document does not.
TASK='Given a description of a song (its lyrics, theme or a remembered line), retrieve the matching song.'
def q_instruct(q): return f'Instruct: {TASK}\nQuery:{q}'
qe=sg.encode([q_instruct(it['q']) for it in items],normalize_embeddings=True,batch_size=32,show_progress_bar=False)
print(f"qwen3-0.6b query dim={qe.shape[1]} (must equal {dim})", flush=True)
LYt=torch.tensor(LY,device='cuda'); QE=torch.tensor(qe,device='cuda',dtype=torch.float32)
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
def rank_in(lst,gt): return lst.index(gt)+1 if gt in lst else 10**9
sig=[]; txt=[]; rrf=[]
for i,it in enumerate(items):
    sc=(QE[i]@LYt.T).cpu().numpy(); ly=list(np.argsort(-sc)[:500]); sig.append(rank_in(ly,it['gt']))
    tl=pool.get(f"{it['sid']}|{it['tn']}",[]); txt.append(rank_in(tl,it['gt']))
    rr=np.zeros(N)
    for r,idx in enumerate(ly): rr[idx]+=1/(60+r+1)
    for r,idx in enumerate(tl): rr[idx]+=1/(60+r+1)
    rrf.append(rank_in(list(np.argsort(-rr)[:500]),it['gt']))
rec=lambda rk,k:100*np.mean([r<=k for r in rk])
print(f"\n=== recall on {len(items)} LYRICS requests ===")
print(f"  {'channel':<22}{'R@20':>7}{'R@50':>7}{'R@500':>7}")
for lab,rk in [('text (current pool)',txt),('Qwen3 lyrics',sig),('RRF text+lyrics',rrf)]:
    print(f"  {lab:<22}{rec(rk,20):>7.1f}{rec(rk,50):>7.1f}{rec(rk,500):>7.1f}")
