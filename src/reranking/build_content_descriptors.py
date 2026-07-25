"""Build CONTENT descriptors derived from embeddings without cross-space alignment:
- SOUND: consensus tags from nearest AUDIO neighbors (laion_clap)
- THEME: consensus tags from nearest LYRICS neighbors (Qwen3)
Content-to-content similarity is valid within audio/lyrics spaces; consensus neighbor tags provide
a clean descriptor even when the track's own tags are sparse. Save cache/content_desc.json and examples.

    CUDA_VISIBLE_DEVICES=0 python src/reranking/build_content_descriptors.py
"""
import os, json, re, warnings, glob
from collections import Counter
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
DATA='data/'
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids)
# Filtered tags per track.
JUNK=set('favorite favorites favourites seen live love loved beautiful awesome fucking fuck great good best amazing my essentials classic classics favoritesongs spotify fip somafm kdwb radio playlist owned vinyl albums i own want check out wishlist'.split())
def cleantags(tl):
    if not isinstance(tl,(list,np.ndarray)): return []
    out=[]
    for t in tl:
        t=str(t).strip().lower()
        if not t or t in JUNK or len(t)<3 or any(ch.isdigit() for ch in t): continue
        out.append(t)
    return out
TAGS=[cleantags(tm.iloc[i]['tag_list']) for i in range(N)]
# Tag document frequency; retain discriminative genres and moods.
df=Counter()
for ts in TAGS:
    for t in set(ts): df[t]+=1
KEEP={t for t,c in df.items() if 30<=c<=15000}
TAGS=[[t for t in ts if t in KEEP] for ts in TAGS]
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
def mat(col):
    dim=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0); M=np.zeros((N,dim),np.float32); ok=np.zeros(N,bool)
    for tid,row in emb[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: M[tidx[tid]]=row; ok[tidx[tid]]=True
    return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9), ok
AUD,okA=mat('audio-laion_clap'); LYR,okL=mat('lyrics-qwen3_embedding_0.6b')
art=tm['artist_id'].astype(str).values
def nn_consensus(M, ok, K=25):
    Mt=torch.tensor(M,device='cuda'); desc=[[] for _ in range(N)]
    for b in range(0,N,512):
        sims=(Mt[b:b+512]@Mt.T)  # [bs,N]
        top=torch.topk(sims,K+1,dim=1).indices.cpu().numpy()  # includes self
        for r,i in enumerate(range(b,min(b+512,N))):
            if not ok[i]: continue
            cnt=Counter()
            for j in top[r]:
                if j==i or art[j]==art[i]: continue   # Exclude self and same artist (avoid trivial matches).
                for t in TAGS[j]: cnt[t]+=1
            desc[i]=[t for t,_ in cnt.most_common(5)]
    return desc
print('audio neighbours...',flush=True); SOUND=nn_consensus(AUD,okA)
print('lyrics neighbours...',flush=True); THEME=nn_consensus(LYR,okL)
out={tids[i]:{'sound':SOUND[i],'theme':THEME[i]} for i in range(N)}
os.makedirs('cache',exist_ok=True); json.dump(out,open('cache/content_desc.json','w'))
print('saved cache/content_desc.json',flush=True)
# Examples: find informative tracks.
def show(name_sub):
    for i in range(N):
        if name_sub.lower() in tm.iloc[i]['track_name'].lower():
            print(f"  {tm.iloc[i]['track_name'][:40]} — {tm.iloc[i]['artist_name'][:25]} | clean tags={TAGS[i][:4]} | SOUND(audio-NN)={SOUND[i]} | THEME={THEME[i]}")
            return
for s in ['Rain Sounds','Music For Sleep','Weightless','Get Lucky','Pumped Up Kicks','American Idiot','HYFR']:
    show(s)
