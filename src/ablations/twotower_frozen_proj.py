"""Two-tower model with a FROZEN track tower of concatenated provided multimodal embeddings
and LEARNED projection heads mapping the conversational request into track space.

The ctx1024 dense baseline encodes tracks from short name/artist/tag text. Here, each track is the
frozen concatenation of six provided embeddings (audio/image/CF/attributes/lyrics/metadata), and a
query-to-track-space projection is LEARNED with InfoNCE and in-batch negatives. This differs from
multimodal_rrf_channel.py, which uses nearest neighbors over mean history as an RRF channel.

Cached reusable costs under cache/mm/:
  - track_dense: 47k track texts encoded by the 4B (baseline + variant 2), about 15 minutes
  - q_dev / q_tr: dev requests + training subsample encoded by the 4B
Once cached, MLP training and evaluation take minutes.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/twotower_frozen_proj.py --dev_n 200 --n_train 30000
    # Full dev reuses caches and re-encodes only the 8,000 dev requests:
    CUDA_VISIBLE_DEVICES=0 python src/ablations/twotower_frozen_proj.py --dev_n 8000 --n_train 30000
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, warnings, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, glob
from pathlib import Path
warnings.filterwarnings('ignore'); torch.manual_seed(42); random.seed(42)

ap=argparse.ArgumentParser()
ap.add_argument('--dev_n', type=int, default=200)        # 0/8000 = full-dev
ap.add_argument('--n_train', type=int, default=30000)    # Training turns used to fit the MLPs.
ap.add_argument('--dim', type=int, default=512)          # shared space
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--bs', type=int, default=512)
ap.add_argument('--temp', type=float, default=0.05)
ap.add_argument('--lr', type=float, default=1e-3)
args=ap.parse_args()
DATA=Path('data'); MODEL=Path('models/qwen3_ft_dualencoder_4b_ctx1024')
CACHE=Path('cache/mm'); CACHE.mkdir(parents=True, exist_ok=True)
TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
DEV=torch.device('cuda')
t0=time.time(); log=lambda m: print(f'[{time.time()-t0:.0f}s] {m}', flush=True)

# Data.
train=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
dev=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids); lk=tm.set_index('track_id')
def short(t):
    if t not in lk.index: return t
    r=lk.loc[t]; return f"{r['track_name']} - {r['artist_name']}"
def conv_text(convs,tt):
    L=[]
    for t in convs:
        if t['turn_number']>=tt: break
        ro,co=t['role'],t['content']
        if ro=='music': ro,co='assistant_played',short(co)
        L.append(f'{ro}: {co}')
    for t in convs:
        if t['turn_number']==tt and t['role']=='user': L.append(f"user (REQUEST): {t['content']}"); break
    return '\n'.join(L)
def dense_text(r):
    p=[f"track_name: {r['track_name']}",f"artist_name: {r['artist_name']}",f"album_name: {r['album_name']}"]
    if isinstance(r['tag_list'],(list,np.ndarray)) and len(r['tag_list'])>0: p.append(f"tags: {', '.join(r['tag_list'])}")
    return '\n'.join(p)
def build_items(df):
    out=[]
    for _,s in df.iterrows():
        gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
        for tn in range(1,9):
            gt=gtbt.get(tn)
            if gt is None or gt not in tidx: continue
            out.append({'q':conv_text(s['conversations'],tn),'gt':tidx[gt]})
    return out
dev_items=build_items(dev)
if args.dev_n and args.dev_n<len(dev_items):
    dev_items=random.Random(0).sample(dev_items, args.dev_n)
train_items=build_items(train)
random.Random(42).shuffle(train_items); train_items=train_items[:args.n_train]
log(f'{len(train_items)} train turns, {len(dev_items)} dev turns')

# 4B encoder with caching.
def encode(texts, prefix=None, bs=16, mxl=1024):
    from sentence_transformers import SentenceTransformer
    global _ENC
    if '_ENC' not in globals() or _ENC is None:
        _ENC=SentenceTransformer(str(MODEL),device='cuda',model_kwargs={'torch_dtype':torch.bfloat16})
        _ENC.max_seq_length=1024; _ENC.tokenizer.truncation_side='left'; _ENC.eval()
    xs=[f'Instruct: {prefix}\nQuery: {t.lower()}' for t in texts] if prefix else texts
    e=_ENC.encode(xs,batch_size=bs,show_progress_bar=True,normalize_embeddings=True,
                  convert_to_numpy=True,device='cuda')
    return e.astype(np.float32)
def cached(name, fn):
    p=CACHE/name
    if p.exists(): log(f'cache hit {name}'); return np.load(p)
    log(f'computing {name} ...'); a=fn(); np.save(p,a); return a
_ENC=None
# Track texts (dense baseline + variant 2), frozen and reusable.
track_dense=cached('track_dense.npy', lambda: encode([dense_text(r) for _,r in tm.iterrows()], bs=64))
# Requests: dev depends on dev_n and training on the subsample, so use dedicated names.
q_dev=cached(f'q_dev_{len(dev_items)}.npy', lambda: encode([it['q'] for it in dev_items], prefix=TASK))
q_tr =cached(f'q_tr_{len(train_items)}.npy', lambda: encode([it['q'] for it in train_items], prefix=TASK))
if _ENC is not None: del _ENC; _ENC=None; torch.cuda.empty_cache()

# ---------- provided multimodal track embeddings ----------
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA/'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet')))],ignore_index=True).set_index('track_id')
MODS=['audio-laion_clap','image-siglip2','cf-bpr','attributes-qwen3_embedding_0.6b','lyrics-qwen3_embedding_0.6b','metadata-qwen3_embedding_0.6b']
def mat(col):
    dim=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0)
    M=np.zeros((N,dim),dtype=np.float32)
    for tid,row in emb[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: M[tidx[tid]]=np.asarray(row,np.float32)
    M=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
    return M
blocks={m:mat(m) for m in MODS}; log('multimodal embeddings loaded')
MM=np.concatenate([blocks[m] for m in MODS],axis=1)                 # 4480
MMd=np.concatenate([MM, track_dense],axis=1)                        # 4480 + 2560 (variant 2)
dims=[blocks[m].shape[1] for m in MODS]

# Projection heads + contrastive training.
class Proj(nn.Module):
    def __init__(s,din,d): super().__init__(); s.net=nn.Sequential(nn.Linear(din,1024),nn.GELU(),nn.Linear(1024,d))
    def forward(s,x): z=s.net(x); return z/(z.norm(dim=-1,keepdim=True)+1e-9)
class Gate(nn.Module):  # Learned per-modality weighting (variant 3).
    def __init__(s,dims,d):
        super().__init__(); s.dims=dims; s.w=nn.Parameter(torch.zeros(len(dims))); s.proj=Proj(sum(dims),d)
    def forward(s,x):
        wsm=torch.softmax(s.w,0); parts=[]; o=0
        for wi,dd in zip(wsm,s.dims): parts.append(x[:,o:o+dd]*wi*len(s.dims)); o+=dd
        return s.proj(torch.cat(parts,1))

def train_eval(name, Tmat, track_head):
    Dtr=Tmat.shape[1]
    qp=Proj(q_tr.shape[1],args.dim).to(DEV)
    th=track_head if track_head is not None else Proj(Dtr,args.dim).to(DEV)
    th=th.to(DEV)
    T=torch.tensor(Tmat,device=DEV)
    Qtr=torch.tensor(q_tr,device=DEV); gt_tr=torch.tensor([it['gt'] for it in train_items],device=DEV)
    opt=torch.optim.AdamW(list(qp.parameters())+list(th.parameters()),lr=args.lr,weight_decay=1e-4)
    n=len(train_items); idx=torch.arange(n,device=DEV)
    for ep in range(args.epochs):
        qp.train(); th.train(); perm=idx[torch.randperm(n,device=DEV)]
        for b in range(0,n,args.bs):
            sl=perm[b:b+args.bs]
            zq=qp(Qtr[sl]); zt=th(T[gt_tr[sl]])
            logits=zq@zt.T/args.temp
            loss=nn.functional.cross_entropy(logits,torch.arange(len(sl),device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    # Evaluation: project the entire track catalog once.
    qp.eval(); th.eval()
    with torch.no_grad():
        Tp=torch.cat([th(T[i:i+4096]) for i in range(0,N,4096)],0)
        Qd=qp(torch.tensor(q_dev,device=DEV))
        sims=Qd@Tp.T
        gtd=torch.tensor([it['gt'] for it in dev_items],device=DEV)
        ranks=(sims> sims.gather(1,gtd[:,None])).sum(1)+1   # GT rank (1-based)
    r=ranks.cpu().numpy()
    ndcg=np.mean([1/np.log2(x+1) if x<=20 else 0 for x in r])
    res={'nDCG@20':ndcg,'R@20':np.mean(r<=20),'R@50':np.mean(r<=50),'R@500':np.mean(r<=500)}
    log(f'{name:<22} ' + '  '.join(f'{k} {v:.4f}' for k,v in res.items()))
    return res

# Baseline: raw dense ctx1024 cosine similarity without learned projection.
def baseline():
    T=torch.tensor(track_dense,device=DEV); T=T/(T.norm(dim=1,keepdim=True)+1e-9)
    Qd=torch.tensor(q_dev,device=DEV)
    sims=Qd@T.T; gtd=torch.tensor([it['gt'] for it in dev_items],device=DEV)
    r=((sims>sims.gather(1,gtd[:,None])).sum(1)+1).cpu().numpy()
    ndcg=np.mean([1/np.log2(x+1) if x<=20 else 0 for x in r])
    log(f'{"BASELINE dense ctx1024":<22} nDCG@20 {ndcg:.4f}  R@20 {np.mean(r<=20):.4f}  R@50 {np.mean(r<=50):.4f}  R@500 {np.mean(r<=500):.4f}')
baseline()
train_eval('V1 mm(6) concat', MM, None)
train_eval('V2 mm(6)+dense', MMd, None)
train_eval('V3 mm(6) gating', MM, Gate(dims,args.dim))
print('\n(200 = noisy quick read; reconfirm on full dev before drawing conclusions)')
