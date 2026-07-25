"""Dev prototype: intent-conditioned COVER-ART channel using SigLIP v1.
Detect cover/artwork requests, encode the description with the SigLIP TEXT encoder,
compare it by cosine similarity with the provided 768-d image embeddings, and measure GT recall.
Compare with the current ctx1024 text pool on the SAME turns and with RRF fusion.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/cover_channel_siglipv1.py
"""
import os, json, re, warnings, glob
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
DATA='data/'
dev=pd.read_parquet(DATA+'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids)

# provided image-siglip2 embeddings
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
dim=next(len(r) for r in emb['image-siglip2'] if isinstance(r,(list,np.ndarray)) and len(r)>0)
IMG=np.zeros((N,dim),np.float32); has=np.zeros(N,bool)
for tid,row in emb['image-siglip2'].items():
    if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: IMG[tidx[tid]]=row; has[tidx[tid]]=True
IMG=IMG/(np.linalg.norm(IMG,axis=1,keepdims=True)+1e-9)
print(f"image-siglip2: dim={dim}, present={has.sum()}/{N}", flush=True)

# Detect cover-art requests in the latest user turn.
COVER=re.compile(r'\b(cover|album art|artwork|art work|sleeve|album cover|the art|visual|painting|illustration|drawing|design|colou?rs?\b.*\bcover)\b', re.I)
items=[]
for _,s in dev.iterrows():
    cs=s['conversations']; gtbt={int(t['turn_number']):t['content'] for t in cs if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx: continue
        lu=[t['content'] for t in cs if t['role']=='user' and int(t['turn_number'])==tn]
        if lu and COVER.search(lu[0]):
            items.append({'q':lu[0],'gt':tidx[gt],'sid':s['session_id'],'tn':tn})
print(f"cover-art requests detected on dev: {len(items)}", flush=True)
if not items: raise SystemExit

# SigLIP2 text encoder
from transformers import AutoModel, AutoProcessor
M='google/siglip-base-patch16-224'
proc=AutoProcessor.from_pretrained(M); sg=AutoModel.from_pretrained(M,torch_dtype=torch.float32).to('cuda').eval()
@torch.no_grad()
def txt_emb(texts):
    out=[]
    for i in range(0,len(texts),32):
        b=texts[i:i+32]
        inp=proc(text=b,return_tensors='pt',padding='max_length',truncation=True,max_length=64).to('cuda')
        o=sg.get_text_features(**inp)
        f=o if torch.is_tensor(o) else (o.pooler_output if getattr(o,'pooler_output',None) is not None else o.last_hidden_state[:,0])
        f=torch.nn.functional.normalize(f,dim=-1)
        out.append(f.cpu().numpy())
    return np.concatenate(out,0)
qe=txt_emb([it['q'] for it in items])
print(f"siglip2 text dim={qe.shape[1]} (must equal {dim})", flush=True)

IMGt=torch.tensor(IMG,device='cuda')
# Current text pool (ctx1024) as the baseline on these turns.
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
KS=[20,50,500]
def rank_in(lst,gt):
    return lst.index(gt)+1 if gt in lst else 10**9
sig_rank=[]; txt_rank=[]; rrf_rank=[]
QE=torch.tensor(qe,device='cuda')
for i,it in enumerate(items):
    sc=(QE[i]@IMGt.T).cpu().numpy(); sig=list(np.argsort(-sc)[:500])
    sig_rank.append(rank_in(sig,it['gt']))
    tl=pool.get(f"{it['sid']}|{it['tn']}",[]); txt_rank.append(rank_in(tl,it['gt']))
    # RRF fusion: siglip + text
    rrf=np.zeros(N);
    for r,idx in enumerate(sig): rrf[idx]+=1/(60+r+1)
    for r,idx in enumerate(tl): rrf[idx]+=1/(60+r+1)
    order=list(np.argsort(-rrf)[:500]); rrf_rank.append(rank_in(order,it['gt']))
def rec(ranks,k): return 100*np.mean([r<=k for r in ranks])
print(f"\n=== recall on {len(items)} COVER-ART requests ===")
print(f"  {'channel':<22}{'R@20':>7}{'R@50':>7}{'R@500':>7}")
for lab,rk in [('text (current pool)',txt_rank),('SigLIP2 cover art',sig_rank),('RRF text+cover',rrf_rank)]:
    print(f"  {lab:<22}{rec(rk,20):>7.1f}{rec(rk,50):>7.1f}{rec(rk,500):>7.1f}")
