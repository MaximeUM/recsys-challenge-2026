"""Align the query with image/lyrics spaces, as done for text.
Train a contrastive MLP from the frozen FT-4B query to each modality space, using the GT as positive
and in-batch negatives. Evaluate on dev cover/lyrics requests and compare off-the-shelf raw
SigLIP2/Qwen3, MLP-aligned, and text results. Reuse cache/mm/q_tr_30000.npy (same items as twotower_frozen_proj.py).

    CUDA_VISIBLE_DEVICES=0 python src/ablations/aligned_modality_channel.py
"""
import os, json, re, warnings, glob, random
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore'); torch.manual_seed(42); random.seed(42)
DATA='data/'; DEV=torch.device('cuda'); TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
train=pd.read_parquet(DATA+'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
dev=pd.read_parquet(DATA+'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
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
def build_items(df):
    out=[]
    for _,s in df.iterrows():
        gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
        for tn in range(1,9):
            gt=gtbt.get(tn)
            if gt is None or gt not in tidx: continue
            out.append({'q':conv_text(s['conversations'],tn),'gt':tidx[gt]})
    return out
# Reconstruct the exact items from 122 so GT aligns with q_tr_30000.npy.
tr_items=build_items(train); random.Random(42).shuffle(tr_items); tr_items=tr_items[:30000]
q_tr=np.load('cache/mm/q_tr_30000.npy'); assert len(q_tr)==len(tr_items), (len(q_tr),len(tr_items))
gt_tr=np.array([it['gt'] for it in tr_items])
# Modality matrices.
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
def mat(col):
    dim=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0); M=np.zeros((N,dim),np.float32)
    for tid,row in emb[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: M[tidx[tid]]=row
    return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
IMG=mat('image-siglip2'); LY=mat('lyrics-qwen3_embedding_0.6b')
# Dev cover/lyrics requests.
COVER=re.compile(r'\b(cover|album art|artwork|sleeve|album cover|the art|visual|painting|illustration|drawing|design)\b',re.I)
LYR=re.compile(r"\b(lyric|lyrics|the line|the words|goes like|sings?|singing|the part where|verse|chorus|a line about|words go|hook)\b",re.I)
def dev_q(rx):
    its=[]
    for _,s in dev.iterrows():
        gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
        for tn in sorted(gtbt):
            gt=gtbt[tn]
            if gt not in tidx: continue
            lu=[t['content'] for t in s['conversations'] if t['role']=='user' and int(t['turn_number'])==tn]
            if lu and rx.search(lu[0]): its.append({'q':conv_text(s['conversations'],tn),'gt':tidx[gt],'sid':s['session_id'],'tn':tn})
    return its
cov=dev_q(COVER); lyq=dev_q(LYR)
print(f"dev cover={len(cov)} lyrics={len(lyq)}",flush=True)
# Encode dev requests with FT-4B.
from sentence_transformers import SentenceTransformer
ft=SentenceTransformer('models/qwen3_ft_dualencoder_4b_ctx1024',device='cuda',model_kwargs={'torch_dtype':torch.bfloat16})
ft.max_seq_length=1024; ft.tokenizer.truncation_side='left'; ft.eval()
def enc(qs): return ft.encode([f'Instruct: {TASK}\nQuery: {q.lower()}' for q in qs],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,device='cuda').astype(np.float32)
qcov=enc([it['q'] for it in cov]); qlyr=enc([it['q'] for it in lyq])
del ft; torch.cuda.empty_cache()
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
class Proj(nn.Module):
    def __init__(s,di,do): super().__init__(); s.n=nn.Sequential(nn.Linear(di,1024),nn.GELU(),nn.Linear(1024,do))
    def forward(s,x): z=s.n(x); return z/(z.norm(dim=-1,keepdim=True)+1e-9)
def train_proj(MOD):
    Mt=torch.tensor(MOD,device=DEV); Q=torch.tensor(q_tr,device=DEV); G=torch.tensor(gt_tr,device=DEV)
    p=Proj(q_tr.shape[1],MOD.shape[1]).to(DEV); opt=torch.optim.AdamW(p.parameters(),lr=1e-3,weight_decay=1e-4)
    n=len(Q); idx=torch.arange(n,device=DEV)
    for ep in range(40):
        p.train(); perm=idx[torch.randperm(n,device=DEV)]
        for b in range(0,n,512):
            sl=perm[b:b+512]; zq=p(Q[sl]); zt=Mt[G[sl]]
            loss=nn.functional.cross_entropy(zq@zt.T/0.05,torch.arange(len(sl),device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    p.eval(); return p,Mt
def rk(lst,gt): return lst.index(gt)+1 if gt in lst else 10**9
def evalp(p,Mt,qd,its):
    with torch.no_grad(): zq=p(torch.tensor(qd,device=DEV))
    sims=zq@Mt.T; modrank={}; R=[]
    for i,it in enumerate(its):
        order=torch.topk(sims[i],500).indices.cpu().tolist(); R.append(rk(order,it['gt'])); modrank[i]=order
    return R,modrank
rec=lambda r,k:100*np.mean([x<=k for x in r])
def rrf(text_list, mod_order, k=500):
    sc={}
    for r,idx in enumerate(text_list): sc[idx]=sc.get(idx,0)+1/(60+r+1)
    for r,idx in enumerate(mod_order): sc[idx]=sc.get(idx,0)+1/(60+r+1)
    return [i for i,_ in sorted(sc.items(),key=lambda x:-x[1])][:k]
for name,MOD,qd,its in [('COVER ART (image)',IMG,qcov,cov),('LYRICS',LY,qlyr,lyq)]:
    p,Mt=train_proj(MOD); R,modrank=evalp(p,Mt,qd,its)
    txt=[]; fus=[]
    for i,it in enumerate(its):
        tl=pool.get(f"{it['sid']}|{it['tn']}",[]); txt.append(rk(tl,it['gt']))
        fus.append(rk(rrf(tl,modrank[i]),it['gt']))
    print(f"\n=== {name} ({len(its)} req) ===")
    print(f"  {'channel':<30}{'R@20':>7}{'R@50':>7}{'R@500':>7}")
    print(f"  {'text (pool)':<30}{rec(txt,20):>7.1f}{rec(txt,50):>7.1f}{rec(txt,500):>7.1f}")
    print(f"  {'query->modality MLP':<30}{rec(R,20):>7.1f}{rec(R,50):>7.1f}{rec(R,500):>7.1f}")
    print(f"  {'RRF text + modality':<30}{rec(fus,20):>7.1f}{rec(fus,50):>7.1f}{rec(fus,500):>7.1f}")
