"""Prototype: use sound match as a pool RERANKING signal, following the lyric_match_rerank.py setup.
Train a contrastive query(4B)-to-audio(CLAP) projection MLP, then on the dev SOUND/MOOD subset compute
sound_match = cos(proj(query), candidate audio-laion_clap)
and compare single-GT nDCG@20: POOL order / sorted by sound match / RRF / scorehead.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/sound_match_rerank.py
"""
import os, glob, re, json, random, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore'); torch.manual_seed(42); random.seed(42)
DATA='data/'; DEV=torch.device('cuda')
TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
train=pd.read_parquet(DATA+'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
dev=pd.read_parquet(DATA+'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids); lk=tm.set_index('track_id')
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
col='audio-laion_clap'; ad=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0)
AUD=np.zeros((N,ad),np.float32); hasA=np.zeros(N,bool)
for tid,row in emb[col].items():
    if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==ad: AUD[tidx[tid]]=row; hasA[tidx[tid]]=True
AUD=AUD/(np.linalg.norm(AUD,axis=1,keepdims=True)+1e-9)
print(f"audio-clap dim={ad} present={hasA.sum()}/{N}",flush=True)
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
SOUND=re.compile(r'\b(electronic|acoustic|instrumental|ambient|synth|beat|tempo|upbeat|energetic|energy|chill|mellow|dark|moody|melanchol|sound|vibe|groovy|danc|punk|metal|jazz|hip.?hop|rock|folk|lo.?fi|distort|guitar|bass|drum|vocal|fast|slow|aggressive|soft|heavy|nature sound)\b',re.I)
# Training items (conversation + GT) for the MLP.
tr=[]
for _,s in train.iterrows():
    gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx or not hasA[tidx[gt]]: continue
        tr.append({'q':conv_text(s['conversations'],tn),'gt':tidx[gt]})
random.Random(42).shuffle(tr); tr=tr[:30000]
# dev SOUND subset
dq=[]
for _,s in dev.iterrows():
    gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx: continue
        lu=[t['content'] for t in s['conversations'] if t['role']=='user' and int(t['turn_number'])==tn]
        if lu and SOUND.search(lu[0]): dq.append({'q':conv_text(s['conversations'],tn),'gt':tidx[gt],'key':f"{s['session_id']}|{tn}"})
print(f"train items={len(tr)} | dev SOUND requests={len(dq)}",flush=True)
from sentence_transformers import SentenceTransformer
ft=SentenceTransformer('models/qwen3_ft_dualencoder_4b_ctx1024',device='cuda',model_kwargs={'torch_dtype':torch.bfloat16})
ft.max_seq_length=1024; ft.tokenizer.truncation_side='left'; ft.eval()
def enc(texts): return ft.encode([f'Instruct: {TASK}\nQuery: {t.lower()}' for t in texts],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,device='cuda').astype(np.float32)
print("encode train...",flush=True); Qtr=enc([it['q'] for it in tr])
print("encode dev SOUND...",flush=True); Qdv=enc([it['q'] for it in dq])
del ft; import gc; gc.collect(); torch.cuda.empty_cache()
class Proj(nn.Module):
    def __init__(s,di,do): super().__init__(); s.n=nn.Sequential(nn.Linear(di,1024),nn.GELU(),nn.Linear(1024,do))
    def forward(s,x): z=s.n(x); return z/(z.norm(dim=-1,keepdim=True)+1e-9)
Mt=torch.tensor(AUD,device=DEV); Q=torch.tensor(Qtr,device=DEV); G=torch.tensor([it['gt'] for it in tr],device=DEV)
p=Proj(Qtr.shape[1],ad).to(DEV); opt=torch.optim.AdamW(p.parameters(),lr=1e-3,weight_decay=1e-4)
n=len(Q); idx=torch.arange(n,device=DEV)
print("train Proj...",flush=True)
for ep in range(40):
    p.train(); perm=idx[torch.randperm(n,device=DEV)]
    for b in range(0,n,512):
        sl=perm[b:b+512]; zq=p(Q[sl])
        loss=nn.functional.cross_entropy(zq@Mt[G[sl]].T/0.05,torch.arange(len(sl),device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
p.eval()
# scorehead baseline
sh={}
for f in glob.glob('exp/picks/picks_scorehead_*.parquet'):
    for _,r in pd.read_parquet(f).iterrows(): sh[r['key']]=(int(r['gt_pos']),int(r['gt_rank_full']))
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
K=200
with torch.no_grad(): Zq=p(torch.tensor(Qdv,device=DEV)).cpu().numpy()
def dcg(rank): return 1.0/np.log2(rank+1) if (rank is not None and rank<=20) else 0.0
S={'pool':[],'sound':[],'rrf':[],'scorehead':[]}; ginp=0
for it,zq in zip(dq,Zq):
    pl=pool.get(it['key'],[])[:K]; gt=it['gt']; inp=gt in pl; ginp+=inp
    r_pool=pl.index(gt)+1 if inp else None
    sm=np.array([float(zq@AUD[c]) if hasA[c] else -1e9 for c in pl])
    order_s=[pl[i] for i in np.argsort(-sm)]; r_sound=order_s.index(gt)+1 if inp else None
    rr={c:0.0 for c in pl}
    for rk,c in enumerate(pl): rr[c]+=1/(60+rk+1)
    for rk,c in enumerate(order_s): rr[c]+=1/(60+rk+1)
    order_r=[c for c,_ in sorted(rr.items(),key=lambda x:-x[1])]; r_rrf=order_r.index(gt)+1 if inp else None
    sv=sh.get(it['key']); r_sh=sv[1] if (sv and sv[0]>=0) else None
    S['pool'].append(dcg(r_pool)); S['sound'].append(dcg(r_sound)); S['rrf'].append(dcg(r_rrf)); S['scorehead'].append(dcg(r_sh))
nn_=len(dq)
print(f"\n=== nDCG@20 SOUND subset ({nn_} turns, GT in pool {ginp/nn_*100:.1f}%, denominator=all) ===")
for k in ['pool','sound','rrf','scorehead']: print(f"  {k:<12}: {np.mean(S[k]):.4f}")
