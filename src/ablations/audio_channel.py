"""AUDIO channel for sound/genre/mood failures. Align query -> CLAP audio with an MLP,
evaluate on dev sound/mood requests, and inspect recoverable versus unusual GT cases.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/audio_channel.py
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
def tags(i):
    tl=tm.iloc[i]['tag_list']; return ', '.join(list(tl)[:6]) if isinstance(tl,(list,np.ndarray)) and len(tl)>0 else ''
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
def gts(df):
    out=[]
    for _,s in df.iterrows():
        gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
        for tn in range(1,9):
            gt=gtbt.get(tn)
            if gt is None or gt not in tidx: continue
            out.append({'gt':tidx[gt]})
    return out
tr_items=gts(train); random.Random(42).shuffle(tr_items); tr_items=tr_items[:30000]
q_tr=np.load('cache/mm/q_tr_30000.npy'); gt_tr=np.array([it['gt'] for it in tr_items])
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
def mat(col):
    dim=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0); M=np.zeros((N,dim),np.float32)
    for tid,row in emb[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: M[tidx[tid]]=row
    return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
AUD=mat('audio-laion_clap')
SOUND=re.compile(r'\b(electronic|acoustic|instrumental|ambient|synth|beat|tempo|upbeat|energetic|energy|chill|mellow|dark|moody|melanchol|sound|vibe|groovy|danc|punk|metal|jazz|hip.?hop|rock|folk|lo.?fi|distort|guitar|bass|drum|vocal|fast|slow|aggressive|soft|heavy|nature sound)\b',re.I)
dq=[]
for _,s in dev.iterrows():
    gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx: continue
        lu=[t['content'] for t in s['conversations'] if t['role']=='user' and int(t['turn_number'])==tn]
        if lu and SOUND.search(lu[0]): dq.append({'q':conv_text(s['conversations'],tn),'req':lu[0],'gt':tidx[gt],'sid':s['session_id'],'tn':tn})
print(f"sound/mood requests: {len(dq)}",flush=True)
class Proj(nn.Module):
    def __init__(s,di,do): super().__init__(); s.n=nn.Sequential(nn.Linear(di,1024),nn.GELU(),nn.Linear(1024,do))
    def forward(s,x): z=s.n(x); return z/(z.norm(dim=-1,keepdim=True)+1e-9)
Mt=torch.tensor(AUD,device=DEV); Q=torch.tensor(q_tr,device=DEV); G=torch.tensor(gt_tr,device=DEV)
p=Proj(q_tr.shape[1],AUD.shape[1]).to(DEV); opt=torch.optim.AdamW(p.parameters(),lr=1e-3,weight_decay=1e-4)
n=len(Q); idx=torch.arange(n,device=DEV)
for ep in range(40):
    p.train(); perm=idx[torch.randperm(n,device=DEV)]
    for b in range(0,n,512):
        sl=perm[b:b+512]; zq=p(Q[sl])
        loss=nn.functional.cross_entropy(zq@Mt[G[sl]].T/0.05,torch.arange(len(sl),device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
p.eval()
from sentence_transformers import SentenceTransformer
ft=SentenceTransformer('models/qwen3_ft_dualencoder_4b_ctx1024',device='cuda',model_kwargs={'torch_dtype':torch.bfloat16}); ft.max_seq_length=1024; ft.tokenizer.truncation_side='left'; ft.eval()
qd=ft.encode([f'Instruct: {TASK}\nQuery: {it["q"].lower()}' for it in dq],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,device='cuda').astype(np.float32)
del ft; import gc; gc.collect(); torch.cuda.empty_cache()
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
def rk(lst,gt): return lst.index(gt)+1 if gt in lst else 10**9
with torch.no_grad(): zq=p(torch.tensor(qd,device=DEV))
sims=zq@Mt.T
def rrf(tl,mo):
    sc={}
    for r,i in enumerate(tl): sc[i]=sc.get(i,0)+1/(60+r+1)
    for r,i in enumerate(mo): sc[i]=sc.get(i,0)+1/(60+r+1)
    return [i for i,_ in sorted(sc.items(),key=lambda x:-x[1])][:500]
txt=[];aud=[];fus=[]
for i,it in enumerate(dq):
    tl=pool.get(f"{it['sid']}|{it['tn']}",[]); mo=torch.topk(sims[i],500).indices.cpu().tolist()
    txt.append(rk(tl,it['gt'])); aud.append(rk(mo,it['gt'])); fus.append(rk(rrf(tl,mo),it['gt']))
rec=lambda r,k:100*np.mean([x<=k for x in r])
print(f"\n=== sound/mood requests ({len(dq)}) ===")
print(f"  {'channel':<22}{'R@20':>7}{'R@50':>7}{'R@500':>7}")
for lab,r in [('text',txt),('audio (aligned)',aud),('RRF',fus)]:
    print(f"  {lab:<22}{rec(r,20):>7.1f}{rec(r,50):>7.1f}{rec(r,500):>7.1f}")
# Inspect GT cases where TEXT misses (GT outside top 50).
print("\n=== GT inspection (cases where text misses @50) ===")
miss=[i for i in range(len(dq)) if txt[i]>50][:8]
for i in miss:
    it=dq[i]; g=it['gt']
    print(f"  req: {it['req'][:85]}")
    print(f"    GT: {short(tids[g])} | tags: {tags(g)} | text rank={txt[i] if txt[i]<10**8 else '>500'} audio={aud[i] if aud[i]<10**8 else '>500'}")
