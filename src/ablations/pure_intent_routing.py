"""Test routing on CLEAR intent. A language model classifies cover-art requests as
'pure visual' (ONLY cover art, no artist/title/genre) versus 'mixed'. On the PURE subset,
text has no lexical anchor, so the MLP-aligned image channel should win.
Compare text, image channel, and RRF separately on PURE and MIXED requests.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/pure_intent_routing.py
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
            out.append({'gt':tidx[gt]})
    return out
tr_items=build_items(train); random.Random(42).shuffle(tr_items); tr_items=tr_items[:30000]
q_tr=np.load('cache/mm/q_tr_30000.npy'); gt_tr=np.array([it['gt'] for it in tr_items])
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
def mat(col):
    dim=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0); M=np.zeros((N,dim),np.float32)
    for tid,row in emb[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: M[tidx[tid]]=row
    return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
IMG=mat('image-siglip2')
COVER=re.compile(r'\b(cover|album art|artwork|sleeve|album cover|the art|visual|painting|illustration|drawing|design)\b',re.I)
cov=[]
for _,s in dev.iterrows():
    gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx: continue
        lu=[t['content'] for t in s['conversations'] if t['role']=='user' and int(t['turn_number'])==tn]
        if lu and COVER.search(lu[0]): cov.append({'q':conv_text(s['conversations'],tn),'req':lu[0],'gt':tidx[gt],'sid':s['session_id'],'tn':tn})
print(f"cover queries: {len(cov)}",flush=True)
# Query-to-image MLP.
class Proj(nn.Module):
    def __init__(s,di,do): super().__init__(); s.n=nn.Sequential(nn.Linear(di,1024),nn.GELU(),nn.Linear(1024,do))
    def forward(s,x): z=s.n(x); return z/(z.norm(dim=-1,keepdim=True)+1e-9)
Mt=torch.tensor(IMG,device=DEV); Q=torch.tensor(q_tr,device=DEV); G=torch.tensor(gt_tr,device=DEV)
p=Proj(q_tr.shape[1],IMG.shape[1]).to(DEV); opt=torch.optim.AdamW(p.parameters(),lr=1e-3,weight_decay=1e-4)
n=len(Q); idx=torch.arange(n,device=DEV)
for ep in range(40):
    p.train(); perm=idx[torch.randperm(n,device=DEV)]
    for b in range(0,n,512):
        sl=perm[b:b+512]; zq=p(Q[sl])
        loss=nn.functional.cross_entropy(zq@Mt[G[sl]].T/0.05,torch.arange(len(sl),device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
p.eval()
# Encode dev cover-art requests.
from sentence_transformers import SentenceTransformer
ft=SentenceTransformer('models/qwen3_ft_dualencoder_4b_ctx1024',device='cuda',model_kwargs={'torch_dtype':torch.bfloat16}); ft.max_seq_length=1024; ft.tokenizer.truncation_side='left'; ft.eval()
qcov=ft.encode([f'Instruct: {TASK}\nQuery: {it["q"].lower()}' for it in cov],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,device='cuda').astype(np.float32)
del ft; import gc; gc.collect(); torch.cuda.empty_cache()
# Classify PURE versus MIXED with a language model.
from transformers import AutoModelForCausalLM, AutoTokenizer
GG='google/gemma-3n-E4B-it'; gt=AutoTokenizer.from_pretrained(GG); gt.padding_side='left'; gt.truncation_side='left'
if gt.pad_token_id is None: gt.pad_token=gt.eos_token
gm=AutoModelForCausalLM.from_pretrained(GG,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
SYS=("The user is trying to find a song/album. Does the user identify it using ONLY the ALBUM COVER / visual appearance "
     "(colors, shapes, imagery) with NO other clue (no artist, no title, no genre, no lyrics, no era)? "
     "Answer EXACTLY 'PURE' if cover is the only clue, else 'MIXED'.")
@torch.no_grad()
def classify(reqs):
    out=[]
    for i in range(0,len(reqs),16):
        b=reqs[i:i+16]; txts=[gt.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':r}],tokenize=False,add_generation_prompt=True) for r in b]
        enc=gt(txts,return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda:0')
        o=gm.generate(**enc,max_new_tokens=4,do_sample=False,pad_token_id=gt.eos_token_id); L=enc['input_ids'].shape[1]
        out+=['pure' in gt.decode(o[j,L:],skip_special_tokens=True).strip().lower() for j in range(len(b))]
    return out
pure=classify([it['req'] for it in cov]); del gm; gc.collect(); torch.cuda.empty_cache()
print(f"PURE={sum(pure)} / MIXED={len(pure)-sum(pure)}",flush=True)
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
def rk(lst,gt): return lst.index(gt)+1 if gt in lst else 10**9
with torch.no_grad(): zq=p(torch.tensor(qcov,device=DEV))
sims=zq@Mt.T
def rrf(tl,mo):
    sc={}
    for r,i in enumerate(tl): sc[i]=sc.get(i,0)+1/(60+r+1)
    for r,i in enumerate(mo): sc[i]=sc.get(i,0)+1/(60+r+1)
    return [i for i,_ in sorted(sc.items(),key=lambda x:-x[1])][:500]
rec=lambda r,k:100*np.mean([x<=k for x in r]) if r else 0
for sub,mask in [('PURE (cover art only)',pure),('MIXED',[not x for x in pure])]:
    ids=[i for i,m in enumerate(mask) if m]
    if not ids: continue
    txt=[rk(pool.get(f"{cov[i]['sid']}|{cov[i]['tn']}",[]),cov[i]['gt']) for i in ids]
    img=[rk(torch.topk(sims[i],500).indices.cpu().tolist(),cov[i]['gt']) for i in ids]
    fus=[rk(rrf(pool.get(f"{cov[i]['sid']}|{cov[i]['tn']}",[]),torch.topk(sims[i],500).indices.cpu().tolist()),cov[i]['gt']) for i in ids]
    print(f"\n=== {sub} ({len(ids)} req) ===")
    print(f"  {'channel':<22}{'R@20':>7}{'R@50':>7}{'R@500':>7}")
    for lab,r in [('text',txt),('image (aligned)',img),('RRF',fus)]:
        print(f"  {lab:<22}{rec(r,20):>7.1f}{rec(r,50):>7.1f}{rec(r,500):>7.1f}")
