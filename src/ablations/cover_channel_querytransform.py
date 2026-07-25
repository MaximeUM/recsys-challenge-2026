"""Test: query transformation for the COVER-ART channel.
A language model (gemma-3n) converts the conversational request to a short VISUAL CAPTION in SigLIP's
native input form, then compares SigLIP2 text by cosine similarity with image-siglip2. Compare RAW
REQUEST, CAPTION, and text pool on dev cover-art requests.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/cover_channel_querytransform.py
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
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
dim=next(len(r) for r in emb['image-siglip2'] if isinstance(r,(list,np.ndarray)) and len(r)>0)
IMG=np.zeros((N,dim),np.float32)
for tid,row in emb['image-siglip2'].items():
    if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: IMG[tidx[tid]]=row
IMG=IMG/(np.linalg.norm(IMG,axis=1,keepdims=True)+1e-9)
COVER=re.compile(r'\b(cover|album art|artwork|sleeve|album cover|the art|visual|painting|illustration|drawing|design)\b', re.I)
items=[]
for _,s in dev.iterrows():
    cs=s['conversations']; gtbt={int(t['turn_number']):t['content'] for t in cs if t['role']=='music'}
    for tn in sorted(gtbt):
        gt=gtbt[tn]
        if gt not in tidx: continue
        lu=[t['content'] for t in cs if t['role']=='user' and int(t['turn_number'])==tn]
        if lu and COVER.search(lu[0]): items.append({'q':lu[0],'gt':tidx[gt],'sid':s['session_id'],'tn':tn})
print(f"cover-art requests: {len(items)}", flush=True)

# 1) Language model -> short visual caption.
from transformers import AutoModelForCausalLM, AutoTokenizer
G='google/gemma-3n-E4B-it'; gt=AutoTokenizer.from_pretrained(G); gt.padding_side='left'; gt.truncation_side='left'
if gt.pad_token_id is None: gt.pad_token=gt.eos_token
gm=AutoModelForCausalLM.from_pretrained(G,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
SYS=("Extract ONLY a short visual caption (~8 words) of the ALBUM COVER the user is describing: objects, colors, style, layout. "
     "No preamble, no quotes, just the caption.")
@torch.no_grad()
def caption(qs):
    out=[]
    for i in range(0,len(qs),16):
        b=qs[i:i+16]
        txts=[gt.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True) for q in b]
        enc=gt(txts,return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda:0')
        o=gm.generate(**enc,max_new_tokens=30,do_sample=False,pad_token_id=gt.eos_token_id); L=enc['input_ids'].shape[1]
        out+=[gt.decode(o[j,L:],skip_special_tokens=True).strip().replace('\n',' ') for j in range(len(b))]
    return out
caps=caption([it['q'] for it in items])
for it,c in zip(items[:4],caps[:4]): print(f"  Q: {it['q'][:80]} -> CAP: {c[:80]}", flush=True)
import gc; del gm,gt; gc.collect(); torch.cuda.empty_cache()

# 2) SigLIP2 text encoder
from transformers import AutoModel, AutoProcessor
M='google/siglip2-base-patch16-224'; proc=AutoProcessor.from_pretrained(M); sg=AutoModel.from_pretrained(M,torch_dtype=torch.float32).to('cuda').eval()
@torch.no_grad()
def sgtxt(texts):
    out=[]
    for i in range(0,len(texts),32):
        inp=proc(text=texts[i:i+32],return_tensors='pt',padding='max_length',truncation=True,max_length=64).to('cuda')
        o=sg.get_text_features(**inp); f=o if torch.is_tensor(o) else o.pooler_output
        out.append(torch.nn.functional.normalize(f,dim=-1).cpu().numpy())
    return np.concatenate(out,0)
qe_raw=sgtxt([it['q'] for it in items]); qe_cap=sgtxt(caps)
IMGt=torch.tensor(IMG,device='cuda')
pool={f"{r['session_id']}|{int(r['turn'])}":json.loads(r['pool']) for _,r in pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet').iterrows()}
def rk(lst,gt): return lst.index(gt)+1 if gt in lst else 10**9
def chan(qe):
    QE=torch.tensor(qe,device='cuda'); R=[]
    for i,it in enumerate(items):
        sc=(QE[i]@IMGt.T).cpu().numpy(); R.append(rk(list(np.argsort(-sc)[:500]),it['gt']))
    return R
raw=chan(qe_raw); cap=chan(qe_cap); txt=[rk(pool.get(f"{it['sid']}|{it['tn']}",[]),it['gt']) for it in items]
rec=lambda r,k:100*np.mean([x<=k for x in r])
print(f"\n=== cover-art recall ({len(items)} requests) — query transformation ===")
print(f"  {'channel':<26}{'R@20':>7}{'R@50':>7}{'R@500':>7}")
for lab,r in [('text (pool)',txt),('SigLIP2 RAW REQUEST',raw),('SigLIP2 CAPTION (model)',cap)]:
    print(f"  {lab:<26}{rec(r,20):>7.1f}{rec(r,50):>7.1f}{rec(r,500):>7.1f}")
