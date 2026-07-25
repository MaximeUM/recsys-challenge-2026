"""Enrich the firstpos dataset with SOUND (audio-NN cache from build_content_descriptors.py)
and THEME (Qwen3 theme vocabulary) descriptors for every candidate. Map candidate lines of the form
'Name by Artist [tags]' to track_id. Save sft_ctx1024_enriched_firstpos.parquet and
cache/content_desc_full.json (sound+theme, reused at inference).

    CUDA_VISIBLE_DEVICES=0 python src/reranking/build_full_descriptors.py
    CUDA_VISIBLE_DEVICES=0 python src/reranking/build_full_descriptors.py --desc_only

--desc_only: stop after cache/content_desc_full.json without modifying the SFT dataset.
This is all INFERENCE needs and avoids requiring the training data.
"""
import os, json, re, warnings, glob, argparse
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
_ap = argparse.ArgumentParser()
_ap.add_argument('--desc_only', action='store_true',
                 help="write only cache/content_desc_full.json, then stop")
_args = _ap.parse_args()
DATA='data/'
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids)
na2id={}
for i in range(N): na2id[(tm.iloc[i]['track_name'].strip().lower(), tm.iloc[i]['artist_name'].strip().lower())]=tids[i]
sound=json.load(open('cache/content_desc.json'))  # {tid:{sound:[...],theme:[...](ignored)}}
# THEME via qwen3 (text-to-text)
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA+'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True).set_index('track_id')
dim=next(len(r) for r in emb['lyrics-qwen3_embedding_0.6b'] if isinstance(r,(list,np.ndarray)) and len(r)>0)
LY=np.zeros((N,dim),np.float32); okL=np.zeros(N,bool)
for tid,row in emb['lyrics-qwen3_embedding_0.6b'].items():
    if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: LY[tidx[tid]]=row; okL[tidx[tid]]=True
LY=LY/(np.linalg.norm(LY,axis=1,keepdims=True)+1e-9)
THEMES=['love','heartbreak / breakup','party and dancing','sadness / melancholy','anger and aggression','death and loss','politics and protest','religion and faith','drugs and getting high','money and wealth','loneliness','nostalgia and memories','hope and inspiration','sex and desire','war','rebellion and freedom','friendship','nature and the outdoors','violence and crime','spirituality','growing up','fame']
from sentence_transformers import SentenceTransformer
sg=SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',device='cuda')
te=sg.encode([f'Instruct: Identify the main lyrical theme of a song.\nQuery:{t}' for t in THEMES],normalize_embeddings=True,convert_to_numpy=True)
sims=torch.tensor(LY,device='cuda')@torch.tensor(te,device='cuda').T
topt=torch.topk(sims,2,dim=1).indices.cpu().numpy()
theme={tids[i]:[THEMES[j] for j in topt[i]] for i in range(N) if okL[i]}
# full descriptor per track_id
desc={}
for tid in tids:
    s=sound.get(tid,{}).get('sound',[])[:3]; th=theme.get(tid,[])
    parts=[]
    if s: parts.append("sound: "+", ".join(s))
    if th: parts.append("themes: "+", ".join(th))
    desc[tid]=" | ".join(parts)
json.dump({'desc':desc}, open('cache/content_desc_full.json','w'))
print("descriptors saved", flush=True)
if _args.desc_only:
    raise SystemExit(0)
# enrich the parquet
def parse(line):
    l=re.sub(r'^\d+\.\s*','',line); l=re.sub(r'\s*\[[^\]]*\]\s*$','',l)
    if ' by ' in l:
        nm,ar=l.rsplit(' by ',1); return (nm.strip().lower(), ar.strip().lower())
    return None
df=pd.read_parquet('models/_sft_dataset_cache/sft_ctx1024_60000_top50_firstpos.parquet')
matched=tot=0
def enrich(cands):
    global matched,tot
    out=[]
    for line in cands.split('\n'):
        tot+=1; key=parse(line); d=''
        if key and key in na2id:
            d=desc.get(na2id[key],'')
            if d: matched+=1
        out.append(line + (f"  {{{d}}}" if d else ""))
    return '\n'.join(out)
df['candidates']=df['candidates'].apply(enrich)
out='models/_sft_dataset_cache/sft_ctx1024_enriched_firstpos.parquet'; df.to_parquet(out)
print(f"enriched {matched}/{tot} candidates ({100*matched/tot:.0f}%) -> {out}", flush=True)
print("\nenriched example:"); print('\n'.join(df['candidates'].iloc[0].split('\n')[:4]))
