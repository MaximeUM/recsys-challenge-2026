"""Enrich the firstpos TOP-200 dataset by adding {sound|themes} to every candidate.
Reuse cache/content_desc_full.json from build_full_descriptors.py; no GPU is needed. Preserve turn_number.
  In  : models/_sft_dataset_cache/sft_ctx1024_60000_top200_firstpos.parquet
  Out : models/_sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet

    python src/reranking/enrich_reranker_data.py
"""
import json, re
import numpy as np, pandas as pd
DATA='data/'
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); N=len(tids)
na2id={}
for i in range(N): na2id[(tm.iloc[i]['track_name'].strip().lower(), tm.iloc[i]['artist_name'].strip().lower())]=tids[i]
desc=json.load(open('cache/content_desc_full.json'))['desc']
def parse(line):
    l=re.sub(r'^\d+\.\s*','',line); l=re.sub(r'\s*\[[^\]]*\]\s*$','',l)
    if ' by ' in l:
        nm,ar=l.rsplit(' by ',1); return (nm.strip().lower(), ar.strip().lower())
    return None
IN='models/_sft_dataset_cache/sft_ctx1024_60000_top200_firstpos.parquet'
OUT='models/_sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet'
df=pd.read_parquet(IN)
assert 'turn_number' in df.columns, "turn_number is missing (regenerate the dataset with build_reranker_data.py)"
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
df.to_parquet(OUT)
print(f"enriched {matched}/{tot} candidates ({100*matched/tot:.0f}%) | n={len(df)} | cols={list(df.columns)}",flush=True)
print("distribution turn_number:\n", df['turn_number'].value_counts().sort_index().to_string(), flush=True)
print(f"-> {OUT}", flush=True)
