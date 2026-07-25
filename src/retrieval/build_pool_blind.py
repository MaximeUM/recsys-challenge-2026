"""combined Blind-A pool with the 4B retriever (top 200, played tracks excluded).

Build the pool with RRF (BM25+dense4B+co-occ+artist, excluding already-played
tracks), then save the per-session top-200 pool for reuse by
src/reranking/predict_firstpos.py (--mode blindA). Columns: session_id, turn, pool.

    SFT_GPU=1 python src/retrieval/build_pool_blind.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('SFT_GPU', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, warnings, argparse
from collections import defaultdict, Counter
import numpy as np, pandas as pd, torch, bm25s
from pathlib import Path
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

ENC_BS = int(os.environ.get('ENC_BATCH_SIZE', '16'))  # Lower this value when VRAM is limited.
DATA = Path('data'); MODEL_FT_DIR = Path('models/qwen3_ft_dualencoder_4b_ctx1024')
DEVICE='cuda'; TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
RRF_K=60; POOL_N=200; RETRIEVE_MARGIN=60
_ap=argparse.ArgumentParser()
_ap.add_argument('--blind_parquet', default='data/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet',
                 help='Blind-set Parquet file (for Blind-B, pass its path)')
_ap.add_argument('--out', default='exp/combined_pool_ctx1024_blindA.parquet')
_args=_ap.parse_args()
OUT = Path(_args.out)

blind = pd.read_parquet(_args.blind_parquet)
train = pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c]=tm[c].apply(lambda x: x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; lk=tm.set_index('track_id')
artist_id=tm['artist_id'].astype(str).values; N=len(tids)
track_key=(tm['track_name'].str.lower().str.strip()+' || '+tm['artist_name'].str.lower().str.strip()).values
a2t=defaultdict(list)
for i,a in enumerate(artist_id): a2t[a].append(i)

def short(t):
    if t not in lk.index: return t
    r=lk.loc[t]; return f"{r['track_name']} - {r['artist_name']}"
def conv(cs,tt):
    L=[]
    for t in cs:
        if t['turn_number']>=tt: break
        ro,co=t['role'],t['content']
        if ro=='music': ro,co='assistant_played',short(co)
        L.append(f'{ro}: {co}')
    for t in cs:
        if t['turn_number']==tt and t['role']=='user': L.append(f"user (REQUEST): {t['content']}"); break
    return '\n'.join(L)
def dtext(r):
    p=[f"track_name: {r['track_name']}",f"artist_name: {r['artist_name']}",f"album_name: {r['album_name']}"]
    if isinstance(r['tag_list'],(list,np.ndarray)) and len(r['tag_list'])>0: p.append(f"tags: {', '.join(r['tag_list'])}")
    return '\n'.join(p)

print('co-occ...',flush=True)
cooc=defaultdict(Counter)
for _,s in train.iterrows():
    seq=[tidx[t['content']] for t in s['conversations'] if t['role']=='music' and t['content'] in tidx]
    u=set(seq)
    for a in u:
        for b in u:
            if a!=b: cooc[a][b]+=1

items=[]
for _,s in blind.iterrows():
    cs=s['conversations']; tt=cs[-1]['turn_number']
    hist=[tidx[t['content']] for t in cs if t['role']=='music' and t['turn_number']<tt and t['content'] in tidx]
    items.append({'session':s,'tt':int(tt),'q':conv(cs,tt),'hist':hist})
print(f'{len(items)} sessions <- {_args.blind_parquet}',flush=True)

print('BM25+dense(4B)...',flush=True)
corpus=[]
for _,r in tm.iterrows():
    p=[f"track_name: {r['track_name']}",f"artist_name: {r['artist_name']}",f"album_name: {r['album_name']}"]
    if isinstance(r['tag_list'],(list,np.ndarray)): p.append(f"tags: {', '.join(r['tag_list'])}")
    corpus.append('\n'.join(p))
retr=bm25s.BM25(); retr.index(bm25s.tokenize(corpus,show_progress=False))
ft=SentenceTransformer(str(MODEL_FT_DIR),device=DEVICE,model_kwargs={'torch_dtype':torch.bfloat16}); ft.max_seq_length=1024; ft.tokenizer.truncation_side='left'; ft.eval()
fe=ft.encode([dtext(r) for _,r in tm.iterrows()],batch_size=ENC_BS,show_progress_bar=False,normalize_embeddings=True,device=DEVICE,convert_to_tensor=True).to(torch.bfloat16)
qs=[it['q'] for it in items]
bm=retr.retrieve(bm25s.tokenize([q.lower() for q in qs],show_progress=False),k=300,return_as='tuple')
qe=ft.encode([f'Instruct: {TASK}\nQuery: {q.lower()}' for q in qs],batch_size=ENC_BS,show_progress_bar=False,normalize_embeddings=True,device=DEVICE,convert_to_tensor=True).to(torch.bfloat16)

def rrf_topk(channels,k):
    sc=defaultdict(float)
    for lst in channels:
        for r,idx in enumerate(lst): sc[int(idx)]+=1.0/(RRF_K+r+1)
    return [i for i,_ in sorted(sc.items(),key=lambda x:-x[1])][:k]

rows=[]
for i,it in enumerate(items):
    bm_l=np.asarray(bm.documents[i],dtype=np.int64)
    dn_l=torch.topk((qe[i]@fe.T).float(),300).indices.cpu().numpy()
    chans=[bm_l,dn_l]
    if it['hist']:
        agg=Counter()
        for h in it['hist']:
            if h in cooc: agg.update(cooc[h])
        for h in it['hist']: agg.pop(h,None)
        co=[idx for idx,v in agg.most_common() if v>0][:300]
        if co: chans.append(np.array(co,dtype=np.int64))
        art=sorted({t for h in it['hist'] for t in a2t.get(artist_id[h],[])})
        if art:
            art=np.array(art,dtype=np.int64); asc=(qe[i]@fe[art].T).float().cpu().numpy()
            chans.append(art[np.argsort(-asc)][:300])
    cand=rrf_topk(chans,POOL_N+RETRIEVE_MARGIN)
    pkeys={track_key[h] for h in it['hist']}; phist=set(it['hist'])
    cand=[c for c in cand if c not in phist and track_key[c] not in pkeys][:POOL_N]
    rows.append({'session_id':it['session']['session_id'],'turn':it['tt'],'pool':json.dumps([int(x) for x in cand])})

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(OUT)
print(f'Saved {len(rows)} pools (top-{POOL_N}) -> {OUT}',flush=True)
