"""Blind-A submission with the CROSS-ENCODER (approach A).

Combined pool (BM25+dense+co-occ+artist, played tracks excluded) -> fine-tuned
cross-encoder scoring -> top 20. Responses are generated separately (same generation as convbestofN).

Run: CUDA_VISIBLE_DEVICES=0 python src/ablations/crossencoder_blind.py
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, warnings
from collections import defaultdict, Counter
import numpy as np, pandas as pd, torch, bm25s
from pathlib import Path
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
warnings.filterwarnings('ignore')

DATA = Path('data'); MODEL_FT_DIR = Path('models/qwen3_ft_dualencoder_v1_4gpu')
CE = 'models/crossenc_reranker'
DEVICE='cuda'; TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
RRF_K=60; N_RAG_TOP=50; TOPK=20; MARGIN=40; MAXLEN=384
OUT = Path('exp/inference/blindset_A/crossenc_blindA.json')

blind = pd.read_parquet(DATA/'TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet')
train = pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c]=tm[c].apply(lambda x: x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; lk=tm.set_index('track_id')
artist_id=tm['artist_id'].astype(str).values
track_key=(tm['track_name'].str.lower().str.strip()+' || '+tm['artist_name'].str.lower().str.strip()).values
a2t=defaultdict(list)
for i,a in enumerate(artist_id): a2t[a].append(i)

def short(t):
    if t not in lk.index: return t
    r=lk.loc[t]; return f"{r['track_name']} - {r['artist_name']}"
def passage(i):
    r=tm.iloc[i]; p=f"{r['track_name']} by {r['artist_name']}"
    if isinstance(r['tag_list'],(list,np.ndarray)) and len(r['tag_list'])>0: p+=f" [{', '.join(list(r['tag_list'])[:5])}]"
    return p
def fmt_profile(up):
    if up is None: return ''
    return ', '.join(f'{k}={up.get(k)}' for k in ['age_group','country_name','gender','preferred_language','preferred_musical_culture'] if up.get(k))
def fmt_goal(g):
    if g is None: return ''
    return ', '.join(f'{k}={g.get(k)}' for k in ['category','specificity','listener_goal'] if g.get(k))
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
def build_query(up,goal,c):
    h=[]
    if up: h.append(f"User profile: {up}")
    if goal: h.append(f"Goal: {goal}")
    return ('\n'.join(h)+'\n' if h else '')+f"Conversation:\n{c}"

# co-occ
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
print(f'{len(items)} sessions blind-A',flush=True)

# BM25 + dense
corpus=[]
for _,r in tm.iterrows():
    p=[f"track_name: {r['track_name']}",f"artist_name: {r['artist_name']}",f"album_name: {r['album_name']}"]
    if isinstance(r['tag_list'],(list,np.ndarray)): p.append(f"tags: {', '.join(r['tag_list'])}")
    corpus.append('\n'.join(p))
retr=bm25s.BM25(); retr.index(bm25s.tokenize(corpus,show_progress=False))
ft=SentenceTransformer(str(MODEL_FT_DIR),device=DEVICE,model_kwargs={'torch_dtype':torch.bfloat16}); ft.max_seq_length=256; ft.eval()
fe=ft.encode([dtext(r) for _,r in tm.iterrows()],batch_size=128,show_progress_bar=False,normalize_embeddings=True,device=DEVICE,convert_to_tensor=True).to(torch.bfloat16)
qs=[it['q'] for it in items]
bm=retr.retrieve(bm25s.tokenize([q.lower() for q in qs],show_progress=False),k=200,return_as='tuple')
qe=ft.encode([f'Instruct: {TASK}\nQuery: {q.lower()}' for q in qs],batch_size=32,show_progress_bar=False,normalize_embeddings=True,device=DEVICE,convert_to_tensor=True).to(torch.bfloat16)

def rrf_topk(channels,k):
    sc=defaultdict(float)
    for lst in channels:
        for r,idx in enumerate(lst): sc[int(idx)]+=1.0/(RRF_K+r+1)
    return [i for i,_ in sorted(sc.items(),key=lambda x:-x[1])][:k]

pools=[]
for i,it in enumerate(items):
    chans=[np.asarray(bm.documents[i],dtype=np.int64), torch.topk((qe[i]@fe.T).float(),200).indices.cpu().numpy()]
    if it['hist']:
        agg=Counter()
        for h in it['hist']:
            if h in cooc: agg.update(cooc[h])
        for h in it['hist']: agg.pop(h,None)
        co=[idx for idx,v in agg.most_common() if v>0][:200]
        if co: chans.append(np.array(co,dtype=np.int64))
        art=sorted({t for h in it['hist'] for t in a2t.get(artist_id[h],[])})
        if art:
            art=np.array(art,dtype=np.int64); asc=(qe[i]@fe[art].T).float().cpu().numpy()
            chans.append(art[np.argsort(-asc)][:200])
    cand=rrf_topk(chans,N_RAG_TOP+MARGIN)
    pk={track_key[h] for h in it['hist']}; ph=set(it['hist'])
    pools.append([c for c in cand if c not in ph and track_key[c] not in pk][:N_RAG_TOP])
del ft,fe; torch.cuda.empty_cache()

# cross-encoder scoring
tok=AutoTokenizer.from_pretrained(CE); tok.truncation_side='left'
model=AutoModelForSequenceClassification.from_pretrained(CE,num_labels=1,torch_dtype=torch.bfloat16).to('cuda:0').eval()
@torch.no_grad()
def rank(q,cand):
    enc=tok([q]*len(cand),[passage(i) for i in cand],padding=True,truncation=True,max_length=MAXLEN,return_tensors='pt').to('cuda:0')
    sc=model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return [int(cand[o]) for o in np.argsort(-sc)]

subs=[]
for i,it in enumerate(tqdm(items)):
    s=it['session']
    q=build_query(fmt_profile(s.get('user_profile')),fmt_goal(s.get('conversation_goal')),conv(s['conversations'],it['tt']))
    final=rank(q,pools[i])[:TOPK]
    subs.append({'session_id':s['session_id'],'user_id':s['user_id'],'turn_number':it['tt'],
                 'predicted_track_ids':[tids[i] for i in final],'predicted_response':''})
vts=set(tids); ok=all(len(e['predicted_track_ids'])==TOPK and len(set(e['predicted_track_ids']))==TOPK and all(t in vts for t in e['predicted_track_ids']) for e in subs)
print(f'{len(subs)} entries | valid: {ok}',flush=True)
with open(OUT,'w') as f: json.dump(subs,f,ensure_ascii=False, indent=2)
print(f'Saved -> {OUT}',flush=True)
