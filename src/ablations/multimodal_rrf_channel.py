"""Diagnostic: (1) recall breakdown by RRF channel (BM25/dense/co-occ/artist
+ unique contributions), (2) re-ranking of the top-200 by dense ctx1024 cosine.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/multimodal_rrf_channel.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GPU','0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings, json
from collections import defaultdict, Counter
import numpy as np, pandas as pd, torch, bm25s
from pathlib import Path
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

DATA=Path('data'); MODEL=Path('models/qwen3_ft_dualencoder_4b_ctx1024')
TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
RRF_K=60; N_CHAN=500; KS=[20,50,500]
train=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
dev=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}
lk=tm.set_index('track_id'); aid=tm['artist_id'].astype(str).values; N=len(tids)
a2t=defaultdict(list)
for i,a in enumerate(aid): a2t[a].append(i)
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
def dense_text(r):
    p=[f"track_name: {r['track_name']}",f"artist_name: {r['artist_name']}",f"album_name: {r['album_name']}"]
    if isinstance(r['tag_list'],(list,np.ndarray)) and len(r['tag_list'])>0: p.append(f"tags: {', '.join(r['tag_list'])}")
    return '\n'.join(p)
print('co-occ...',flush=True)
cooc=defaultdict(Counter)
for _,s in train.iterrows():
    seq=[tidx[t['content']] for t in s['conversations'] if t['role']=='music' and t['content'] in tidx]; u=set(seq)
    for a in u:
        for b in u:
            if a!=b: cooc[a][b]+=1

import glob
print('loading provided embeddings (audio/lyrics/cf)...',flush=True)
emb_df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet'))],ignore_index=True)
emb_df=emb_df.set_index('track_id')
def mat(col):
    dim=0
    for row in emb_df[col]:
        if isinstance(row,(list,np.ndarray)) and len(row)>0: dim=len(row); break
    M=np.zeros((N,dim),dtype=np.float32); present=np.zeros(N,dtype=bool)
    for tid,row in emb_df[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim:
            M[tidx[tid]]=np.asarray(row,dtype=np.float32); present[tidx[tid]]=True
    M=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
    print(f'  {col}: dim={dim}, present={present.sum()}/{N}',flush=True)
    return torch.tensor(M,device='cuda'), present
AUD,_=mat('audio-laion_clap'); LYR,_=mat('lyrics-qwen3_embedding_0.6b'); CF,_=mat('cf-bpr')
print('embeddings loaded',flush=True)
def nn_channel(M, hist, topn=N_CHAN):
    if not hist: return np.array([],dtype=np.int64)
    q=M[hist].mean(0); q=q/(q.norm()+1e-9)
    sc=(M@q).float().cpu().numpy()
    return np.argsort(-sc)[:topn]

def cooc_rank(hist,topn=N_CHAN):
    agg=Counter()
    for h in hist:
        if h in cooc: agg.update(cooc[h])
    for h in hist: agg.pop(h,None)
    return [i for i,_ in agg.most_common(topn)]
items=[]
for _,s in dev.iterrows():
    gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
    for tn in range(1,9):
        gt=gtbt.get(tn)
        if gt is None or gt not in tidx: continue
        hist=[tidx[t['content']] for t in s['conversations'] if t['role']=='music' and t['turn_number']<tn and t['content'] in tidx]
        items.append({'q':conv_text(s['conversations'],tn),'gt':tidx[gt],'hist':hist,
                      'pa':{aid[h] for h in hist}})
Q=len(items); print(f'{Q} items',flush=True)
print('BM25...',flush=True)
corpus=[dense_text(r) for _,r in tm.iterrows()]
retr=bm25s.BM25(); retr.index(bm25s.tokenize(corpus,show_progress=False))
print('dense encode (ctx1024, seq1024 left)...',flush=True)
ft=SentenceTransformer(str(MODEL),device='cuda',model_kwargs={'torch_dtype':torch.bfloat16})
ft.max_seq_length=1024; ft.tokenizer.truncation_side='left'; ft.eval()
fe=ft.encode(corpus,batch_size=16,show_progress_bar=False,normalize_embeddings=True,device='cuda',convert_to_tensor=True).to(torch.bfloat16)
qs=[it['q'] for it in items]
bm=retr.retrieve(bm25s.tokenize([q.lower() for q in qs],show_progress=False),k=N_CHAN,return_as='tuple')
qe=ft.encode([f'Instruct: {TASK}\nQuery: {q.lower()}' for q in qs],batch_size=16,show_progress_bar=False,normalize_embeddings=True,device='cuda',convert_to_tensor=True).to(torch.bfloat16)
def rrf(chs):
    sc=np.zeros(N,dtype=np.float32)
    for lst in chs:
        for r,idx in enumerate(lst): sc[idx]+=1.0/(RRF_K+r+1)
    return np.argsort(-sc)[:N_CHAN]

chan_hit={c:{k:0 for k in KS} for c in ['bm25','dense','cooc','artist']}
uniq={c:0 for c in ['bm25','dense','cooc','artist']}      # GT found (<=500) by THIS channel ALONE.
comb_hit={k:0 for k in KS}; aud_hit={k:0 for k in KS}; all_hit={k:0 for k in KS}; dense_re_hit={k:0 for k in KS}; ndcg_rrf=0.0; ndcg_aud=0.0; ndcg_all=0.0; ndcg_dense=0.0
for i,it in enumerate(items):
    gt=it['gt']
    bm_l=np.asarray(bm.documents[i],dtype=np.int64)
    dn_l=torch.topk((qe[i]@fe.T).float(),N_CHAN).indices.cpu().numpy()
    co=np.array(cooc_rank(it['hist']),dtype=np.int64) if it['hist'] else np.array([],dtype=np.int64)
    art=np.array(sorted({t for a in it['pa'] for t in a2t.get(a,[])}),dtype=np.int64) if it['hist'] else np.array([],dtype=np.int64)
    if len(art):
        asc=(qe[i]@fe[art].T).float().cpu().numpy(); art=art[np.argsort(-asc)]
    chans={'bm25':bm_l,'dense':dn_l,'cooc':co,'artist':art}
    found={}
    for c,l in chans.items():
        ll=list(l[:N_CHAN])
        for k in KS: chan_hit[c][k]+= gt in ll[:k]
        found[c]= gt in ll
    # Unique contribution: GT found (<=500) by c and by NO other channel.
    for c in chans:
        if found[c] and not any(found[o] for o in chans if o!=c): uniq[c]+=1
    base=[bm_l,dn_l]+([co] if len(co) else [])+([art] if len(art) else [])
    aud=nn_channel(AUD,it['hist']); lyr=nn_channel(LYR,it['hist']); cf=nn_channel(CF,it['hist'])
    combl=list(rrf(base))
    comb_aud=list(rrf(base+([aud] if len(aud) else [])))
    comb_all=list(rrf(base+[x for x in [aud,lyr,cf] if len(x)]))
    for k in KS: comb_hit[k]+= gt in combl[:k]
    if gt in combl[:20]: ndcg_rrf+=1.0/np.log2(combl.index(gt)+2)
    if gt in comb_aud[:20]: ndcg_aud+=1.0/np.log2(comb_aud.index(gt)+2)
    if gt in comb_all[:20]: ndcg_all+=1.0/np.log2(comb_all.index(gt)+2)
    for k in KS:
        aud_hit[k]+= gt in comb_aud[:k]; all_hit[k]+= gt in comb_all[:k]
    # re-rank the top-200 by dense cosine


print('\n'+'='*70)
print(f'{"channel":<10}'+''.join(f'R@{k:<6}' for k in KS)+'  uniq(GT only, <=500)')
for c in ['bm25','dense','cooc','artist']:
    print(f'{c:<10}'+''.join(f'{100*chan_hit[c][k]/Q:>6.1f} ' for k in KS)+f'   {100*uniq[c]/Q:>5.1f}%')
print('-'*70)
print(f'{"COMBINED":<10}'+''.join(f'{100*comb_hit[k]/Q:>6.1f} ' for k in KS))
print('='*70)
print(f'{"RRF variant":<24}{"R@20":>7}{"R@50":>7}{"R@500":>7}{"nDCG@20 raw":>14}')
print(f'{"base (4 channels)":<24}{100*comb_hit[20]/Q:>7.1f}{100*comb_hit[50]/Q:>7.1f}{100*comb_hit[500]/Q:>7.1f}{ndcg_rrf/Q:>14.4f}')
print(f'{"+ audio":<24}{100*aud_hit[20]/Q:>7.1f}{100*aud_hit[50]/Q:>7.1f}{100*aud_hit[500]/Q:>7.1f}{ndcg_aud/Q:>14.4f}')
print(f'{"+ audio+lyrics+cf":<24}{100*all_hit[20]/Q:>7.1f}{100*all_hit[50]/Q:>7.1f}{100*all_hit[500]/Q:>7.1f}{ndcg_all/Q:>14.4f}')
print('(raw nDCG = RRF order without reranker; the reranker adds about +0.05)')
