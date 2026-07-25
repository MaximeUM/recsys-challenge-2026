"""Dev predictions with the cross-encoder (approach A) -> official evaluator format.

Score the 50 candidates in the combined dev pool, rank them, and retain the top 20. Then:
    cd music-crs-evaluator && python evaluate_devset.py --tid crossenc

Usage :
    CUDA_VISIBLE_DEVICES=0 python src/ablations/crossencoder_predict.py --model models/crossenc_reranker --tid crossenc
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, argparse, warnings
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='models/crossenc_reranker')
ap.add_argument('--pool', default='exp/combined_pool_ctx1024_dev.parquet')
ap.add_argument('--tid', default='crossenc')
args = ap.parse_args()

DATA = Path('data'); OUT = Path('music-crs-evaluator/exp/inference/devset'); OUT.mkdir(parents=True, exist_ok=True)
N_CAND, TOPK, MAXLEN = 50, 20, 384

dev = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm  = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c] = tm[c].apply(lambda x: x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
track_ids = tm['track_id'].tolist(); lk = tm.set_index('track_id')
sess = {s['session_id']: s for _, s in dev.iterrows()}

def fmt_profile(up):
    if up is None: return ''
    return ', '.join(f'{k}={up.get(k)}' for k in ['age_group','country_name','gender','preferred_language','preferred_musical_culture'] if up.get(k))
def fmt_goal(g):
    if g is None: return ''
    return ', '.join(f'{k}={g.get(k)}' for k in ['category','specificity','listener_goal'] if g.get(k))
def short(t):
    if t not in lk.index: return t
    r = lk.loc[t]; return f"{r['track_name']} - {r['artist_name']}"
def passage(i):
    r = tm.iloc[i]; p = f"{r['track_name']} by {r['artist_name']}"
    if isinstance(r['tag_list'],(list,np.ndarray)) and len(r['tag_list'])>0: p += f" [{', '.join(list(r['tag_list'])[:5])}]"
    return p
def conv(cs, tt):
    L=[]
    for t in cs:
        if t['turn_number']>=tt: break
        ro,co=t['role'],t['content']
        if ro=='music': ro,co='assistant_played',short(co)
        L.append(f'{ro}: {co}')
    for t in cs:
        if t['turn_number']==tt and t['role']=='user': L.append(f"user (REQUEST): {t['content']}"); break
    return '\n'.join(L)
def build_query(up, goal, c):
    h=[]
    if up: h.append(f"User profile: {up}")
    if goal: h.append(f"Goal: {goal}")
    return ('\n'.join(h)+'\n' if h else '') + f"Conversation:\n{c}"

pool = pd.read_parquet(args.pool)
print(f'{len(pool)} dev turns | model={args.model}', flush=True)
tok = AutoTokenizer.from_pretrained(args.model); tok.truncation_side = 'left'
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, torch_dtype=torch.bfloat16).to('cuda:0').eval()

@torch.no_grad()
def score_rank(q, cand):
    enc = tok([q]*len(cand), [passage(i) for i in cand], padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt').to('cuda:0')
    sc = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    order = np.argsort(-sc)
    return [int(cand[o]) for o in order]

subs = []
for _, r in tqdm(pool.iterrows(), total=len(pool)):
    s = sess[r['session_id']]; tn = int(r['turn']); cand = json.loads(r['pool'])[:N_CAND]
    q = build_query(fmt_profile(s.get('user_profile')), fmt_goal(s.get('conversation_goal')), conv(s['conversations'], tn))
    ranked = score_rank(q, cand)[:TOPK]
    subs.append({'session_id': r['session_id'], 'user_id': s['user_id'], 'turn_number': tn,
                 'predicted_track_ids': [track_ids[i] for i in ranked], 'predicted_response': ''})

with open(OUT / f'{args.tid}.json', 'w') as f: json.dump(subs, f)
print(f'Saved {len(subs)} -> {OUT / (args.tid + ".json")}', flush=True)
