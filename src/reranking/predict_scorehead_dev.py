"""SCORING-HEAD reranker picks on a dev (or Blind-B) shard. For each turn, use one forward pass,
read every candidate score at its marker, and emit both the argmax pick (TOP-1 mode) and the GT rank
in the COMPLETE score ranking (TOP-20 mode). Output: exp/picks/picks_<name>_<shard>.parquet.
Columns: (key,gt_pos,n_cand,pick,gt_rank_full).

    CUDA_VISIBLE_DEVICES=0 python src/reranking/predict_scorehead_dev.py --path models/llama32_3b_scorehead_ctx1024 \
        --name scorehead --shard 0 --nshards 4 --maxlen 4096
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import json, argparse, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
warnings.filterwarnings('ignore')
ap=argparse.ArgumentParser()
ap.add_argument('--path',required=True); ap.add_argument('--name',default='scorehead')
ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
ap.add_argument('--n_cand',type=int,default=50); ap.add_argument('--maxlen',type=int,default=4096)
ap.add_argument('--pool',default='exp/combined_pool_ctx1024_dev.parquet')
ap.add_argument('--dataset',default='TalkPlayData-Challenge-Dataset')
ap.add_argument('--no_goal',action='store_true',help='hide conversation_goal (simulate Blind-B where goal=null)')
args=ap.parse_args()
DATA=Path('data'); OUT=Path('exp/picks'); OUT.mkdir(parents=True,exist_ok=True)
SYS=("You are an expert music recommender. You get a user profile, a conversation goal, the conversation history "
     "ending with the user's CURRENT request, and candidate tracks. Each candidate is annotated: "
     "[community tags] {sound: genre/feel inferred from the AUDIO | themes: lyrical themes} and ends with a score "
     "marker. RATE how well each candidate fits the user's CURRENT request. CRITICAL: honor what the user wants RIGHT "
     "NOW and what they explicitly REJECT — match the sound/genre/themes to their current intent; a track whose "
     "sound/themes contradict what they asked for (or said they do NOT want) must score low.")
def build_user(profile,goal,conv,marked,n):
    return (f"User profile: {profile}\nConversation goal: {goal}\n\nConversation:\n{conv}\n\n"
            f"Candidate tracks ({n} candidates, 1-based; each ends with a score marker):\n{marked}")
dev=pd.read_parquet(DATA/args.dataset/'data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; lk=tm.set_index('track_id'); sess={s['session_id']:s for _,s in dev.iterrows()}
DESC=json.load(open('cache/content_desc_full.json'))['desc']
def fmt_profile(up):
    if up is None: return ''
    return ', '.join(f'{k}={up.get(k)}' for k in ['age_group','country_name','gender','preferred_language','preferred_musical_culture'] if up.get(k))
def fmt_goal(g):
    if g is None: return ''
    return ', '.join(f'{k}={g.get(k)}' for k in ['category','specificity','listener_goal'] if g.get(k))
def short(t):
    if t not in lk.index: return t
    r=lk.loc[t]; return f"{r['track_name']} - {r['artist_name']}"
def tcard(i):  # Enriched candidate (rendered exactly as during training).
    r=tm.iloc[i]; s=f"{r['track_name']} by {r['artist_name']}"; tl=r['tag_list']
    if isinstance(tl,(list,np.ndarray)) and len(tl)>0: s+=f" [{', '.join(list(tl)[:5])}]"
    d=DESC.get(tids[i],'')
    if d: s+=f"  {{{d}}}"
    return s
def conv(cs,tt):
    L=[]
    for t in cs:
        if int(t['turn_number'])>=tt: break
        ro,co=t['role'],t['content']
        if ro=='music': ro,co='assistant_played',short(co)
        L.append(f'{ro}: {co}')
    for t in cs:
        if int(t['turn_number'])==tt and t['role']=='user': L.append(f"user (REQUEST): {t['content']}"); break
    return '\n'.join(L)
tok=AutoTokenizer.from_pretrained(args.path); tok.truncation_side='left'
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
ck=torch.load(Path(args.path)/'score_head.pt',map_location='cpu'); H=ck['hidden']; MARK_ID=ck['marker_id']
MARK_STR=tok.convert_ids_to_tokens(MARK_ID)  # Marker DERIVED from model (Llama: reserved_special_token_5; Qwen: box_end).
pool=pd.read_parquet(args.pool)
all_items=[]
for _,r in pool.iterrows():
    sid=r['session_id']; tn=int(r['turn']); s=sess[sid]
    gtbt={int(t['turn_number']):t['content'] for t in s['conversations'] if t['role']=='music'}
    gt=gtbt.get(tn)
    if gt is None or gt not in tidx: continue
    cand=json.loads(r['pool'])[:args.n_cand]; gt_idx=tidx[gt]; n=len(cand)
    marked='\n'.join(f'{k}. {tcard(i)}{MARK_STR}' for k,i in enumerate(cand,1))
    goaltxt='' if args.no_goal else fmt_goal(s.get('conversation_goal'))
    user=(f"User profile: {fmt_profile(s.get('user_profile'))}\nConversation goal: {goaltxt}\n\n"
          f"Conversation:\n{conv(s['conversations'],tn)}\n\n"
          f"Candidate tracks ({n} candidates, 1-based; each ends with a score marker):\n{marked}")
    all_items.append({'key':f'{sid}|{tn}','gt_pos':cand.index(gt_idx) if gt_idx in cand else -1,'n_cand':n,'user':user})
items=all_items[args.shard::args.nshards]
print(f'[{args.name} {args.shard}/{args.nshards}] {len(items)}/{len(all_items)} | marker={MARK_STR} (id {MARK_ID})',flush=True)
base=AutoModel.from_pretrained(args.path,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
head=nn.Linear(H,1,bias=False).to(torch.bfloat16).to('cuda:0'); head.load_state_dict(ck['head']); head.eval()
rows=[]
with torch.no_grad():
    for it in tqdm(items,desc=f'{args.name}{args.shard}'):
        n=it['n_cand']
        text=tok.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':it['user']}],
                                     tokenize=False,add_generation_prompt=False)
        enc=tok(text,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=args.maxlen).to('cuda:0')
        h=base(**enc).last_hidden_state[0]  # [L,H]
        pos=(enc['input_ids'][0]==MARK_ID).nonzero().squeeze(-1)
        count=int(pos.numel()); base_idx=n-count  # survivors = candidates [base_idx, n)
        gscore=np.full(n,-1e30,dtype=np.float64)
        if count>0:
            s=head(h[pos]).squeeze(-1).float().cpu().numpy(); gscore[base_idx:n]=s
        pick=int(np.argmax(gscore))
        # Complete ranking: survivors sorted by descending score, then truncated items in pool order.
        surv_order=list((np.argsort(-gscore[base_idx:n])+base_idx)) if count>0 else []
        full=surv_order+list(range(0,base_idx))
        gp=it['gt_pos']; gt_rank_full=(full.index(gp)+1) if gp in range(n) else n
        rows.append({'key':it['key'],'gt_pos':gp,'n_cand':n,'pick':pick if 0<=pick<n else -1,'gt_rank_full':gt_rank_full})
pd.DataFrame(rows).to_parquet(OUT/f'picks_{args.name}_{args.shard}.parquet')
print(f'[{args.name} {args.shard}] saved {len(rows)}',flush=True)
