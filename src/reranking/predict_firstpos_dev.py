"""Picks of the INTENT+MULTIMODAL reranker on one dev shard. ENRICHED candidates (sound|themes) +
intent prompt (identical to the intent+mm trainer). Output exp/picks/picks_<name>_<shard>.parquet (key,gt_pos,n_cand,pick).

    CUDA_VISIBLE_DEVICES=0 python src/reranking/predict_firstpos_dev.py --path models/llama32_3b_intent_mm_ctx1024 --name intentmm --shard 0 --nshards 4
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import json, re, argparse, warnings
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')
ap=argparse.ArgumentParser()
ap.add_argument('--path',required=True); ap.add_argument('--name',required=True)
ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
ap.add_argument('--n_cand',type=int,default=50); ap.add_argument('--maxlen',type=int,default=3584)
args=ap.parse_args()
DATA=Path('data'); OUT=Path('exp/picks'); OUT.mkdir(parents=True,exist_ok=True)
dev=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
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
def tcard(i):  # Enriched candidate (same rendering as during training).
    r=tm.iloc[i]; s=f"{r['track_name']} by {r['artist_name']}"
    tl=r['tag_list']
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
SYS=("You are an expert music recommender. You get a user profile, a conversation goal, the conversation history "
     "ending with the user's CURRENT request, and candidate tracks. Each candidate is annotated: "
     "[community tags] {sound: genre/feel inferred from the AUDIO | themes: lyrical themes}. "
     "Pick THE single best track for the user's CURRENT request. CRITICAL: honor what the user wants RIGHT NOW and "
     "what they explicitly REJECT — match the sound/genre/themes to their current intent; NEVER pick a track whose "
     "sound/themes contradict what they asked for or said they do NOT want. "
     "Output ONLY a JSON array with that one candidate index (1-based), e.g. [12]. Nothing else.")
pool=pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet')
all_items=[]
for _,r in pool.iterrows():
    sid=r['session_id']; tn=int(r['turn']); s=sess[sid]
    gtbt={int(t['turn_number']):t['content'] for t in s['conversations'] if t['role']=='music'}
    gt=gtbt.get(tn)
    if gt is None or gt not in tidx: continue
    cand=json.loads(r['pool'])[:args.n_cand]; gt_idx=tidx[gt]
    lines=[f'{k}. {tcard(i)}' for k,i in enumerate(cand,1)]
    user=(f"User profile: {fmt_profile(s.get('user_profile'))}\nConversation goal: {fmt_goal(s.get('conversation_goal'))}\n\n"
          f"Conversation:\n{conv(s['conversations'],tn)}\n\nCandidate tracks ({len(cand)} candidates, 1-based; each has [tags] {{sound | themes}}):\n"+
          '\n'.join(lines)+"\n\nPick the single best track for the user's CURRENT request, respecting what they want and reject. Output JSON array with one index.")
    all_items.append({'key':f'{sid}|{tn}','gt_pos':cand.index(gt_idx) if gt_idx in cand else -1,'n_cand':len(cand),'user':user})
items=all_items[args.shard::args.nshards]
print(f'[{args.name} {args.shard}/{args.nshards}] {len(items)}/{len(all_items)}',flush=True)
tok=AutoTokenizer.from_pretrained(args.path); tok.truncation_side='left'
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
llm=AutoModelForCausalLM.from_pretrained(args.path,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
PICK=re.compile(r'\[\s*(\d+)'); rows=[]
with torch.no_grad():
    for it in tqdm(items,desc=f'{args.name}{args.shard}'):
        text=tok.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':it['user']}],tokenize=False,add_generation_prompt=True)
        enc=tok(text,return_tensors='pt',truncation=True,max_length=args.maxlen).to('cuda:0')
        out=llm.generate(**enc,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
        g=tok.decode(out[0,enc['input_ids'].shape[1]:],skip_special_tokens=True); m=PICK.search(g); p=int(m.group(1))-1 if m else -1
        rows.append({'key':it['key'],'gt_pos':it['gt_pos'],'n_cand':it['n_cand'],'pick':p if 0<=p<it['n_cand'] else -1})
pd.DataFrame(rows).to_parquet(OUT/f'picks_{args.name}_{args.shard}.parquet')
print(f'[{args.name} {args.shard}] saved {len(rows)}',flush=True)
