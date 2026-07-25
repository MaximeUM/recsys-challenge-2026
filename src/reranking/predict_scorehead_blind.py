"""Blind-B submission — SCORING-HEAD reranker (Qwen-8B top 200, best scorehead, dev TOP-20=0.2186).
Qwen-8B variant (vs Llama-3B): only MODEL + OUT change. Scores every
pool candidate, rank by descending score -> predicted_track_ids = TOP-20 (position 1 = highest score).
predicted_response is empty (filled later by convbestofN). Marker derived from score_head (Qwen: box_end 151649).

    CUDA_VISIBLE_DEVICES=0 python src/reranking/predict_scorehead_blind.py
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
warnings.filterwarnings('ignore')
DATA=Path('data'); MODEL='models/qwen3_8b_scorehead_top200'; N_CAND=200; MAXLEN=14336; TOPK=20
SYS=("You are an expert music recommender. You get a user profile, a conversation goal, the conversation history "
     "ending with the user's CURRENT request, and candidate tracks. Each candidate is annotated: "
     "[community tags] {sound: genre/feel inferred from the AUDIO | themes: lyrical themes} and ends with a score "
     "marker. RATE how well each candidate fits the user's CURRENT request. CRITICAL: honor what the user wants RIGHT "
     "NOW and what they explicitly REJECT — match the sound/genre/themes to their current intent; a track whose "
     "sound/themes contradict what they asked for (or said they do NOT want) must score low.")
blind=pd.read_parquet(DATA/'TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; lk=tm.set_index('track_id'); sess={s['session_id']:s for _,s in blind.iterrows()}
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
def tcard(i):
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
pool=pd.read_parquet('exp/combined_pool_ctx1024_blindB.parquet')
tok=AutoTokenizer.from_pretrained(MODEL); tok.truncation_side='left'
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
ck=torch.load(Path(MODEL)/'score_head.pt',map_location='cpu'); H=ck['hidden']; MARK_ID=ck['marker_id']
MARK_STR=tok.convert_ids_to_tokens(MARK_ID)
# DEVICE_MAP='auto' distributes the model across GPUs when it does not fit on
# one card (8B in bf16 = ~16 GB). The unchanged default is cuda:0.
DEVMAP=os.environ.get('DEVICE_MAP','cuda:0')
base=AutoModel.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map=DEVMAP).eval()
DEV=next(base.parameters()).device if DEVMAP=='auto' else torch.device(DEVMAP)
head=nn.Linear(H,1,bias=False).to(torch.bfloat16); head.load_state_dict(ck['head']); head.eval()
print(f"marker={MARK_STR} (id {MARK_ID}) | {len(pool)} Blind-B turns | model {MODEL}",flush=True)
subs=[]
with torch.no_grad():
    for _,r in tqdm(pool.iterrows(),total=len(pool)):
        sid=r['session_id']; tn=int(r['turn']); s=sess[sid]; cand=json.loads(r['pool'])[:N_CAND]; n=len(cand)
        marked='\n'.join(f'{k}. {tcard(i)}{MARK_STR}' for k,i in enumerate(cand,1))
        user=(f"User profile: {fmt_profile(s.get('user_profile'))}\nConversation goal: {fmt_goal(s.get('conversation_goal'))}\n\n"
              f"Conversation:\n{conv(s['conversations'],tn)}\n\n"
              f"Candidate tracks ({n} candidates, 1-based; each ends with a score marker):\n{marked}")
        text=tok.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':user}],tokenize=False,add_generation_prompt=False)
        enc=tok(text,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=MAXLEN).to(DEV)
        h=base(**enc).last_hidden_state[0]; pos=(enc['input_ids'][0]==MARK_ID).nonzero().squeeze(-1)
        count=int(pos.numel()); base_idx=n-count
        gscore=np.full(n,-1e30);
        if count>0:
            hp=h[pos]
            if head.weight.device!=hp.device: head.to(hp.device)
            gscore[base_idx:n]=head(hp).squeeze(-1).float().cpu().numpy()
        order=list(np.argsort(-gscore))  # Pool candidates ranked by score (truncated items moved to the tail via -inf).
        track_order=[tids[cand[i]] for i in order][:TOPK]
        subs.append({'session_id':sid,'user_id':s['user_id'],'turn_number':tn,
                     'predicted_track_ids':track_order,'predicted_response':''})
OUT='exp/inference/blindset_B/firstpos_scorehead_qwen_blindB.json'
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(subs,open(OUT,'w'),ensure_ascii=False,indent=2)
print(f'Saved {len(subs)} -> {OUT}',flush=True)
