"""Past-event hallucination pass over the Blind-B submission.
Phase 1 (gemma-4-E2B detector): given the ACTUALLY played titles for each response,
  flag claims that the user liked/heard/found an ABSENT track or artist (not a
  simple comparison "fans of X").
Phase 2 (gemma-3n correction): regenerate flagged responses with a strict prompt requiring the exact
  title and ONLY listed played titles; select a candidate that names the recommendation without external titles.
Merge into the submission. track_ids remain unchanged.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/hallucination_detector_llm.py
"""
import os, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
warnings.filterwarnings('ignore'); torch.manual_seed(0)
SUB='exp/inference/blindset_A/firstpos_top50_ctx1024_blindB_convbestofN.json'
B='data/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet'
b=pd.read_parquet(B); tm=pd.read_parquet('data/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
lk=tm.set_index('track_id'); sess={s['session_id']:s for _,s in b.iterrows()}
norm=lambda s:re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()
def nm(t): return f"{lk.loc[t,'track_name']} by {lk.loc[t,'artist_name']}" if t in lk.index else str(t)
def tg(t):
    if t not in lk.index: return ''
    tl=lk.loc[t,'tag_list']; return ', '.join(list(tl)[:6]) if isinstance(tl,(list,np.ndarray)) and len(tl)>0 else ''
def played(e):
    sid=e['session_id']; tt=int(e['turn_number'])
    return [t['content'] for t in sess[sid]['conversations'] if t['role']=='music' and int(t['turn_number'])<tt and t['content'] in lk.index]
def full_conv(cs,tt):
    o={'user':0,'music':1,'assistant':2}; L=[]
    for t in sorted(cs,key=lambda x:(int(x['turn_number']),o[x['role']])):
        if int(t['turn_number'])>tt: break
        if int(t['turn_number'])==tt and t['role']!='user': continue
        if t['role']=='user': L.append(f"User: {t['content']}")
        elif t['role']=='music': L.append(f"Assistant played: {nm(t['content'])}")
        elif t['role']=='assistant': L.append(f"Assistant: {t['content']}")
    return '\n'.join(L)
d=json.load(open(SUB))

# Phase 1: gemma-4-E2B detector.
J='google/gemma-4-E2B-it'
jt=AutoTokenizer.from_pretrained(J); jt.padding_side='left'; jt.truncation_side='left'
if jt.pad_token_id is None: jt.pad_token=jt.eos_token
jm=AutoModelForCausalLM.from_pretrained(J,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
DSYS=("You verify a music-chat reply for FALSE MEMORY claims. You are given the tracks ALREADY PLAYED in the "
      "conversation and the assistant's reply. Flag ONLY if the reply states or implies the user previously "
      "liked / heard / found / played a SPECIFIC named track or artist that is NOT in the played list. "
      "Generic comparisons ('fans of X', 'similar to Y', describing the recommended track) are FINE. "
      "Answer EXACTLY 'NO' if no false claim, else 'YES'.")
@torch.no_grad()
def detect(batch):
    txts=[]
    for e in batch:
        pl=played(e); plist='\n'.join(f"- {nm(t)}" for t in pl) if pl else "(none)"
        u=f"Tracks already played in this conversation:\n{plist}\n\nAssistant's reply:\n{e['predicted_response']}\n\nFalse memory claim? (YES/NO):"
        txts.append(jt.apply_chat_template([{'role':'system','content':DSYS},{'role':'user','content':u}],tokenize=False,add_generation_prompt=True))
    enc=jt(txts,return_tensors='pt',truncation=True,max_length=3072,padding=True).to('cuda:0')
    out=jm.generate(**enc,max_new_tokens=4,do_sample=False,pad_token_id=jt.eos_token_id); L=enc['input_ids'].shape[1]
    return [jt.decode(out[j,L:],skip_special_tokens=True).strip().lower().startswith('yes') for j in range(len(batch))]
print('Phase 1: gemma-4 detection...', flush=True)
flag=[]
for i in tqdm(range(0,len(d),8)):
    sub=d[i:i+8]
    for e,f in zip(sub,detect(sub)):
        if f: flag.append(e)
print(f"{len(flag)} responses flagged (potential past-event hallucination)", flush=True)
for e in flag: print("   -",e['session_id'][:8],":",e['predicted_response'][:110], flush=True)
del jm; torch.cuda.empty_cache()

# ---------- Phase 2: gemma-3n correction ----------
if flag:
    G='google/gemma-3n-E4B-it'
    gt=AutoTokenizer.from_pretrained(G); gt.padding_side='left'; gt.truncation_side='left'
    if gt.pad_token_id is None: gt.pad_token=gt.eos_token
    gm=AutoModelForCausalLM.from_pretrained(G,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
    print('Phase 2: gemma-3n correction...', flush=True)
    for e in flag:
        tid=e['predicted_track_ids'][0]; tt=int(e['turn_number']); pl=played(e)
        allow=set([norm(lk.loc[tid,'track_name'])]+[norm(lk.loc[t,'track_name']) for t in pl])
        plist=', '.join(f'"{lk.loc[t,"track_name"]}"' for t in pl) if pl else "none"
        SYS=(f'You are the assistant in a music chat. Recommend EXACTLY this track: "{nm(tid)}". '
             f'You may ONLY reference tracks the user actually heard in this conversation: {plist}. '
             'NEVER claim the user liked/heard/found any other specific track or artist. '
             f'Write ~40 words, warm, naming "{lk.loc[tid,"track_name"]}" explicitly and explaining why it fits.')
        usr=f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'],tt)}\n\nRecommend this exact track: {nm(tid)}"+(f" (tags: {tg(tid)})" if tg(tid) else '')
        txt=gt.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':usr}],tokenize=False,add_generation_prompt=True)
        enc=gt(txt,return_tensors='pt',truncation=True,max_length=3072).to('cuda:0')
        with torch.no_grad():
            out=gm.generate(**enc,max_new_tokens=90,do_sample=True,temperature=0.7,top_p=0.9,num_return_sequences=8,pad_token_id=gt.eos_token_id)
        cand=[gt.decode(o[enc['input_ids'].shape[1]:],skip_special_tokens=True).strip().replace('\n',' ') for o in out]
        QUOTE=re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”')
        def clean(c):
            cn=norm(c); tnn=norm(lk.loc[tid,'track_name'])
            names_track = tnn in cn or ' '.join(tnn.split()[:3]) in cn
            ext=[q.group(1) or q.group(2) for q in QUOTE.finditer(c)]
            no_ext=all(any(norm(x)==a or norm(x) in a or a in norm(x) for a in allow if a) for x in ext) if ext else True
            return names_track and no_ext
        good=[c for c in cand if clean(c)]
        e['predicted_response']=good[0] if good else next((c for c in cand if norm(lk.loc[tid,'track_name']) in norm(c)), cand[0])
        print(f"   {e['session_id'][:8]}: {'cleanly CORRECTED' if good else 'best effort'} -> {e['predicted_response'][:120]}", flush=True)
    json.dump(d, open(SUB,'w'), ensure_ascii=False, indent=2)
    print("merged ->", SUB, flush=True)
else:
    print("nothing to correct.", flush=True)
