"""Test the past-event hallucination detector with the larger gemma-4-E4B and FULL grounding.
Use the entire conversation, including played titles and titles cited by the user. The detector must
answer NO or 'YES: <invented title>'. Then structurally verify whether that title appears in the
conversation to measure the detector's actual precision.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/hallucination_detector_e4b.py
"""
import os, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
SUB='exp/inference/blindset_A/firstpos_top50_ctx1024_blindB_convbestofN.json'
b=pd.read_parquet('data/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet')
tm=pd.read_parquet('data/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
lk=tm.set_index('track_id'); sess={s['session_id']:s for _,s in b.iterrows()}
norm=lambda s:re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()
def nm(t): return f"{lk.loc[t,'track_name']} by {lk.loc[t,'artist_name']}" if t in lk.index else str(t)
def full_conv(cs,tt):
    o={'user':0,'music':1,'assistant':2}; L=[]
    for t in sorted(cs,key=lambda x:(int(x['turn_number']),o[x['role']])):
        if int(t['turn_number'])>tt: break
        if int(t['turn_number'])==tt and t['role']!='user': continue
        if t['role']=='user': L.append(f"User: {t['content']}")
        elif t['role']=='music': L.append(f"Assistant played: {nm(t['content'])}")
        elif t['role']=='assistant': L.append(f"Assistant: {t['content']}")
    return '\n'.join(L)
def conv_text_norm(e):  # Entire conversation text for structural verification.
    cs=sess[e['session_id']]['conversations']; tt=int(e['turn_number']); parts=[]
    for t in cs:
        if int(t['turn_number'])>tt: break
        parts.append(nm(t['content']) if t['role']=='music' else str(t['content']))
    parts.append(nm(e['predicted_track_ids'][0]))  # + the recommended track
    return norm(' || '.join(parts))
d=json.load(open(SUB))
G='google/gemma-4-E4B-it'
t=AutoTokenizer.from_pretrained(G); t.padding_side='left'; t.truncation_side='left'
if t.pad_token_id is None: t.pad_token=t.eos_token
m=AutoModelForCausalLM.from_pretrained(G,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
DSYS=("You check a music-chat reply for FALSE MEMORY claims. Below is the FULL conversation (everything the user said "
      "and every track played) and the assistant's latest reply. Flag ONLY if the reply states the user previously "
      "liked / heard / found / played a SPECIFIC named track or artist that appears NOWHERE in the conversation above "
      "(not played, not mentioned by the user). Generic comparisons ('fans of X', 'similar to Y') are FINE. "
      "Answer 'NO' if there is no false claim. Otherwise answer 'YES: <the invented track or artist name>'.")
@torch.no_grad()
def detect(batch):
    txts=[t.apply_chat_template([{'role':'system','content':DSYS},
          {'role':'user','content':f"FULL conversation:\n{full_conv(sess[e['session_id']]['conversations'],int(e['turn_number']))}\n\nAssistant's latest reply:\n{e['predicted_response']}\n\nFalse memory claim?"}],
          tokenize=False,add_generation_prompt=True) for e in batch]
    enc=t(txts,return_tensors='pt',truncation=True,max_length=4096,padding=True).to('cuda:0')
    out=m.generate(**enc,max_new_tokens=24,do_sample=False,pad_token_id=t.eos_token_id); L=enc['input_ids'].shape[1]
    return [t.decode(out[j,L:],skip_special_tokens=True).strip() for j in range(len(batch))]
res=[]
for i in tqdm(range(0,len(d),4)):
    sub=d[i:i+4]
    for e,g in zip(sub,detect(sub)): res.append((e,g))
flag=[(e,g) for e,g in res if g.lower().startswith('yes')]
print(f"\n=== gemma-4-E4B (full grounding): {len(flag)}/80 flagged ===")
real=fp=0
for e,g in flag:
    inv=g.split(':',1)[1].strip() if ':' in g else ''
    ninv=norm(inv); ctx=conv_text_norm(e)
    present = ninv and (ninv in ctx or any(w in ctx for w in ninv.split() if len(w)>=5))
    tag='FALSE POSITIVE (present in conversation)' if present else 'TRUE (absent)'
    if present: fp+=1
    else: real+=1
    print(f"  [{e['session_id'][:8]}] claimed invented: '{inv}' -> {tag}")
    print(f"       response: {e['predicted_response'][:150]}")
print(f"\nE4B detector summary: {real} true / {fp} false positives (out of {len(flag)} flags)")
