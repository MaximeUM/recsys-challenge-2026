"""convbestof-N (large N) + CLEAN selection: for every turn, generate N candidates with gemma-3n,
grounded in the full conversation. Select a candidate that names the recommended track and cites NO
title absent from the conversation (a reliable structural check), then maximize the local gemma-4-E2B
judge score among clean candidates. If none is clean, use the best judge score. track_ids remain unchanged.

    CUDA_VISIBLE_DEVICES=0 python src/response/convbestof20_clean.py \\
        --input exp/inference/blindset_A/firstpos_top50_ctx1024_blindB.json \\
        --blind_parquet data/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet \\
        --out exp/inference/blindset_A/firstpos_top50_ctx1024_blindB_convbestofN.json --n 20
"""
import os, argparse, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
warnings.filterwarnings('ignore'); torch.manual_seed(42)
ap=argparse.ArgumentParser()
ap.add_argument('--input', required=True); ap.add_argument('--blind_parquet', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=20); ap.add_argument('--gen', default='google/gemma-3n-E4B-it'); ap.add_argument('--judge', default='google/gemma-4-E2B-it')
a=ap.parse_args(); MAXLEN=3072
b=pd.read_parquet(a.blind_parquet); tm=pd.read_parquet('data/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']: tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
lk=tm.set_index('track_id'); sess={s['session_id']:s for _,s in b.iterrows()}
norm=lambda s:re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()
def nm(t): return f"{lk.loc[t,'track_name']} by {lk.loc[t,'artist_name']}" if t in lk.index else str(t)
def tg(t):
    if t not in lk.index: return ''
    tl=lk.loc[t,'tag_list']; return ', '.join(list(tl)[:5]) if isinstance(tl,(list,np.ndarray)) and len(tl)>0 else ''
def full_conv(cs,tt):
    o={'user':0,'music':1,'assistant':2}; L=[]
    for t in sorted(cs,key=lambda x:(int(x['turn_number']),o[x['role']])):
        if int(t['turn_number'])>tt: break
        if int(t['turn_number'])==tt and t['role']!='user': continue
        if t['role']=='user': L.append(f"User: {t['content']}")
        elif t['role']=='music': L.append(f"Assistant played: {nm(t['content'])}")
        elif t['role']=='assistant': L.append(f"Assistant: {t['content']}")
    return '\n'.join(L)
def conv_norm(e):  # Full conversation text (user + played tracks) + recommendation for structural checking.
    cs=sess[e['session_id']]['conversations']; tt=int(e['turn_number']); P=[]
    for t in cs:
        if int(t['turn_number'])>tt: break
        P.append(nm(t['content']) if t['role']=='music' else str(t['content']))
    P.append(nm(e['predicted_track_ids'][0]))
    return norm(' || '.join(P))
GEN_SYS=("You are the assistant in an ongoing music chat. Read the FULL conversation, then write your next reply "
 "(~45 words) recommending the given track. (1) be coherent with the conversation — build on what the user asked, "
 "liked or refined; (2) explain concretely WHY this track fits (genre, energy, mood, era, artist, lyrics). "
 "Warm, natural, specific.")
def gctx(e):
    top=e['predicted_track_ids'][0]
    return f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'],int(e['turn_number']))}\nRecommend: {nm(top)}"+(f" (tags: {tg(top)})" if tg(top) else '')
QUOTE=re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”')
def clean(cand, e):
    top=e['predicted_track_ids'][0]; tnn=norm(lk.loc[top,'track_name']); ar=norm(lk.loc[top,'artist_name']); r=norm(cand)
    names_rec = tnn in r or (len(tnn)>=4 and ' '.join(tnn.split()[:3]) in r) or (ar.split()[0] in r and len(ar.split()[0])>=4)
    ctx=conv_norm(e)
    for m in QUOTE.finditer(cand):
        q=norm(m.group(1) or m.group(2))
        if len(q)<3: continue
        if not (q in ctx or any(w in ctx for w in q.split() if len(w)>=5)): return False  # Cited title absent from conversation.
    return names_rec

data=json.load(open(a.input))
# Phase 1: generate N candidates.
gt=AutoTokenizer.from_pretrained(a.gen); gt.padding_side='left'; gt.truncation_side='left'
if gt.pad_token_id is None: gt.pad_token=gt.eos_token
gm=AutoModelForCausalLM.from_pretrained(a.gen,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
print(f'Phase 1: {a.n} candidates/turn ({a.gen})...', flush=True)
for e in tqdm(data):
    txt=gt.apply_chat_template([{'role':'system','content':GEN_SYS},{'role':'user','content':gctx(e)}],tokenize=False,add_generation_prompt=True)
    enc=gt(txt,return_tensors='pt',truncation=True,max_length=MAXLEN).to('cuda:0')
    with torch.no_grad():
        out=gm.generate(**enc,max_new_tokens=100,do_sample=True,temperature=0.9,top_p=0.95,num_return_sequences=a.n,pad_token_id=gt.eos_token_id)
    L=enc['input_ids'].shape[1]
    e['_cands']=[gt.decode(o[L:],skip_special_tokens=True).strip().replace('\n',' ') for o in out]
del gm; torch.cuda.empty_cache()
# Phase 2: judge + clean selection.
jt=AutoTokenizer.from_pretrained(a.judge); jt.truncation_side='left'; jt.padding_side='left'
if jt.pad_token_id is None: jt.pad_token=jt.eos_token
jm=AutoModelForCausalLM.from_pretrained(a.judge,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
JSYS=("You evaluate the assistant's LAST reply in a music chat, two dims (ignore track correctness): "
 "Personalization (1-5) + Explanation Quality (1-5). Be critical. Output EXACTLY: Personalization: <n>, Explanation: <n>")
NUM=re.compile(r'Personalization:\s*([1-5]).*?Explanation:\s*([1-5])',re.S)
@torch.no_grad()
def judge_batch(items):  # items: (e, cand)
    txts=[jt.apply_chat_template([{'role':'system','content':JSYS},{'role':'user','content':
        f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'],int(e['turn_number']))}\n\nRecommended track: {nm(e['predicted_track_ids'][0])}\n\nAssistant's last reply to evaluate:\n{c}\n\nScores:"}],
        tokenize=False,add_generation_prompt=True) for e,c in items]
    enc=jt(txts,return_tensors='pt',truncation=True,max_length=MAXLEN,padding=True).to('cuda:0')
    out=jm.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=jt.eos_token_id); L=enc['input_ids'].shape[1]
    res=[]
    for j in range(len(items)):
        g=jt.decode(out[j,L:],skip_special_tokens=True); m=NUM.search(g); res.append((int(m.group(1))+int(m.group(2))) if m else 0)
    return res
print('Phase 2: judge + clean selection...', flush=True)
pairs=[(e,c) for e in data for c in e['_cands']]
scores=[]
for i in tqdm(range(0,len(pairs),16)): scores.extend(judge_batch(pairs[i:i+16]))
it=iter(scores); nclean=0; tot=0.0
for e in data:
    cs=e['_cands']; sc=[next(it) for _ in cs]; cl=[clean(c,e) for c in cs]
    idxs=[i for i in range(len(cs)) if cl[i]]
    if idxs: best=max(idxs,key=lambda i:sc[i]); nclean+=1
    else: best=int(np.argmax(sc))
    e['predicted_response']=cs[best]; tot+=sc[best]; e.pop('_cands',None)
import sys; sys.path.insert(0,'music-crs-evaluator')
try:
    from metrics import compute_lexical_diversity
    ld=compute_lexical_diversity([e['predicted_response'] for e in data])
except Exception: ld=-1
print(f"clean selection available: {nclean}/{len(data)} | mean selected local judge={tot/len(data):.2f}/10 | lexical_div={ld:.4f}", flush=True)
json.dump(data, open(a.out,'w'), ensure_ascii=False, indent=2)
print('Saved ->', a.out, flush=True)
