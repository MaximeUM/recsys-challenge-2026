"""Reselect from existing v2 candidates (cands.json): compare three strategies WITHOUT regenerating.
(a) pure best-judge selection (hallucination filter OFF, no diversity)
(b) clean candidates (name the recommendation + no out-of-conversation title), then best judge
(c) clean + diversity-aware (current v2)
Report: mean local judge score / lexical diversity / distinct openings / hallucinations.

    CUDA_VISIBLE_DEVICES=0 python src/response/reselect_compare.py
"""
import os, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
CANDS='exp/inference/blindset_B/firstpos_top50_ctx1024_blindB_convbestofN_v2.json.cands.json'
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
def conv_norm(e):
    cs=sess[e['session_id']]['conversations']; tt=int(e['turn_number']); P=[]
    for t in cs:
        if int(t['turn_number'])>tt: break
        P.append(nm(t['content']) if t['role']=='music' else str(t['content']))
    P.append(nm(e['predicted_track_ids'][0])); return norm(' || '.join(P))
QUOTE=re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”')
def clean(cand,e):
    top=e['predicted_track_ids'][0]; tnn=norm(lk.loc[top,'track_name']); ar=norm(lk.loc[top,'artist_name']); r=norm(cand)
    names=tnn in r or (len(tnn)>=4 and ' '.join(tnn.split()[:3]) in r) or (ar.split()[0] in r and len(ar.split()[0])>=4)
    ctx=conv_norm(e)
    for m in QUOTE.finditer(cand):
        q=norm(m.group(1) or m.group(2))
        if len(q)>=3 and not (q in ctx or any(w in ctx for w in q.split() if len(w)>=5)): return False
    return names
op=lambda r:' '.join(norm(r).split()[:3])
data=json.load(open(CANDS))
# Judge.
J='google/gemma-4-E2B-it'; jt=AutoTokenizer.from_pretrained(J); jt.truncation_side='left'; jt.padding_side='left'
if jt.pad_token_id is None: jt.pad_token=jt.eos_token
jm=AutoModelForCausalLM.from_pretrained(J,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
JSYS=("You evaluate the assistant's LAST reply, two dims (ignore track correctness): Personalization (1-5) + "
 "Explanation Quality (1-5). Be critical. Output EXACTLY: Personalization: <n>, Explanation: <n>")
NUM=re.compile(r'Personalization:\s*([1-5]).*?Explanation:\s*([1-5])',re.S)
@torch.no_grad()
def judge(items):
    txts=[jt.apply_chat_template([{'role':'system','content':JSYS},{'role':'user','content':
      f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'],int(e['turn_number']))}\n\nRecommended track: {nm(e['predicted_track_ids'][0])}\n\nAssistant's last reply to evaluate:\n{c}\n\nScores:"}],
      tokenize=False,add_generation_prompt=True) for e,c in items]
    enc=jt(txts,return_tensors='pt',truncation=True,max_length=3072,padding=True).to('cuda:0')
    out=jm.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=jt.eos_token_id); L=enc['input_ids'].shape[1]
    r=[]
    for j in range(len(items)):
        g=jt.decode(out[j,L:],skip_special_tokens=True); m=NUM.search(g); r.append((int(m.group(1))+int(m.group(2))) if m else 0)
    return r
pairs=[(e,c) for e in data for c in e['_cands']]; sc=[]
for i in tqdm(range(0,len(pairs),8)):
    try: sc.extend(judge(pairs[i:i+8]))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        for p in pairs[i:i+8]:
            try: sc.extend(judge([p]))
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); sc.append(0)
it=iter(sc)
for e in data: e['_sc']=[next(it) for _ in e['_cands']]
import sys; sys.path.insert(0,'music-crs-evaluator'); from metrics import compute_lexical_diversity
def evaluate(select):
    used=Counter(); chosen=[]; jt_=0; halluc=0
    for e in data:
        cs=e['_cands']; sc_=e['_sc']; i=select(e,cs,sc_,used)
        used[op(cs[i])]+=1; chosen.append(cs[i]); jt_+=sc_[i]
        if not clean(cs[i],e): halluc+=1
    ld=compute_lexical_diversity(chosen); nd=len(set(op(c) for c in chosen))
    return jt_/len(data), ld, nd, halluc
def s_pure(e,cs,sc,used): return int(np.argmax(sc))
def s_clean(e,cs,sc,used):
    idx=[i for i in range(len(cs)) if clean(cs[i],e)] or list(range(len(cs)))
    return max(idx,key=lambda i:sc[i])
def s_cleandiv(e,cs,sc,used):
    idx=[i for i in range(len(cs)) if clean(cs[i],e)] or list(range(len(cs)))
    fresh=[i for i in idx if used[op(cs[i])]==0] or idx
    return max(fresh,key=lambda i:sc[i])
print("\n=== RESELECTION (same v2 candidates) ===")
print(f"  {'strategy':<34}{'judge':>6}{'lex_div':>9}{'openings':>9}{'halluc':>8}")
for lab,fn in [('(a) pure best judge (halluc. OFF)',s_pure),('(b) clean + best judge',s_clean),('(c) clean + diversity (= v2)',s_cleandiv)]:
    j,ld,nd,h=evaluate(fn); print(f"  {lab:<34}{j:>6.2f}{ld:>9.4f}{nd:>8}/80{h:>7}")
