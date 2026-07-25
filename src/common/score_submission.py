"""Score all Blind-B submission candidates UNIFORMLY using the same local judge (gemma-4-E2B),
lexical diversity, distinct openings, and structural hallucination check.
Output a consistent table for SUBMISSIONS_BLINDB.md.

    CUDA_VISIBLE_DEVICES=0 python src/common/score_submission.py
"""
import os, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')
DATA='data/'
b=pd.read_parquet(DATA+'TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet')
tm=pd.read_parquet(DATA+'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
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
QUOTE=re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”'); op=lambda r:' '.join(norm(r).split()[:3])
def halluc(e):
    tid=e['predicted_track_ids'][0]; r=norm(e['predicted_response'])
    tn=norm(lk.loc[tid,'track_name']); tnb=norm(re.sub(r'\([^)]*\)','',lk.loc[tid,'track_name'])); ar=norm(lk.loc[tid,'artist_name'])
    names=(tn in r) or (tnb and tnb in r) or (len(tnb)>=4 and ' '.join(tnb.split()[:3]) in r) or (ar.split()[0] in r and len(ar.split()[0])>=4)
    if not names: return True
    ctx=conv_norm(e)
    for m in QUOTE.finditer(e['predicted_response']):
        q=norm(m.group(1) or m.group(2))
        if len(q)>=3 and not (q in ctx or any(w in ctx for w in q.split() if len(w)>=5)): return True
    return False
J='google/gemma-4-E2B-it'; jt=AutoTokenizer.from_pretrained(J); jt.truncation_side='left'; jt.padding_side='left'
if jt.pad_token_id is None: jt.pad_token=jt.eos_token
jm=AutoModelForCausalLM.from_pretrained(J,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
JSYS=("You evaluate the assistant's LAST reply, two dims (ignore track correctness): Personalization (1-5) + Explanation Quality (1-5). Be critical. Output EXACTLY: Personalization: <n>, Explanation: <n>")
NUM=re.compile(r'Personalization:\s*([1-5]).*?Explanation:\s*([1-5])',re.S)
@torch.no_grad()
def judge(items):
    txts=[jt.apply_chat_template([{'role':'system','content':JSYS},{'role':'user','content':f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'],int(e['turn_number']))}\n\nRecommended track: {nm(e['predicted_track_ids'][0])}\n\nAssistant's last reply to evaluate:\n{e['predicted_response']}\n\nScores:"}],tokenize=False,add_generation_prompt=True) for e in items]
    enc=jt(txts,return_tensors='pt',truncation=True,max_length=3072,padding=True).to('cuda:0')
    out=jm.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=jt.eos_token_id); L=enc['input_ids'].shape[1]
    return [(int(NUM.search(g).group(1))+int(NUM.search(g).group(2))) if NUM.search(g) else 0 for g in [jt.decode(out[j,L:],skip_special_tokens=True) for j in range(len(items))]]
import sys; sys.path.insert(0,'music-crs-evaluator'); from metrics import compute_lexical_diversity
FILES={'vote (#1)':('firstpos_vote_blindB_convbestofN.json',0.1747),
       'intent+mm (#2)':('firstpos_intentmm_blindB_convbestofN.json',0.1718),
       'v2b (#3)':('firstpos_top50_ctx1024_blindB_convbestofN_v2b.json',0.1681),
       'v2 (alt)':('firstpos_top50_ctx1024_blindB_convbestofN_v2.json',0.1681)}
print(f"\n{'candidate':<16}{'nDCG_dev':>9}{'local_judge':>12}{'lexical':>9}{'opening':>8}{'halluc':>7}{'words':>6}")
for lab,(f,nd) in FILES.items():
    d=json.load(open('exp/inference/blindset_B/'+f))
    sc=[]
    for i in range(0,len(d),8): sc+=judge(d[i:i+8])
    rs=[e['predicted_response'] for e in d]
    print(f"{lab:<16}{nd:>9.4f}{np.mean(sc):>9.2f}{compute_lexical_diversity(rs):>9.4f}{len(set(op(r) for r in rs)):>6}/80{sum(halluc(e) for e in d):>7}{np.mean([len(r.split()) for r in rs]):>6.0f}",flush=True)
