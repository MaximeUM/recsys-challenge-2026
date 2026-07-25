"""Large-scale RFT generation (Lucia A100 40 GB), ROBUST two-phase process with one resident model.
Generator = gemma-4-E4B-it; judge = gemma-4-E2B-it. Retain scores at or above the threshold.

Two phases avoid OOM from keeping both models resident because gemma-4 has large activations.
Phase 1 runs generation alone with a large batch; phase 2 runs the judge alone, nearly eliminating OOM.

Safeguards: INCREMENTAL saves (candidates then output) and per-chunk OOM handling (skip),
plus a time budget (--time_budget) that stops and saves before the SLURM limit.

    CUDA_VISIBLE_DEVICES=0 python src/ablations/rft_distillation_generate.py --shard_id 0 --total_shards 32 \\
        --total_turns 121592 --gen_bs 6 --judge_bs 48 --n 8 --keep 9 --out exp/rft_lucia/rft_0.parquet
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0'); os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse, re, time, random, warnings, gc
import numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore'); torch.manual_seed(42)
ap=argparse.ArgumentParser()
ap.add_argument('--gen', default='google/gemma-4-E4B-it')
ap.add_argument('--judge', default='google/gemma-4-E2B-it')
ap.add_argument('--total_turns', type=int, default=121592)
ap.add_argument('--total_shards', type=int, default=32)
ap.add_argument('--shard_id', type=int, required=True)
ap.add_argument('--gen_bs', type=int, default=6)
ap.add_argument('--judge_bs', type=int, default=48)
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--keep', type=int, default=9)
ap.add_argument('--gen_chunk', type=int, default=120)
ap.add_argument('--maxlen', type=int, default=2048)
ap.add_argument('--time_budget', type=int, default=6600)
ap.add_argument('--out', required=True)
args=ap.parse_args()
DATA=Path('data'); MAXLEN=args.maxlen; t0=time.time()
def log(m): print(f'[shard {args.shard_id}] {m}', flush=True)
TMP=Path(args.out).with_suffix('.cand.parquet')

train=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
tm=pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']:
    tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
lk=tm.set_index('track_id')
def name(t):
    if t not in lk.index: return t
    r=lk.loc[t]; return f"{r['track_name']} by {r['artist_name']}"
def tags(t):
    if t not in lk.index: return ''
    tl=lk.loc[t,'tag_list']; return ', '.join(list(tl)[:5]) if isinstance(tl,(list,np.ndarray)) and len(tl)>0 else ''
def goal_str(s):
    g=s.get('conversation_goal'); return g['listener_goal'] if isinstance(g,dict) and g.get('listener_goal') else ''
def prof(s):
    up=s.get('user_profile') or {}; p=[]
    if up.get('age'): p.append(f"{up['age']} y/o")
    for k in ['gender','country_name','preferred_language']:
        if up.get(k): p.append(str(up[k]))
    if up.get('preferred_musical_culture'): p.append(f"into {up['preferred_musical_culture']}")
    return ', '.join(p)
def full_conv(cs,tt):
    order={'user':0,'music':1,'assistant':2}; L=[]
    for t in sorted(cs,key=lambda x:(int(x['turn_number']),order[x['role']])):
        if int(t['turn_number'])>tt: break
        if int(t['turn_number'])==tt and t['role']!='user': continue
        if t['role']=='user': L.append(f"User: {t['content']}")
        elif t['role']=='music': L.append(f"Assistant played: {name(t['content'])}")
        elif t['role']=='assistant': L.append(f"Assistant: {t['content']}")
    return '\n'.join(L)
def gen_ctx(s,tt,track):
    ctx=f"Conversation:\n{full_conv(s['conversations'],tt)}"
    if goal_str(s): ctx=f"Session goal: {goal_str(s)}\n"+ctx
    if prof(s): ctx+=f"\n(User: {prof(s)})"
    ctx+=f"\nRecommend: {name(track)}"+(f" (tags: {tags(track)})" if tags(track) else '')
    return ctx
GEN_SYS=("You are the assistant in an ongoing music chat. Read the FULL conversation, then write your next reply "
 "(~45 words) recommending the given track. Your reply MUST: (1) be coherent with the conversation — build on what the "
 "user asked, liked or refined; (2) explain concretely WHY this track fits (genre, energy, mood, era, artist, lyrics). "
 "Warm, natural, specific — never generic.")
JSYS=("You rate an assistant's music recommendation reply on personalization AND explanation quality. "
      "Output ONLY a single integer from 0 to 10 (10=excellent on both, 0=terrible). No words, just the number.")
NUM=re.compile(r'\b(10|[0-9])\b')

items_all=[]
for _,s in train.iterrows():
    bt={}
    for t in s['conversations']: bt.setdefault(int(t['turn_number']),{})[t['role']]=t['content']
    for tt,roles in bt.items():
        if 'music' in roles: items_all.append((s,tt,roles['music']))
random.Random(42).shuffle(items_all); items_all=items_all[:args.total_turns]
per=(len(items_all)+args.total_shards-1)//args.total_shards
items=items_all[args.shard_id*per:(args.shard_id+1)*per]
log(f'{len(items)} turns (slice {args.shard_id}/{args.total_shards})')

def jctx(s,tt,track,resp):
    h=''
    if goal_str(s): h+=f"Session goal: {goal_str(s)}\n"
    if prof(s): h+=f"User: {prof(s)}\n"
    return (f"{h}Conversation:\n{full_conv(s['conversations'],tt)}\n\nRecommended track: {name(track)}\n\n"
            f"Assistant's last reply to evaluate:\n{resp}\n\nScore (0-10):")

# PHASE 1: GENERATION (gemma-4-E4B only).
log(f'PHASE 1 generation {args.gen} (gen_bs={args.gen_bs})')
gtok=AutoTokenizer.from_pretrained(args.gen); gtok.pad_token=gtok.pad_token or gtok.eos_token
gtok.padding_side='left'; gtok.truncation_side='left'
gen=AutoModelForCausalLM.from_pretrained(args.gen,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
@torch.no_grad()
def gen_batch(chunk):
    texts=[gtok.apply_chat_template([{'role':'system','content':GEN_SYS},{'role':'user','content':gen_ctx(s,tt,tr)}],tokenize=False,add_generation_prompt=True) for s,tt,tr in chunk]
    enc=gtok(texts,return_tensors='pt',truncation=True,max_length=MAXLEN,padding=True).to('cuda:0')
    out=gen.generate(**enc,max_new_tokens=100,do_sample=True,temperature=0.9,top_p=0.95,num_return_sequences=args.n,pad_token_id=gtok.eos_token_id)
    L=enc['input_ids'].shape[1]
    dec=[gtok.decode(out[j,L:],skip_special_tokens=True).strip().replace('\n',' ') for j in range(out.shape[0])]
    return [dec[i*args.n:(i+1)*args.n] for i in range(len(chunk))]
cand_rows=[]; gdone=0
gen_budget=args.time_budget*0.6
for ci in range(0,len(items),args.gen_chunk):
    if time.time()-t0>gen_budget: log('generation budget reached'); break
    chunk=items[ci:ci+args.gen_chunk]
    try:
        for b in range(0,len(chunk),args.gen_bs):
            sub=chunk[b:b+args.gen_bs]
            for (s,tt,tr),resps in zip(sub,gen_batch(sub)):
                gp=gen_ctx(s,tt,tr)
                for r in resps: cand_rows.append({'gen_prompt':gp,'judge_prompt':jctx(s,tt,tr,r),'target':r})
        gdone+=len(chunk); pd.DataFrame(cand_rows).to_parquet(TMP)
        if (ci//args.gen_chunk)%5==0: log(f'generated {gdone}/{len(items)} | {len(cand_rows)} candidates | {time.time()-t0:.0f}s')
    except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); log(f'generation OOM @{ci}, skipped'); continue
    except Exception as e: torch.cuda.empty_cache(); log(f'generation error @{ci}: {str(e)[:100]}'); continue
pd.DataFrame(cand_rows).to_parquet(TMP)
log(f'PHASE 1 done: {len(cand_rows)} candidates')
del gen; gc.collect(); torch.cuda.empty_cache()

# PHASE 2: JUDGE (gemma-4-E2B only).
log(f'PHASE 2 judge {args.judge} (judge_bs={args.judge_bs})')
jtok=AutoTokenizer.from_pretrained(args.judge); jtok.truncation_side='left'; jtok.padding_side='left'
judge=AutoModelForCausalLM.from_pretrained(args.judge,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
@torch.no_grad()
def judge_batch(texts):
    enc=jtok(texts,return_tensors='pt',truncation=True,max_length=MAXLEN,padding=True).to('cuda:0')
    out=judge.generate(**enc,max_new_tokens=8,do_sample=False,pad_token_id=jtok.eos_token_id)
    L=enc['input_ids'].shape[1]; res=[]
    for j in range(out.shape[0]):
        g=jtok.decode(out[j,L:],skip_special_tokens=True); m=NUM.search(g)
        res.append(int(m.group(1)) if m else 0)
    return res
cand=pd.read_parquet(TMP); rows=[]; jdone=0
for b in range(0,len(cand),args.judge_bs):
    if time.time()-t0>args.time_budget: log('total time budget reached'); break
    sub=cand.iloc[b:b+args.judge_bs]
    try:
        scs=judge_batch(sub['judge_prompt'].tolist())
        for (_,r),sc in zip(sub.iterrows(),scs):
            if sc>=args.keep: rows.append({'prompt':r['gen_prompt'],'target':r['target'],'score':sc})
        jdone+=len(sub)
        if (b//args.judge_bs)%20==0:
            pd.DataFrame(rows).to_parquet(args.out); log(f'judged {jdone}/{len(cand)} | retained {len(rows)} | {time.time()-t0:.0f}s')
    except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); log(f'judge OOM @{b}, skipped'); continue
    except Exception as e: torch.cuda.empty_cache(); log(f'judge error @{b}: {str(e)[:100]}'); continue
pd.DataFrame(rows).to_parquet(args.out)
try: TMP.unlink()
except: pass
log(f'DONE: {gdone} turns, {len(cand)} candidates, {len(rows)} retained (>= {args.keep}) -> {args.out}')
