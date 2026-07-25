"""single-pick reranker inference: full dev (8,000) or Blind-A.

The model outputs THE best candidate; place it at rank 1 and fill ranks 2–20 in
pool order (neutral for single-GT nDCG). Write the submission for the official
evaluator (dev) or for Blind-A.

Dev :
    CUDA_VISIBLE_DEVICES=0 python src/reranking/predict_firstpos.py --model models/llama32_3b_firstpos_top50 \\
        --pool exp/combined_pool_ctx1024_dev.parquet --tid firstpos_top50 --n_cand 50
BlindA :
    CUDA_VISIBLE_DEVICES=0 python src/reranking/predict_firstpos.py --model ... --mode blindA \\
        --pool exp/combined_pool_ctx1024_blindA.parquet --tid firstpos_top50 --n_cand 50
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys, json, re, argparse, warnings
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--pool', required=True)
ap.add_argument('--tid', required=True)
ap.add_argument('--n_cand', type=int, default=50)
ap.add_argument('--mode', default='dev', choices=['dev', 'blindA'])
ap.add_argument('--max_seq_len', type=int, default=6144)
ap.add_argument('--blind_parquet', default='data/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet',
                help='(Blind-A mode) blind-set Parquet file to score; for Blind-B, pass its path')
ap.add_argument('--out_dir', default='exp/inference/blindset_A',
                help='(Blind-A mode) submission output directory')
args = ap.parse_args()

DATA = Path('data')
if args.mode == 'dev':
    OUT = Path('music-crs-evaluator/exp/inference/devset'); OUT.mkdir(parents=True, exist_ok=True)
    OUT_FILE = OUT / f'{args.tid}.json'
    src = DATA / 'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet'
else:
    OUT_FILE = Path(args.out_dir) / f'{args.tid}.json'; OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    src = Path(args.blind_parquet)

def SYSTEM_PROMPT(n):
    return ("You are an expert music recommender. Given a user profile, a conversation goal, "
            f"a conversation history ending with a user request, and a list of {n} candidate tracks, "
            "pick THE single best track for the user's final request. "
            "Output ONLY a JSON array with that one candidate index (1-based), e.g. [12]. "
            "Do not output anything else.")
TOPK = 20

ds = pd.read_parquet(src)
tm  = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c] = tm[c].apply(lambda x: x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
track_ids = tm['track_id'].tolist(); lk = tm.set_index('track_id')
sess = {s['session_id']: s for _, s in ds.iterrows()}

def fmt_profile(up):
    if up is None: return ''
    return ', '.join(f'{k}={up.get(k)}' for k in ['age_group','country_name','gender','preferred_language','preferred_musical_culture'] if up.get(k))
def fmt_goal(g):
    if g is None: return ''
    return ', '.join(f'{k}={g.get(k)}' for k in ['category','specificity','listener_goal'] if g.get(k))
def short(t):
    if t not in lk.index: return t
    r = lk.loc[t]; return f"{r['track_name']} - {r['artist_name']}"
def tcompact(row):
    p = [f"{row['track_name']} by {row['artist_name']}"]
    if isinstance(row['tag_list'],(list,np.ndarray)) and len(row['tag_list'])>0: p.append(f"[{', '.join(list(row['tag_list'])[:5])}]")
    return ' '.join(p)
def conv(cs, tt):
    L=[]
    for t in cs:
        if int(t['turn_number'])>=tt: break
        ro,co=t['role'],t['content']
        if ro=='music': ro,co='assistant_played',short(co)
        L.append(f'{ro}: {co}')
    for t in cs:
        if int(t['turn_number'])==tt and t['role']=='user': L.append(f"user (REQUEST): {t['content']}"); break
    return '\n'.join(L)

pool = pd.read_parquet(args.pool)
print(f'{len(pool)} turns | model={args.model} | tid={args.tid} | n_cand={args.n_cand} | mode={args.mode}', flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
tok.truncation_side = 'left'
llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()

JSON_RE = re.compile(r'\[\s*(?:\d+\s*,\s*)*\d+\s*\]')
def parse_rank(text, n):
    m = JSON_RE.search(text)
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out, seen = [], set()
    for x in arr:
        if isinstance(x,int) and 1<=x<=n and x not in seen: out.append(x-1); seen.add(x)
    return out

@torch.no_grad()
def rerank(c, up, goal, cand):
    lines = [f'{k}. {tcompact(tm.iloc[i])}' for k,i in enumerate(cand,1)]
    user = (f"User profile: {up}\nConversation goal: {goal}\n\nConversation:\n{c}\n\n"
            f"Candidate tracks ({len(cand)} candidates, 1-based indices):\n" + '\n'.join(lines) +
            f"\n\nPick the single best track. Output JSON array with one index.")
    text = tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT(len(cand))},{'role':'user','content':user}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors='pt', truncation=True, max_length=args.max_seq_len).to('cuda:0')
    out = llm.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    parsed = parse_rank(gen, len(cand)); order, seen = [], set()
    for p in parsed: order.append(p); seen.add(p)            # Model pick(s) first.
    for p in range(len(cand)):                                # Then pool order (nDCG-neutral).
        if p not in seen: order.append(p)
    return [int(cand[p]) for p in order][:TOPK]

subs = []
for _, r in tqdm(pool.iterrows(), total=len(pool)):
    s = sess[r['session_id']]; tn = int(r['turn']); cand = json.loads(r['pool'])[:args.n_cand]
    final = rerank(conv(s['conversations'], tn), fmt_profile(s.get('user_profile')), fmt_goal(s.get('conversation_goal')), cand)
    subs.append({'session_id': r['session_id'], 'user_id': s['user_id'], 'turn_number': tn,
                 'predicted_track_ids': [track_ids[i] for i in final], 'predicted_response': ''})

with open(OUT_FILE, 'w') as f:
    json.dump(subs, f, ensure_ascii=False, indent=2)
print(f'Saved {len(subs)} preds -> {OUT_FILE}', flush=True)
