"""IntRS'25 dimensions (Manderlier et al., CEUR Vol-4027) evaluated by a local model judge.
Follows the questionnaire protocol of the reference study:
  - seven human-questionnaire statements (Likert 1-5, verbatim English, movies adapted to music), scored in
    a SINGLE call with JSON output;
  - NEUTRAL labels A..G shown to the judge; IDs T1/E1/... and dimension names are analysis-only;
  - user context (the conversation) and an instruction to adopt that user's perspective,
    equivalent to USE_USER_HISTORY=True; temperature 0.
Evaluate 400 generated dev responses and the 400 paired gold responses from the same turns.

    CUDA_VISIBLE_DEVICES=0 python src/explanation_metrics/userstudy_dimensions.py --shard 0 --nshards 4
    python src/explanation_metrics/userstudy_dimensions.py --combine     # Also compare with v1 when present.
"""
import os, argparse, glob, json, re, warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

# Questionnaire: notebook IDs/labels/dimensions and user-study statements adapted from movies to tracks.
QUESTIONS = [
    ('A', 'T1', 'Transparency', "The explanation helps me understand why this track was recommended to me."),
    ('B', 'T2', 'Transparency', "The explanation allows me to understand, in broad terms, how the recommendation system works."),
    ('C', 'E1', 'Effectiveness', "Thanks to this explanation, I feel I can make an informed decision about whether I will like this track or not."),
    ('D', 'E2', 'Effectiveness', "The explanation helps me determine whether I would like the recommended track."),
    ('E', 'P1', 'Persuasion', "The explanation makes me more likely to follow the recommendation and listen to the track."),
    ('F', 'TR1', 'Trust', "Thanks to the explanation, I trust the system more to recommend tracks that match my tastes."),
    ('G', 'S1', 'Satisfaction', "Overall, I am satisfied with the provided explanation."),
]
SCALE = {1: "Strongly disagree", 2: "Somewhat disagree", 3: "Neither agree nor disagree",
         4: "Somewhat agree", 5: "Strongly agree"}
LBL2DIM = {l: d for l, _, d, _ in QUESTIONS}

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=4)
ap.add_argument('--judge', default='google/gemma-4-E2B-it'); ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--tag', default='json', help='suffix of the output directory exp/intrs_dims_<tag>')
ap.add_argument('--thinking', action='store_true', help='Qwen3: enable thinking mode (sampling '
                'recommended Qwen temp 0.6/top-p 0.95/top-k 20; greedy is discouraged in thinking mode)')
ap.add_argument('--vllm', action='store_true', help='use the vLLM backend (requires vllm) instead of transformers')
ap.add_argument('--seed', type=int, default=2026, help='sampling seed (affects only --thinking; direct mode is greedy)')
ap.add_argument('--dev_resp', default='exp/inference/devset/shqwen8b_respsample_convbestofN.json')
ap.add_argument('--combine', action='store_true')
a = ap.parse_args()
OUT = Path(f'exp/intrs_dims_{a.tag}'); OUT.mkdir(exist_ok=True)

def report(df, tag):
    ok = df[df['score'] > 0]
    print(f'\n=== IntRS25 dimensions [{tag}] — paired Likert 1-5 '
          f'({len(df)} scores, {len(df)-len(ok)} parse failures) ===')
    ok = ok.assign(dim=ok['item'].map(LBL2DIM) if ok['item'].isin(LBL2DIM).any() else
                   ok['item'].map({i: d for _, i, d, _ in QUESTIONS}))
    piv = ok.pivot_table(index=['session_id', 'turn', 'cond'], columns='dim', values='score').reset_index()
    from scipy.stats import wilcoxon
    print(f'{"dimension":14s} {"generated":>9s} {"gold":>7s} {"delta":>7s}   Wilcoxon')
    for dim in ['Transparency', 'Effectiveness', 'Persuasion', 'Trust', 'Satisfaction']:
        w = piv.pivot_table(index=['session_id', 'turn'], columns='cond', values=dim).dropna()
        st, p = wilcoxon(w['gen'], w['gold'])
        print(f'{dim:14s} {w["gen"].mean():7.2f} {w["gold"].mean():7.2f} {w["gen"].mean()-w["gold"].mean():+7.2f}   p={p:.1e} (n={len(w)})')

if a.combine:
    for d in ['exp/intrs_dims_json', 'exp/intrs_dims_gemma4_think', 'exp/intrs_dims_qwen3_8b',
              'exp/intrs_dims_qwen3_8b_think', 'exp/intrs_dims_llama32_3b', 'exp/intrs_dims']:
        fs = sorted(glob.glob(f'{d}/scores_[0-9]*.parquet'))
        if not fs: continue
        tag = {'exp/intrs_dims_json': 'gemma-4-E2B judge', 'exp/intrs_dims_qwen3_8b': 'Qwen3-8B judge',
               'exp/intrs_dims_gemma4_think': 'gemma-4-E2B THINKING judge (temp 0.6)',
               'exp/intrs_dims_qwen3_8b_think': 'Qwen3-8B THINKING judge (temp 0.6)',
               'exp/intrs_dims_llama32_3b': 'Llama-3.2-3B judge', 'exp/intrs_dims': 'gemma-4-E2B, item-by-item v1 protocol'}[d]
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        report(df, tag)
        df.to_parquet(Path(d)/'all_scores.parquet')
    raise SystemExit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path('data')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name', 'artist_name']:
    tm[c] = tm[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x).astype(str)
lk = tm.set_index('track_id')
dev = pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
sess = {s['session_id']: s for _, s in dev.iterrows()}
def nm(t):
    return f"{lk.loc[t,'track_name']} by {lk.loc[t,'artist_name']}" if t in lk.index else str(t)
def full_conv(cs, tt):
    o = {'user': 0, 'music': 1, 'assistant': 2}; L = []
    for t in sorted(cs, key=lambda x: (int(x['turn_number']), o[x['role']])):
        if int(t['turn_number']) > tt: break
        if int(t['turn_number']) == tt and t['role'] != 'user': continue
        if t['role'] == 'user': L.append(f"User: {t['content']}")
        elif t['role'] == 'music': L.append(f"Assistant played: {nm(t['content'])}")
        elif t['role'] == 'assistant': L.append(f"Assistant: {t['content']}")
    return '\n'.join(L)

# -- prompts: structure of build_system_prompt()/build_user_message() from the reference protocol --
scale_block = '\n'.join(f'  {s}: {l}' for s, l in SCALE.items())
questions_block = '\n'.join(f'- {l}: "{st}"' for l, _, _, st in QUESTIONS)
expected_json = '{\n' + ',\n'.join(f'  "{l}": <1-5>' for l, _, _, _ in QUESTIONS) + '\n}'
SYS = f"""You are evaluating the quality of music recommendation explanations.

The user's conversation with the music assistant is provided. Put yourself in this user's shoes to assess whether the explanation is relevant and convincing *specifically for them*.

For each track and its explanation, rate the explanation on the following statements using this scale:
{scale_block}

Statements to rate:
{questions_block}

Respond only with a JSON object:
{expected_json}"""

def user_msg(r):
    return (f"Conversation with the music assistant:\n{full_conv(sess[r['session_id']]['conversations'], r['turn'])}\n\n"
            f"Recommended track: {nm(r['track'])}\nExplanation: \"\"\"{r['resp']}\"\"\"")

def extract_json(content):
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if '```' in content: content = content.split('```')[1].lstrip('json').strip()
    if not content.startswith('{'):
        s, e = content.find('{'), content.rfind('}')
        if s != -1 and e > s: content = content[s:e+1]
    return json.loads(content)

gen = json.load(open(a.dev_resp))
rows = []
for e in gen:
    sid, t = e['session_id'], int(e['turn_number'])
    cs = sess[sid]['conversations']
    mus = next(x['content'] for x in cs if x['role'] == 'music' and int(x['turn_number']) == t)
    gold = next(x['content'] for x in cs if x['role'] == 'assistant' and int(x['turn_number']) == t)
    rows.append({'session_id': sid, 'turn': t, 'cond': 'gen', 'track': e['predicted_track_ids'][0], 'resp': e['predicted_response']})
    rows.append({'session_id': sid, 'turn': t, 'cond': 'gold', 'track': mus, 'resp': gold})
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
print(f'shard {a.shard}: {len(rows)} responses × 1 JSON call', flush=True)

if a.vllm:  # vLLM backend: same prompts, one batched generate() call.
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.judge)
    HAS_THINK = any(k in a.judge.lower() for k in ('qwen3', 'gemma-4'))  # Template supports enable_thinking; default OFF verified.
    kw = {'enable_thinking': a.thinking} if HAS_THINK else {}
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_msg(r)}],
                                       tokenize=False, add_generation_prompt=True, **kw) for r in rows]
    llm = LLM(model=a.judge, dtype='bfloat16', max_model_len=4608, gpu_memory_utilization=0.9, seed=a.seed)

    def sp_for(attempt):
        """Sampling parameters for one attempt. Attempt 0 uses nominal settings
        (direct = reproducible greedy decoding). Retries use a deterministically derived seed;
        direct mode also changes temperature because pure greedy decoding would repeat the same failure."""
        if a.thinking:
            seed = a.seed if attempt == 0 else a.seed + attempt * 100_000
            return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, seed=seed)
        if attempt == 0:
            return SamplingParams(temperature=0.0, max_tokens=128)
        return SamplingParams(temperature=0.3, max_tokens=128, seed=a.seed + attempt * 100_000)

    MAX_RETRIES = 5
    final_sc = [None] * len(rows); final_diag = [None] * len(rows)
    pending = list(range(len(rows)))
    attempt = 0
    while pending and attempt <= MAX_RETRIES:
        sp = sp_for(attempt)
        outs = llm.generate([prompts[i] for i in pending], sp)
        still_pending = []
        for idx, o in zip(pending, outs):
            comp = o.outputs[0]
            try:
                d = extract_json(comp.text); sc = {l: int(d.get(l, 0)) for l, _, _, _ in QUESTIONS}
            except Exception:
                sc = {l: 0 for l, _, _, _ in QUESTIONS}
            ok_parse = all(v > 0 for v in sc.values())
            if not ok_parse and attempt < MAX_RETRIES:
                still_pending.append(idx)  # will be retried on the next pass
            else:
                final_sc[idx] = sc
                final_diag[idx] = {'finish_reason': comp.finish_reason, 'n_tokens_out': len(comp.token_ids),
                                    'n_retries': attempt, 'unrecoverable': not ok_parse}
        if attempt > 0:
            print(f'  retry {attempt}/{MAX_RETRIES}: {len(pending)} pending -> '
                  f'{len(pending) - len(still_pending)} recovered, {len(still_pending)} still failing', flush=True)
        pending = still_pending
        attempt += 1

    recs = []; diag = []
    for r, sc, dg in zip(rows, final_sc, final_diag):
        diag.append({'session_id': r['session_id'], 'turn': r['turn'], 'cond': r['cond'], **dg})
        for l, v in sc.items():
            recs.append({'session_id': r['session_id'], 'turn': r['turn'], 'cond': r['cond'], 'item': l, 'score': v})
    out = pd.DataFrame(recs)
    out.to_parquet(OUT/f'scores_{a.shard}.parquet')
    dd = pd.DataFrame(diag)
    dd.to_parquet(OUT/f'diag_{a.shard}.parquet')
    n_trunc = int((dd['finish_reason'] == 'length').sum())
    n_retried = int((dd['n_retries'] > 0).sum()); n_unrecov = int(dd['unrecoverable'].sum())
    print(f"saved -> {OUT}/scores_{a.shard}.parquet | parse failures (unrecoverable after {MAX_RETRIES} retries): "
          f"{(out['score']==0).sum()//7} | recovered by retry: {n_retried - n_unrecov} | "
          f"TRUNCATED (finish_reason=length): {n_trunc}/{len(dd)} | mean tokens_out={dd['n_tokens_out'].mean():.0f} "
          f"max={dd['n_tokens_out'].max()}", flush=True)
    raise SystemExit

jt = AutoTokenizer.from_pretrained(a.judge); jt.truncation_side = 'left'; jt.padding_side = 'left'
if jt.pad_token_id is None: jt.pad_token = jt.eos_token
jm = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()

TPL_KW = {'enable_thinking': a.thinking} if any(k in a.judge.lower() for k in ('qwen3', 'gemma-4')) else {}
if a.thinking:  # Qwen-recommended sampling for thinking mode; greedy may loop. Fixed seed.
    torch.manual_seed(a.seed)
    GEN_KW = dict(max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
else:
    GEN_KW = dict(max_new_tokens=128, do_sample=False)

@torch.no_grad()
def judge_batch(items):
    txts = [jt.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_msg(r)}],
                                   tokenize=False, add_generation_prompt=True, **TPL_KW) for r in items]
    enc = jt(txts, return_tensors='pt', truncation=False, padding=True).to('cuda:0')
    out = jm.generate(**enc, pad_token_id=jt.eos_token_id, **GEN_KW)
    L = enc['input_ids'].shape[1]
    res = []
    for j in range(len(items)):
        try:
            d = extract_json(jt.decode(out[j, L:], skip_special_tokens=True))
            res.append({l: int(d.get(l, 0)) for l, _, _, _ in QUESTIONS})
        except Exception:
            res.append({l: 0 for l, _, _, _ in QUESTIONS})
    return res

from tqdm.auto import tqdm
recs = []
for i in tqdm(range(0, len(rows), a.bs)):
    sub = rows[i:i+a.bs]
    try:
        scs = judge_batch(sub)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); scs = []
        for r in sub:
            try: scs += judge_batch([r])
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); scs.append({l: 0 for l, _, _, _ in QUESTIONS})
    for r, sc in zip(sub, scs):
        for l, v in sc.items():
            recs.append({'session_id': r['session_id'], 'turn': r['turn'], 'cond': r['cond'], 'item': l, 'score': v})
out = pd.DataFrame(recs)
out.to_parquet(OUT/f'scores_{a.shard}.parquet')
print(f"saved -> {OUT}/scores_{a.shard}.parquet | parse failures: {(out['score']==0).sum()//7}", flush=True)
