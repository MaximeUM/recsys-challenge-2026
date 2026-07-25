"""vLLM port of src/response/convbestof20_diverse.py with the same prompts, clean/diversity filters,
and selection; only the inference backend changes (vLLM continuous batching is about 5–10x faster).
One process runs BOTH phases on one GPU: best-of-N generation (gemma-3n-E4B,
n=N in one call, seed 2026), then judging (gemma-4-E2B, greedy).

    CUDA_VISIBLE_DEVICES=0 python src/response/convbestof20_vllm.py \
        --input exp/inference/devset/shqwen8b_respsample_shard0.json \
        --blind_parquet data/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet \
        --out exp/inference/devset/shqwen8b_respsample_shard0_convbestofN.json --n 20
"""
import os, argparse, json, re, warnings, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
import numpy as np, pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True); ap.add_argument('--blind_parquet', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--gen', default='google/gemma-3n-E4B-it'); ap.add_argument('--judge', default='google/gemma-4-E2B-it')
ap.add_argument('--seed', type=int, default=2026)
a = ap.parse_args()

b = pd.read_parquet(a.blind_parquet)
tm = pd.read_parquet('data/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name', 'artist_name']:
    tm[c] = tm[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x).astype(str)
lk = tm.set_index('track_id'); sess = {s['session_id']: s for _, s in b.iterrows()}
norm = lambda s: re.sub(r'[\W_]+', ' ', str(s).lower(), flags=re.UNICODE).strip()  # Unicode-aware (Cyrillic, CJK, accents); treat _ as a separator.
def nm(t): return f"{lk.loc[t,'track_name']} by {lk.loc[t,'artist_name']}" if t in lk.index else str(t)
def tg(t):
    if t not in lk.index: return ''
    tl = lk.loc[t, 'tag_list']; return ', '.join(list(tl)[:5]) if isinstance(tl, (list, np.ndarray)) and len(tl) > 0 else ''
def full_conv(cs, tt):
    o = {'user': 0, 'music': 1, 'assistant': 2}; L = []
    for t in sorted(cs, key=lambda x: (int(x['turn_number']), o[x['role']])):
        if int(t['turn_number']) > tt: break
        if int(t['turn_number']) == tt and t['role'] != 'user': continue
        if t['role'] == 'user': L.append(f"User: {t['content']}")
        elif t['role'] == 'music': L.append(f"Assistant played: {nm(t['content'])}")
        elif t['role'] == 'assistant': L.append(f"Assistant: {t['content']}")
    return '\n'.join(L)
def conv_norm(e):
    cs = sess[e['session_id']]['conversations']; tt = int(e['turn_number']); P = []
    for t in cs:
        if int(t['turn_number']) > tt: break
        P.append(nm(t['content']) if t['role'] == 'music' else str(t['content']))
    P.append(nm(e['predicted_track_ids'][0])); return norm(' || '.join(P))

# Prompts identical to convbestof20_diverse.py (v2).
GEN_SYS = ("You are the assistant in an ongoing music chat. Write your next reply (~45 words) about the given track.\n"
 "- ADDRESS what the user actually wants RIGHT NOW: if they are trying to identify or remember a specific song/album, "
 "present the track as the answer (e.g. 'That's almost certainly \"X\"…'); if they are exploring, recommend it.\n"
 "- Explain concretely WHY it fits (genre, energy, mood, era, artist, lyrics), coherent with the whole conversation.\n"
 "- Do NOT begin with filler ('Okay', 'I understand', 'I hear you', 'Got it', 'Awesome', 'Alright'). Open on something "
 "specific and VARY your phrasing.\n- Warm, natural, specific — never formulaic.")
FEWSHOT = [
 ("Conversation:\nUser: I'm trying to remember a moody synth track from a 2010s sci-fi film, kind of pulsing and melancholic.\n"
  "Recommend: Sequence by S U R V I V E (tags: synthwave, dark, instrumental)",
  "That pulsing, melancholic synth you're picturing is almost certainly \"Sequence\" by S U R V I V E — the band behind the Stranger Things score. "
  "Its analog arpeggios and brooding low end nail that 2010s sci-fi unease you're describing."),
 ("Conversation:\nUser: Loved that last folk track. Give me something with the same warmth but a bit more upbeat.\n"
  "Recommend: Ho Hey by The Lumineers (tags: indie folk, singalong, upbeat)",
  "\"Ho Hey\" by The Lumineers keeps that folk warmth but lifts the tempo — stomp-and-clap rhythm, a big singalong hook, and that same intimate, "
  "campfire feel you just enjoyed, now brighter and more buoyant."),
]
def gctx(e):
    top = e['predicted_track_ids'][0]
    return f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'], int(e['turn_number']))}\nRecommend: {nm(top)}" + (f" (tags: {tg(top)})" if tg(top) else '')
QUOTE = re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”')
def clean(cand, e):
    top = e['predicted_track_ids'][0]; tnn = norm(lk.loc[top, 'track_name']); ar = norm(lk.loc[top, 'artist_name']); r = norm(cand)
    ar0 = ar.split()[0] if ar.split() else ''
    names_rec = tnn in r or (len(tnn) >= 4 and ' '.join(tnn.split()[:3]) in r) or (ar0 in r and len(ar0) >= 4)
    ctx = conv_norm(e)
    for m in QUOTE.finditer(cand):
        q = norm(m.group(1) or m.group(2))
        if len(q) < 3: continue
        if not (q in ctx or any(w in ctx for w in q.split() if len(w) >= 5)): return False
    return names_rec
def opening(c): return ' '.join(norm(c).split()[:3])

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

data = json.load(open(a.input))
CANDS = a.out + '.cands.json'
if os.path.exists(CANDS):
    data = json.load(open(CANDS)); print(f'Phase 1 SKIPPED: candidates loaded from {CANDS}', flush=True)
else:
    gt = AutoTokenizer.from_pretrained(a.gen)
    prompts = []
    for e in data:
        msgs = [{'role': 'system', 'content': GEN_SYS}]
        for u, as_ in FEWSHOT: msgs += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': as_}]
        msgs.append({'role': 'user', 'content': gctx(e)})
        prompts.append(gt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    print(f'Phase 1 (vLLM): {len(data)} turns x {a.n} candidates...', flush=True)
    gen = LLM(model=a.gen, dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.9, seed=a.seed)
    sp = SamplingParams(n=a.n, temperature=0.95, top_p=0.95, max_tokens=100, seed=a.seed)
    outs = gen.generate(prompts, sp)
    for e, o in zip(data, outs):
        e['_cands'] = [c.text.strip().replace('\n', ' ') for c in o.outputs]
    del gen; gc.collect()
    import torch; torch.cuda.empty_cache()
    json.dump(data, open(CANDS, 'w'), ensure_ascii=False); print(f'candidates saved -> {CANDS}', flush=True)

JSYS = ("You evaluate the assistant's LAST reply, two dims (ignore track correctness): Personalization (1-5) + "
 "Explanation Quality (1-5). Be critical. Output EXACTLY: Personalization: <n>, Explanation: <n>")
NUM = re.compile(r'Personalization:\s*([1-5]).*?Explanation:\s*([1-5])', re.S)
jt = AutoTokenizer.from_pretrained(a.judge)
jprompts = [jt.apply_chat_template([{'role': 'system', 'content': JSYS}, {'role': 'user', 'content':
    f"Conversation:\n{full_conv(sess[e['session_id']]['conversations'], int(e['turn_number']))}\n\nRecommended track: "
    f"{nm(e['predicted_track_ids'][0])}\n\nAssistant's last reply to evaluate:\n{c}\n\nScores:"}],
    tokenize=False, add_generation_prompt=True) for e in data for c in e['_cands']]
print(f'Phase 2 (vLLM): judging {len(jprompts)} candidates...', flush=True)
judge = LLM(model=a.judge, dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.9, seed=a.seed)
jouts = judge.generate(jprompts, SamplingParams(temperature=0.0, max_tokens=20))
scores = []
for o in jouts:
    m = NUM.search(o.outputs[0].text); scores.append((int(m.group(1)) + int(m.group(2))) if m else 0)
it = iter(scores)
for e in data: e['_sc'] = [next(it) for _ in e['_cands']]

# Diversity-aware selection identical to convbestof20_diverse.py.
used = Counter(); nclean = 0; tot = 0.0
for e in data:
    cs = e['_cands']; sc = e['_sc']; idx = [i for i in range(len(cs)) if clean(cs[i], e)]
    pool = idx if idx else list(range(len(cs)))
    if idx: nclean += 1
    MARGIN = 1  # On the 2–10 scale, trade quality for novelty only within this margin.
    max_sc = max(sc[i] for i in pool)
    near_best = [i for i in pool if sc[i] >= max_sc - MARGIN]
    fresh = [i for i in near_best if used[opening(cs[i])] == 0]
    cand = fresh if fresh else near_best
    best = max(cand, key=lambda i: sc[i])
    used[opening(cs[best])] += 1; tot += sc[best]
    e['predicted_response'] = cs[best]; e.pop('_cands', None); e.pop('_sc', None)
import sys; sys.path.insert(0, 'music-crs-evaluator')
try:
    from metrics import compute_lexical_diversity; ld = compute_lexical_diversity([e['predicted_response'] for e in data])
except Exception: ld = -1
ndist = len(set(opening(e['predicted_response']) for e in data))
print(f"clean: {nclean}/{len(data)} | mean local judge={tot/len(data):.2f}/10 | lexical_div={ld:.4f} | distinct openings={ndist}/{len(data)}", flush=True)
json.dump(data, open(a.out, 'w'), ensure_ascii=False, indent=2); print('Saved ->', a.out, flush=True)
