"""Best-of-N v2 — responses GROUNDED IN THE FULL CONVERSATION.

Observation: the judge penalizes explanations that ignore the conversation flow.
Provide BOTH generator and judge with the FULL conversation (user messages,
assistant responses, and played tracks through the prediction point) so the
explanation remains consistent with the exchange and recommendation.

Generator: gemma-3n-E4B (best-of-N). Judge: gemma-4-E2B (Personalization +
Explanation Quality with conversation context). track_ids remain unchanged.

Usage :
    CUDA_VISIBLE_DEVICES=0 python src/response/convbestofn_judge.py --input exp/inference/blindset_A/sft_combined_blindA.json --gen google/gemma-3n-E4B-it --n 6

Changelog: add --blind_parquet (default = Blind-A) to rerun the method on Blind-B
without editing code. Blind-A behavior is unchanged when the argument is omitted.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, re, warnings, sys, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')
sys.path.insert(0, 'music-crs-evaluator')
from metrics import compute_lexical_diversity
torch.manual_seed(42)

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='exp/inference/blindset_A/sft_combined_blindA.json')
ap.add_argument('--output', default=None)
ap.add_argument('--n', type=int, default=6)
ap.add_argument('--gen', default='google/gemma-3n-E4B-it')
ap.add_argument('--blind_parquet', default='data/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet',
                help='blind-set conversation Parquet file; for Blind-B, pass its path')
args = ap.parse_args()
SUB_IN = Path(args.input); SUB_OUT = Path(args.output) if args.output else SUB_IN.with_name(SUB_IN.stem + '_convbestofN.json')
N = args.n
DATA = Path('data'); GEN = args.gen; JUDGE = 'google/gemma-4-E2B-it'; MAXLEN = 3072

blind = pd.read_parquet(args.blind_parquet)
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name']:
    tm[c]=tm[c].apply(lambda x: x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
lk = tm.set_index('track_id'); sess = {s['session_id']: s for _, s in blind.iterrows()}

def name(t):
    if t not in lk.index: return t
    r=lk.loc[t]; return f"{r['track_name']} by {r['artist_name']}"
def tags(t):
    if t not in lk.index: return ''
    tl=lk.loc[t,'tag_list']; return ', '.join(list(tl)[:5]) if isinstance(tl,(list,np.ndarray)) and len(tl)>0 else ''
def profile_str(s):
    up=s.get('user_profile') or {}
    return '; '.join(f'{lab}: {up.get(key)}' for key,lab in
        [('age_group','age'),('country_name','country'),('preferred_language','language'),('preferred_musical_culture','musical taste')] if up.get(key))
def full_conv(cs, tt):
    order={'user':0,'music':1,'assistant':2}
    lines=[]
    for t in sorted(cs, key=lambda x:(x['turn_number'], order[x['role']])):
        if t['turn_number']>tt: break
        if t['turn_number']==tt and t['role']!='user': continue  # At the target turn, keep only the user request.
        if t['role']=='user': lines.append(f"User: {t['content']}")
        elif t['role']=='music': lines.append(f"Assistant played: {name(t['content'])}")
        elif t['role']=='assistant': lines.append(f"Assistant: {t['content']}")
    return '\n'.join(lines)

# Phase 1: conversation-grounded generation (gemma-3n-E4B).
GEN_SYS = ("You are the assistant in an ongoing music chat. Read the FULL conversation, then write your next reply "
           "(~45 words) recommending the given track. Your reply MUST: (1) be coherent with the conversation so far — "
           "build on what the user has asked, liked or refined across turns (e.g. 'Since you enjoyed X and now want Y…'); "
           "(2) explain concretely WHY this track fits, with specific musical reasons (genre, energy, mood, era, artist, "
           "lyrics). Warm, natural, specific — never generic.")
FEWSHOT = [
    ("Conversation:\nUser: I'm in the mood for some classic alternative rock.\nAssistant played: American Idiot by Green Day\n"
     "Assistant: Here's American Idiot by Green Day — a punk-rock anthem!\nUser: Loved that. Something else from that early-2000s era?\n"
     "Recommend: Jesus Of Suburbia by Green Day (tags: pop punk, alternative rock, 2000s)",
     "Since American Idiot hit the spot, let's stay on that album with \"Jesus Of Suburbia\" by Green Day. It's a sprawling "
     "nine-minute pop-punk epic from the same early-2000s era — same anthemic energy and storytelling you just loved, taken even further."),
    ("Conversation:\nUser: I want something intense and dramatic to discover.\n"
     "Recommend: The Fiend by Alesana (tags: screamo, post-hardcore, dramatic)",
     "For something intense and dramatic to discover, try \"The Fiend\" by Alesana. Its theatrical screamo builds and "
     "post-hardcore dynamics deliver exactly that dark, emotional drama you're after — a gripping first taste of the band."),
]
def gen_ctx(s, e):
    top=e['predicted_track_ids'][0]
    ctx = f"Conversation:\n{full_conv(s['conversations'], e['turn_number'])}"
    p=profile_str(s)
    if p: ctx += f"\n(User profile: {p})"
    ctx += f"\nRecommend: {name(top)}" + (f" (tags: {tags(top)})" if tags(top) else '')
    return ctx

print(f'Phase 1: generating candidates ({GEN})...', flush=True)
gtok = AutoTokenizer.from_pretrained(GEN); gtok.pad_token = gtok.pad_token or gtok.eos_token
gtok.padding_side='left'; gtok.truncation_side='left'
gen = AutoModelForCausalLM.from_pretrained(GEN, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
cands = []
for e in tqdm(json.load(open(SUB_IN))):
    s=sess[e['session_id']]
    msgs=[{'role':'system','content':GEN_SYS}]
    for u,a in FEWSHOT: msgs += [{'role':'user','content':u},{'role':'assistant','content':a}]
    msgs.append({'role':'user','content':gen_ctx(s,e)})
    text=gtok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    enc=gtok(text,return_tensors='pt',truncation=True,max_length=MAXLEN).to('cuda:0')
    with torch.no_grad():
        out=gen.generate(**enc,max_new_tokens=100,do_sample=True,temperature=0.9,top_p=0.95,
                         num_return_sequences=N,pad_token_id=gtok.eos_token_id)
    outs=[gtok.decode(o[enc['input_ids'].shape[1]:],skip_special_tokens=True).strip().replace('\n',' ') for o in out]
    cands.append({'e':e,'s':s,'cands':outs})
del gen; torch.cuda.empty_cache()

# Phase 2: conversation-aware gemma-4-E2B judge.
print('Phase 2: Gemma judge...', flush=True)
jtok = AutoTokenizer.from_pretrained(JUDGE); jtok.truncation_side='left'
judge = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
JSYS = ("You evaluate the assistant's LAST reply in a music chat, on TWO text dimensions (ignore whether the track is "
        "objectively correct):\n"
        "- Personalization (1-5): does the reply reflect THIS conversation — what the user asked, liked and refined "
        "across turns — rather than being generic?\n"
        "- Explanation Quality (1-5): does it clearly and specifically explain WHY the recommended track fits, with "
        "concrete musical reasons, coherent with the conversation?\n"
        "Be critical and spread scores. Output EXACTLY: Personalization: <n>, Explanation: <n>")
NUM = re.compile(r'Personalization:\s*([1-5]).*?Explanation:\s*([1-5])', re.S)
@torch.no_grad()
def score(s, e, resp):
    ctx=(f"Conversation:\n{full_conv(s['conversations'], e['turn_number'])}\n\n"
         f"Recommended track: {name(e['predicted_track_ids'][0])}\n\n"
         f"Assistant's last reply to evaluate:\n{resp}\n\nScores:")
    text=jtok.apply_chat_template([{'role':'system','content':JSYS},{'role':'user','content':ctx}],tokenize=False,add_generation_prompt=True)
    enc=jtok(text,return_tensors='pt',truncation=True,max_length=MAXLEN).to('cuda:0')
    out=judge.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=jtok.eos_token_id)
    g=jtok.decode(out[0,enc['input_ids'].shape[1]:],skip_special_tokens=True)
    m=NUM.search(g); return (int(m.group(1))+int(m.group(2))) if m else 0

subs=[]; tot=0.0
for item in tqdm(cands):
    e,s=item['e'],item['s']
    best=max(((score(s,e,r),r) for r in item['cands']), key=lambda x:x[0]); tot+=best[0]
    e=dict(e); e['predicted_response']=best[1]; subs.append(e)

ld=compute_lexical_diversity([e['predicted_response'] for e in subs])
print(f'conv-best-of-{N} | mean local judge={tot/len(subs):.2f}/10 | lexical_diversity={ld:.4f}', flush=True)
for e in subs[:3]: print(' ', e['predicted_response'][:220], flush=True)
with open(SUB_OUT,'w',encoding='utf-8') as f: json.dump(subs,f,ensure_ascii=False, indent=2)
print(f'Saved -> {SUB_OUT}', flush=True)
