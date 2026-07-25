"""Approach A — fine-tuned cross-encoder with a listwise "find the GT among 50" objective.

For each turn, score (query, candidate) for all 50 candidates in the combined
pool, apply softmax over the 50 scores, and use cross-entropy with the GT as the
only positive. No ordering is imposed on non-GT candidates; the 49 hard
distractors serve as negatives.

Base: BAAI/bge-reranker-v2-m3 (the same model that failed zero-shot; fine-tuned here).
Data: the top-50 reranker set (conversation + 50 candidates + GT position).
Output: models/crossenc_reranker + dev predictions (official format) for evaluation.

Run:
    accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 --mixed_precision bf16 src/ablations/crossencoder_train.py
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import re, json, warnings, math
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
warnings.filterwarnings('ignore')

BASE   = 'BAAI/bge-reranker-v2-m3'
CACHE  = Path('models/_sft_dataset_cache/sft_combined_60000_top50.parquet')
OUT    = Path('models/crossenc_reranker')
N_CAND, MAXLEN, EPOCHS, BS, LR, GA = 50, 384, 2, 2, 3e-5, 4   # effective batch = BS*procs*GA = 32 queries

acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=GA)
is_main = acc.is_main_process

def build_query(up, goal, conv):
    h = []
    if up:   h.append(f"User profile: {up}")
    if goal: h.append(f"Goal: {goal}")
    return ('\n'.join(h) + '\n' if h else '') + f"Conversation:\n{conv}"

def strip_idx(line):  # "12. name by artist [tags]" -> "name by artist [tags]"
    return re.sub(r'^\s*\d+\.\s*', '', line).strip()

class CE_DS(Dataset):
    def __init__(self, df):
        self.q, self.passages, self.gt = [], [], []
        for _, r in df.iterrows():
            cands = [strip_idx(l) for l in r['candidates'].split('\n') if l.strip()][:N_CAND]
            if len(cands) != N_CAND:  # Keep only records with 50 candidates.
                continue
            gt = json.loads(r['target'])[0] - 1
            if not (0 <= gt < N_CAND):
                continue
            self.q.append(build_query(r['user_profile'], r['conversation_goal'], r['conversation']))
            self.passages.append(cands); self.gt.append(gt)
    def __len__(self): return len(self.q)
    def __getitem__(self, i): return self.q[i], self.passages[i], self.gt[i]

tok = AutoTokenizer.from_pretrained(BASE)
tok.truncation_side = 'left'   # Keep the end of the conversation (the request).

def collate(batch):
    qs, ps, gts = zip(*batch)
    flat_q, flat_p = [], []
    for q, plist in zip(qs, ps):
        flat_q += [q] * N_CAND
        flat_p += plist
    enc = tok(flat_q, flat_p, padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt')
    return enc, torch.tensor(gts, dtype=torch.long)

df = pd.read_parquet(CACHE)
ds = CE_DS(df)
if is_main: print(f'{len(ds)} records (50 cands)', flush=True)
dl = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)

model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
model, opt, dl = acc.prepare(model, opt, dl)
opt_steps_total = math.ceil(len(dl) / GA) * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * opt_steps_total), opt_steps_total)

if is_main: print(f'Training cross-encoder ({opt_steps_total} opt steps, batch eff={BS*acc.num_processes*GA})...', flush=True)
model.train()
opt_step = 0
for ep in range(EPOCHS):
    for enc, gts in dl:
        with acc.accumulate(model):
            logits = model(**enc).logits.squeeze(-1).view(gts.size(0), N_CAND)
            loss = F.cross_entropy(logits, gts.to(logits.device))
            acc.backward(loss)
            opt.step(); opt.zero_grad()
        if acc.sync_gradients:
            sched.step(); opt_step += 1
            if is_main and opt_step % 50 == 0:
                print(f'  ep{ep+1} opt_step {opt_step}/{opt_steps_total} loss={loss.item():.4f}', flush=True)

acc.wait_for_everyone()
if is_main:
    OUT.mkdir(parents=True, exist_ok=True)
    acc.unwrap_model(model).save_pretrained(str(OUT)); tok.save_pretrained(str(OUT))
    print(f'Saved -> {OUT}', flush=True)
