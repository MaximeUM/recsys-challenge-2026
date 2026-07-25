"""train a single-pick reranker (loss on the first/only position).

The model learns to output THE best candidate (its index), not a ranking of 20.
Target = JSON [GT_pos]. At inference, place this pick at rank 1 and
fill ranks 2–20 in pool order (neutral for single-GT nDCG).

    accelerate launch --num_processes N --gpu_ids ... src/reranking/train_firstpos.py \\
        --cache models/_sft_dataset_cache/sft_4b_60000_top50_firstpos.parquet \\
        --output models/llama32_3b_firstpos_top50 --max_seq_len 2048 --n_cand 50
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import PartialState

LLM_CANDIDATES = ['meta-llama/Llama-3.2-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct']
N_EPOCHS, LORA_R, LR = 2, 16, 2e-4

def system_prompt(n_cand):
    return (
        "You are an expert music recommender. Given a user profile, a conversation goal, "
        f"a conversation history ending with a user request, and a list of {n_cand} candidate tracks, "
        "pick THE single best track for the user's final request. "
        "Output ONLY a JSON array with that one candidate index (1-based), e.g. [12]. "
        "Do not output anything else."
    )

def build_user_message(rec, n_cand):
    return (f"User profile: {rec['user_profile']}\n"
            f"Conversation goal: {rec['conversation_goal']}\n\n"
            f"Conversation:\n{rec['conversation']}\n\n"
            f"Candidate tracks ({n_cand} candidates, 1-based indices):\n{rec['candidates']}\n\n"
            f"Pick the single best track. Output JSON array with one index.")

def make_tok_fn(tok, n_cand, max_seq_len):
    sysp = system_prompt(n_cand)
    def f(rec):
        msgs = [{'role': 'system', 'content': sysp}, {'role': 'user', 'content': build_user_message(rec, n_cand)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        comp = rec['target'] + tok.eos_token
        pids = tok(prompt, add_special_tokens=False)['input_ids']
        cids = tok(comp, add_special_tokens=False)['input_ids']
        maxp = max_seq_len - len(cids) - 1
        if len(pids) > maxp:
            pids = pids[:100] + pids[-(maxp - 100):]
        return {'input_ids': pids + cids, 'labels': [-100] * len(pids) + cids}
    return f

class PadCollator:
    def __init__(self, tok): self.pad = tok.pad_token_id or tok.eos_token_id
    def __call__(self, batch):
        m = max(len(x['input_ids']) for x in batch)
        ii, ll, aa = [], [], []
        for x in batch:
            p = m - len(x['input_ids'])
            ii.append(x['input_ids'] + [self.pad] * p); ll.append(x['labels'] + [-100] * p)
            aa.append([1] * len(x['input_ids']) + [0] * p)
        return {'input_ids': torch.tensor(ii), 'labels': torch.tensor(ll), 'attention_mask': torch.tensor(aa)}

def load_llm():
    for name in LLM_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
            if tok.pad_token_id is None: tok.pad_token = tok.eos_token
            print(f'[load] {name}', flush=True)
            return mdl, tok
        except Exception as e:
            print(f'[load] {name} failed: {e}', flush=True)
    raise RuntimeError('no LLM')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max_seq_len', type=int, default=2048)
    ap.add_argument('--n_cand', type=int, default=50)
    ap.add_argument('--per_device_bs', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=4)
    args = ap.parse_args()
    state = PartialState(); is_main = state.is_main_process
    df = pd.read_parquet(args.cache)
    if is_main: print(f'[main] {len(df):,} records | procs={state.num_processes} | n_cand={args.n_cand} | maxlen={args.max_seq_len}', flush=True)
    model, tok = load_llm()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_R*2, lora_dropout=0.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj'], task_type=TaskType.CAUSAL_LM, bias='none'))
    if is_main:
        try: model.print_trainable_parameters()
        except Exception: pass
    ds = Dataset.from_pandas(df)
    ds = ds.map(make_tok_fn(tok, args.n_cand, args.max_seq_len), remove_columns=ds.column_names, num_proc=2)
    ds = ds.filter(lambda x: 10 <= len(x['input_ids']) <= args.max_seq_len)
    if is_main: print(f'[main] after tok+filter: {len(ds):,}', flush=True)
    targs = TrainingArguments(
        output_dir=str(Path(args.output) / 'checkpoints'),
        num_train_epochs=N_EPOCHS, per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, learning_rate=LR,
        lr_scheduler_type='cosine', warmup_ratio=0.05, bf16=True, logging_steps=50,
        save_strategy='no', report_to=[], ddp_find_unused_parameters=False,
        dataloader_num_workers=4, remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=PadCollator(tok))
    if is_main: print('[main] Training...', flush=True)
    trainer.train()
    if is_main:
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        model.merge_and_unload().save_pretrained(str(out)); tok.save_pretrained(str(out))
        print(f'[main] Saved -> {out}', flush=True)
    state.wait_for_everyone()

if __name__ == '__main__':
    main()
