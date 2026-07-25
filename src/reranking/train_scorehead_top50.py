"""Listwise SCORING-HEAD reranker, an alternative to the generative firstpos reranker.
Instead of generating the best candidate index as text, place a marker token after every candidate,
read its hidden state (one forward pass, joint causal context), and apply a linear head to obtain one
score per candidate. Train with listwise cross-entropy using the GT as positive. Uses the SAME dataset
as the generative model (sft_ctx1024_enriched_firstpos.parquet: sound|themes-enriched candidates) and
the SAME Llama-3.2-3B + LoRA base.

    accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 src/reranking/train_scorehead_top50.py \
        --output models/llama32_3b_scorehead_ctx1024 --max_seq_len 4096
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from pathlib import Path
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import PartialState
BASE='meta-llama/Llama-3.2-3B-Instruct'; N_EPOCHS,LORA_R,LR=2,16,2e-4
MARK_STR='<|reserved_special_token_5|>'; MARK_ID=128013
SYS=("You are an expert music recommender. You get a user profile, a conversation goal, the conversation history "
     "ending with the user's CURRENT request, and candidate tracks. Each candidate is annotated: "
     "[community tags] {sound: genre/feel inferred from the AUDIO | themes: lyrical themes} and ends with a score "
     "marker. RATE how well each candidate fits the user's CURRENT request. CRITICAL: honor what the user wants RIGHT "
     "NOW and what they explicitly REJECT — match the sound/genre/themes to their current intent; a track whose "
     "sound/themes contradict what they asked for (or said they do NOT want) must score low.")
def build_user(profile,goal,conv,marked,n):
    return (f"User profile: {profile}\nConversation goal: {goal}\n\nConversation:\n{conv}\n\n"
            f"Candidate tracks ({n} candidates, 1-based; each ends with a score marker):\n{marked}")

def make_tok(tok,ml):
    tok.truncation_side='left'
    def f(rec):
        lines=[l for l in rec['candidates'].split('\n') if l.strip()]; n=len(lines)
        marked='\n'.join(l+MARK_STR for l in lines)
        gt=int(str(rec['target']).strip().strip('[]'))-1  # Zero-based in the complete numbering.
        user=build_user(rec['user_profile'],rec['conversation_goal'],rec['conversation'],marked,n)
        text=tok.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':user}],
                                     tokenize=False,add_generation_prompt=False)
        ids=tok(text,add_special_tokens=False,truncation=True,max_length=ml)['input_ids']
        pos=[i for i,t in enumerate(ids) if t==MARK_ID]; count=len(pos)
        # left truncation => surviving candidates = suffix [n-count, n)
        base_idx=n-count
        if count==0 or gt<base_idx or gt>=n: return {'input_ids':[],'marker_pos':[],'gt_local':-1}
        return {'input_ids':ids,'marker_pos':pos,'gt_local':gt-base_idx}
    return f

class Coll:  # per_device_bs=1
    def __call__(self,b):
        x=b[0]
        return {'input_ids':torch.tensor([x['input_ids']]),
                'attention_mask':torch.ones(1,len(x['input_ids']),dtype=torch.long),
                'marker_pos':torch.tensor(x['marker_pos']),
                'gt_local':torch.tensor(x['gt_local'])}

class RankScorer(nn.Module):
    def __init__(self,base,H):
        super().__init__(); self.base=base; self.config=base.config
        self.head=nn.Linear(H,1,bias=False).to(torch.bfloat16)
    def forward(self,input_ids,attention_mask,marker_pos,gt_local):
        h=self.base(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[0]  # [L,H]
        s=self.head(h[marker_pos]).squeeze(-1)  # [C]
        loss=F.cross_entropy(s.float().unsqueeze(0),gt_local.view(1))
        return {'loss':loss,'logits':s.detach()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='models/_sft_dataset_cache/sft_ctx1024_enriched_firstpos.parquet')
    ap.add_argument('--output',default='models/llama32_3b_scorehead_ctx1024')
    ap.add_argument('--max_seq_len',type=int,default=4096)
    ap.add_argument('--per_device_bs',type=int,default=1); ap.add_argument('--grad_accum',type=int,default=8)
    a=ap.parse_args(); st=PartialState(); im=st.is_main_process
    df=pd.read_parquet(a.cache)
    if im: print(f'[main] {len(df):,} examples | listwise scoring head | maxlen={a.max_seq_len}',flush=True)
    tok=AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    base=AutoModel.from_pretrained(BASE,torch_dtype=torch.bfloat16)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); base.enable_input_require_grads()
    base=get_peft_model(base,LoraConfig(r=LORA_R,lora_alpha=LORA_R*2,lora_dropout=0.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none'))
    H=base.config.hidden_size; mdl=RankScorer(base,H)
    ds=Dataset.from_pandas(df).map(make_tok(tok,a.max_seq_len),remove_columns=df.columns.tolist(),num_proc=4)
    ds=ds.filter(lambda x:x['gt_local']>=0 and 10<=len(x['input_ids'])<=a.max_seq_len)
    if im: print(f'[main] after tokenization/filtering: {len(ds):,}',flush=True)
    targs=TrainingArguments(output_dir=str(Path(a.output)/'ckpt'),num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=a.per_device_bs,gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=False,learning_rate=LR,lr_scheduler_type='cosine',warmup_ratio=0.05,
        bf16=True,logging_steps=50,save_strategy='no',report_to=[],ddp_find_unused_parameters=False,
        dataloader_num_workers=4,remove_unused_columns=False,label_names=['gt_local'])
    tr=Trainer(model=mdl,args=targs,train_dataset=ds,data_collator=Coll())
    if im: print('[main] Training...',flush=True)
    tr.train()
    if im:
        out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
        w=tr.accelerator.unwrap_model(tr.model)
        w.base.merge_and_unload().save_pretrained(str(out))
        torch.save({'head':w.head.state_dict(),'hidden':H,'marker_id':MARK_ID},out/'score_head.pt')
        tok.save_pretrained(str(out)); print(f'[main] Saved -> {out}',flush=True)
    st.wait_for_everyone()
if __name__=='__main__': main()
