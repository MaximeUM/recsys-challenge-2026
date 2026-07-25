"""INTENT + MULTIMODAL-aware single-pick reranker. Use the same firstpos data with ENRICHED
candidates ({sound: audio genre | themes: lyrics}) and a prompt emphasizing the current intent/negation.
Base = Llama-3.2-3B (comparable to the current reranker). Loss applies only to firstpos.

    accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 src/reranking/train_firstpos_intent_mm.py
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from pathlib import Path
import pandas as pd, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import PartialState
BASE='meta-llama/Llama-3.2-3B-Instruct'; N_EPOCHS,LORA_R,LR=2,16,2e-4
SYS=("You are an expert music recommender. You get a user profile, a conversation goal, the conversation history "
     "ending with the user's CURRENT request, and candidate tracks. Each candidate is annotated: "
     "[community tags] {sound: genre/feel inferred from the AUDIO | themes: lyrical themes}. "
     "Pick THE single best track for the user's CURRENT request. CRITICAL: honor what the user wants RIGHT NOW and "
     "what they explicitly REJECT — match the sound/genre/themes to their current intent; NEVER pick a track whose "
     "sound/themes contradict what they asked for or said they do NOT want. "
     "Output ONLY a JSON array with that one candidate index (1-based), e.g. [12]. Nothing else.")
def build_user(rec,n):
    return (f"User profile: {rec['user_profile']}\nConversation goal: {rec['conversation_goal']}\n\n"
            f"Conversation:\n{rec['conversation']}\n\n"
            f"Candidate tracks ({n} candidates, 1-based indices; each has [tags] {{sound | themes}}):\n{rec['candidates']}\n\n"
            f"Pick the single best track for the user's CURRENT request, respecting what they want and reject. Output JSON array with one index.")
def make_tok(tok,n,ml):
    def f(rec):
        msgs=[{'role':'system','content':SYS},{'role':'user','content':build_user(rec,n)}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); comp=rec['target']+tok.eos_token
        pids=tok(p,add_special_tokens=False)['input_ids']; cids=tok(comp,add_special_tokens=False)['input_ids']
        mp=ml-len(cids)-1
        if len(pids)>mp: pids=pids[:100]+pids[-(mp-100):]
        return {'input_ids':pids+cids,'labels':[-100]*len(pids)+cids}
    return f
class Pad:
    def __init__(s,t): s.p=t.pad_token_id or t.eos_token_id
    def __call__(s,b):
        m=max(len(x['input_ids']) for x in b); ii,ll,aa=[],[],[]
        for x in b:
            q=m-len(x['input_ids']); ii.append(x['input_ids']+[s.p]*q); ll.append(x['labels']+[-100]*q); aa.append([1]*len(x['input_ids'])+[0]*q)
        return {'input_ids':torch.tensor(ii),'labels':torch.tensor(ll),'attention_mask':torch.tensor(aa)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='models/_sft_dataset_cache/sft_ctx1024_enriched_firstpos.parquet')
    ap.add_argument('--output',default='models/llama32_3b_intent_mm_ctx1024')
    ap.add_argument('--max_seq_len',type=int,default=3584); ap.add_argument('--n_cand',type=int,default=50)
    ap.add_argument('--per_device_bs',type=int,default=1); ap.add_argument('--grad_accum',type=int,default=8)
    a=ap.parse_args(); st=PartialState(); im=st.is_main_process
    df=pd.read_parquet(a.cache)
    if im: print(f'[main] {len(df):,} ex | enriched | maxlen={a.max_seq_len}',flush=True)
    tok=AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16)
    mdl.gradient_checkpointing_enable(); mdl.enable_input_require_grads()
    mdl=get_peft_model(mdl,LoraConfig(r=LORA_R,lora_alpha=LORA_R*2,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type=TaskType.CAUSAL_LM,bias='none'))
    ds=Dataset.from_pandas(df).map(make_tok(tok,a.n_cand,a.max_seq_len),remove_columns=df.columns.tolist(),num_proc=4)
    ds=ds.filter(lambda x:10<=len(x['input_ids'])<=a.max_seq_len)
    if im: print(f'[main] after tokenization: {len(ds):,}',flush=True)
    targs=TrainingArguments(output_dir=str(Path(a.output)/'ckpt'),num_train_epochs=N_EPOCHS,per_device_train_batch_size=a.per_device_bs,
        gradient_accumulation_steps=a.grad_accum,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        learning_rate=LR,lr_scheduler_type='cosine',warmup_ratio=0.05,bf16=True,logging_steps=50,save_strategy='no',report_to=[],
        ddp_find_unused_parameters=False,dataloader_num_workers=4,remove_unused_columns=False)
    tr=Trainer(model=mdl,args=targs,train_dataset=ds,data_collator=Pad(tok))
    if im: print('[main] Training...',flush=True)
    tr.train()
    if im:
        out=Path(a.output); out.mkdir(parents=True,exist_ok=True); tr.model.merge_and_unload().save_pretrained(str(out)); tok.save_pretrained(str(out)); print(f'[main] Saved -> {out}',flush=True)
    st.wait_for_everyone()
if __name__=='__main__': main()
