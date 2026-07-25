"""MONITORED scoring-head reranker (LoRA + early stopping). Compared with the base scoring head:
- enriched TOP-200 dataset (sft_ctx1024_enriched_top200_firstpos.parquet),
- 80/20 train/val split BY SESSION (leakage prevention: all conversation turns stay in one set),
- periodic validation loss + EARLY STOPPING, saving the BEST LoRA adapter without merging during training,
- train/val curves in loss_curve.png and loss_history.json,
- finally merge the best adapter into a loadable model directory for predict_scorehead_dev.py.

    accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 src/reranking/train_scorehead.py \
        --cache models/_sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet \
        --output models/llama32_3b_scorehead_top200 --max_seq_len 12288 --n_cand 200
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from accelerate.utils import set_seed
LORA_R=16
SYS=("You are an expert music recommender. You get a user profile, a conversation goal, the conversation history "
     "ending with the user's CURRENT request, and candidate tracks. Each candidate is annotated: "
     "[community tags] {sound: genre/feel inferred from the AUDIO | themes: lyrical themes} and ends with a score "
     "marker. RATE how well each candidate fits the user's CURRENT request. CRITICAL: honor what the user wants RIGHT "
     "NOW and what they explicitly REJECT — match the sound/genre/themes to their current intent; a track whose "
     "sound/themes contradict what they asked for (or said they do NOT want) must score low.")
def build_user(profile,goal,conv,marked,n):
    return (f"User profile: {profile}\nConversation goal: {goal}\n\nConversation:\n{conv}\n\n"
            f"Candidate tracks ({n} candidates, 1-based; each ends with a score marker):\n{marked}")

def make_tok(tok,ml,mark_str,mark_id):
    tok.truncation_side='left'
    def f(rec):
        lines=[l for l in rec['candidates'].split('\n') if l.strip()]; n=len(lines)
        marked='\n'.join(l+mark_str for l in lines)
        gt=int(str(rec['target']).strip().strip('[]'))-1
        user=build_user(rec['user_profile'],rec['conversation_goal'],rec['conversation'],marked,n)
        text=tok.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':user}],
                                     tokenize=False,add_generation_prompt=False)
        ids=tok(text,add_special_tokens=False,truncation=True,max_length=ml)['input_ids']
        pos=[i for i,t in enumerate(ids) if t==mark_id]; count=len(pos); base_idx=n-count
        if count==0 or gt<base_idx or gt>=n: return {'input_ids':[],'marker_pos':[],'gt_local':-1}
        return {'input_ids':ids,'marker_pos':pos,'gt_local':gt-base_idx}
    return f

def coll(b):
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
        h=self.base(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[0]
        s=self.head(h[marker_pos]).squeeze(-1)
        loss=F.cross_entropy(s.float().unsqueeze(0),gt_local.view(1))
        return {'loss':loss}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='models/_sft_dataset_cache/sft_ctx1024_enriched_top200_firstpos.parquet')
    ap.add_argument('--output',default='models/llama32_3b_scorehead_top200')
    ap.add_argument('--max_seq_len',type=int,default=14336); ap.add_argument('--n_cand',type=int,default=200)
    ap.add_argument('--max_records',type=int,default=0,help='>0 = subsample (smoke test)')
    ap.add_argument('--max_epochs',type=int,default=4); ap.add_argument('--patience',type=int,default=3)
    ap.add_argument('--eval_steps',type=int,default=300); ap.add_argument('--val_frac',type=float,default=0.2)
    ap.add_argument('--val_eval_size',type=int,default=1000); ap.add_argument('--min_delta',type=float,default=1e-4)
    ap.add_argument('--grad_accum',type=int,default=8); ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--grad_ckpt',type=int,default=1,help='1=gradient checkpointing (less memory), 0=off (faster, more memory)')
    ap.add_argument('--base',default='meta-llama/Llama-3.2-3B-Instruct',help='base model')
    ap.add_argument('--mark_str',default='<|reserved_special_token_5|>',help='single-token marker (Llama: reserved_special_token_5; Qwen: <|box_end|>)')
    a=ap.parse_args()
    acc=Accelerator(gradient_accumulation_steps=a.grad_accum,mixed_precision='bf16')
    set_seed(a.seed); im=acc.is_main_process
    out=Path(a.output); best_dir=out/'best_adapter'
    if im: out.mkdir(parents=True,exist_ok=True); print(f'[main] output -> {out}',flush=True)
    BASE=a.base; MARK_STR=a.mark_str
    tok=AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    MARK_ID=tok.convert_tokens_to_ids(MARK_STR)
    assert MARK_ID is not None and MARK_ID!=tok.unk_token_id and len(tok(MARK_STR,add_special_tokens=False)['input_ids'])==1, f"marker {MARK_STR} is not a single token for {BASE}"
    if im: print(f'[main] base={BASE} | marker={MARK_STR} (id {MARK_ID})',flush=True)
    df=pd.read_parquet(a.cache)
    assert 'session_id' in df.columns and 'turn_number' in df.columns, "session_id/turn_number are required"
    if a.max_records>0:
        df=df.sample(min(a.max_records,len(df)),random_state=a.seed).reset_index(drop=True)
        if im: print(f'[main] SMOKE: subsample {len(df)} examples',flush=True)
    # LOWMEM: ONLY rank 0 tokenizes to avoid RAM OOM from four parallel ranks and saves to disk;
    # other ranks load the memory-mapped Arrow data with negligible RAM use.
    tokdir=out/'_tok_cache'
    if im:
        ds=Dataset.from_pandas(df).map(make_tok(tok,a.max_seq_len,MARK_STR,MARK_ID),
            remove_columns=['user_profile','conversation_goal','conversation','candidates','target'],num_proc=4)
        ds=ds.filter(lambda x:x['gt_local']>=0 and 10<=len(x['input_ids'])<=a.max_seq_len)
        import shutil as _sh
        if tokdir.exists(): _sh.rmtree(tokdir)
        ds.save_to_disk(str(tokdir)); print(f'[main] tokenized dataset -> {tokdir} ({len(ds)} examples)',flush=True)
    acc.wait_for_everyone()
    ds=load_from_disk(str(tokdir))
    # ---- split BY SESSION (leakage prevention) ----
    sids=list(ds['session_id']); turns=list(ds['turn_number'])
    uniq=sorted(set(sids)); rng=random.Random(a.seed); rng.shuffle(uniq)
    nval=int(len(uniq)*a.val_frac); val_sess=set(uniq[:nval])
    tr_idx=[i for i,s in enumerate(sids) if s not in val_sess]
    va_idx=[i for i,s in enumerate(sids) if s in val_sess]
    if im:
        print(f'[main] {len(ds):,} ex | {len(uniq):,} sessions -> train {len(uniq)-nval} sess/{len(tr_idx)} ex | '
              f'val {nval} sess/{len(va_idx)} ex',flush=True)
        def dist(idx):
            c=np.bincount([turns[i] for i in idx],minlength=9)[1:9]; return (c/c.sum()*100).round(1)
        print('[main] turn% train:',dist(tr_idx).tolist(),'| val:',dist(va_idx).tolist(),flush=True)
    ds.set_format(type='python')
    train_ds=ds.select(tr_idx); val_ds=ds.select(va_idx)
    rng2=random.Random(a.seed); vsub=rng2.sample(range(len(va_idx)),min(a.val_eval_size,len(va_idx)))
    val_ds=val_ds.select(vsub)
    train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,collate_fn=coll,num_workers=2)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,collate_fn=coll,num_workers=2)
    # Model.
    base=AutoModel.from_pretrained(BASE,torch_dtype=torch.bfloat16)
    base.config.use_cache=False  # No KV cache during training (single forward pass).
    if a.grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); base.enable_input_require_grads()
    elif im: print('[main] gradient checkpointing DISABLED (faster, more memory)',flush=True)
    base=get_peft_model(base,LoraConfig(r=LORA_R,lora_alpha=LORA_R*2,lora_dropout=0.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none'))
    H=base.config.hidden_size; model=RankScorer(base,H)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    steps_per_epoch=max(1,len(train_loader)//a.grad_accum); total=steps_per_epoch*a.max_epochs
    sched=get_cosine_schedule_with_warmup(opt,int(0.03*total),total)
    model,opt,train_loader,val_loader,sched=acc.prepare(model,opt,train_loader,val_loader,sched)

    @torch.no_grad()
    def evaluate():
        model.eval(); tot=torch.zeros(1,device=acc.device); cnt=torch.zeros(1,device=acc.device)
        for vb in val_loader: tot+=model(**vb)['loss']; cnt+=1
        tot=acc.gather(tot).sum(); cnt=acc.gather(cnt).sum(); model.train()
        return (tot/cnt).item()

    def save_best():
        acc.wait_for_everyone()
        if im:
            uw=acc.unwrap_model(model)
            uw.base.save_pretrained(best_dir)  # LoRA adapter alone, NO merge
            torch.save({'head':uw.head.state_dict(),'hidden':H,'marker_id':MARK_ID},best_dir/'score_head.pt')
    def save_adapter(tag):  # SAVEALL: save the adapter (unmerged) under out/<tag>_adapter
        acc.wait_for_everyone()
        if im:
            d=out/f'{tag}_adapter'; uw=acc.unwrap_model(model)
            uw.base.save_pretrained(d)
            torch.save({'head':uw.head.state_dict(),'hidden':H,'marker_id':MARK_ID},d/'score_head.pt')
            print(f'[main] adapter {tag} saved -> {d}',flush=True)

    best=float('inf'); patience_ctr=0; hist=[]; gstep=0; run=[]; stop=False
    if im: print(f'[main] steps/epoch≈{steps_per_epoch} | eval every {a.eval_steps} steps | patience {a.patience}',flush=True)
    model.train()
    for epoch in range(a.max_epochs):
        for batch in train_loader:
            with acc.accumulate(model):
                loss=model(**batch)['loss']; acc.backward(loss); opt.step(); sched.step(); opt.zero_grad()
            run.append(loss.detach())
            if acc.sync_gradients:
                gstep+=1
                if gstep%a.eval_steps==0:
                    vl=evaluate()
                    tr=acc.gather(torch.stack(run)).mean().item(); run=[]
                    if im:
                        hist.append({'step':gstep,'epoch':epoch,'train_loss':tr,'val_loss':vl})
                        print(f'[step {gstep}] epoch {epoch} train_loss={tr:.4f} val_loss={vl:.4f} '
                              f'best={best:.4f} patience={patience_ctr}',flush=True)
                    improved=vl<best-a.min_delta
                    if improved: best=vl; patience_ctr=0; save_best()
                    else: patience_ctr+=1
                    if patience_ctr>=a.patience:
                        if im: print(f'[main] EARLY STOP (validation has not improved for {a.patience} evaluations)',flush=True)
                        stop=True; break
        save_adapter(f'epoch{epoch}')   # SAVEALL: one checkpoint per epoch
        if stop: break
    # Final evaluation when training ended before the next evaluation point.
    vl=evaluate()
    # NB: freeze the verdict BEFORE touching 'best', otherwise an improvement
    # observed at this final evaluation would never be saved.
    improved_final = vl < best - a.min_delta
    if im and (not hist or hist[-1]['step']!=gstep):
        hist.append({'step':gstep,'epoch':epoch,'train_loss':float('nan'),'val_loss':vl})
    if improved_final:
        best = vl
        save_best()
    acc.wait_for_everyone()
    if im:
        json.dump(hist,open(out/'loss_history.json','w'),indent=2)
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            xs=[h['step'] for h in hist]; tr_=[h['train_loss'] for h in hist]; vl_=[h['val_loss'] for h in hist]
            plt.figure(figsize=(9,5.5))
            plt.plot(xs,tr_,'-o',label='train loss'); plt.plot(xs,vl_,'-s',label='val loss')
            plt.axhline(best,ls='--',c='grey',lw=.8,label=f'best val={best:.4f}')
            ymax=max(v for v in tr_+vl_ if v==v)  # Ignore any NaN.
            prev=hist[0]['epoch']
            for i in range(1,len(hist)):
                if hist[i]['epoch']!=prev:
                    xb=(xs[i]+xs[i-1])/2; plt.axvline(xb,ls=':',c='red',lw=1.1,alpha=.7)
                    plt.text(xb,ymax,f"→ epoch {hist[i]['epoch']}",rotation=90,va='top',ha='right',fontsize=9,c='red'); prev=hist[i]['epoch']
            plt.xlabel('optimizer step'); plt.ylabel('listwise CE loss'); plt.legend(loc='lower left'); plt.grid(alpha=.3)
            plt.title('scorehead — train/val (split by session, epoch boundaries)')
            plt.tight_layout(); plt.savefig(out/'loss_curve.png',dpi=120)
            print(f'[main] curve -> {out/"loss_curve.png"}',flush=True)
        except Exception as e:
            print(f'[main] plot skip: {e}',flush=True)
        # SAVEALL: merge EVERY adapter (best + each epoch) -> out/<tag>/ subdirectories loadable by the predictors
        import shutil
        del model; torch.cuda.empty_cache()
        for d in sorted(out.glob('*_adapter')):
            tag=d.name[:-len('_adapter')]
            print(f'[main] merge {tag}...',flush=True)
            b2=AutoModel.from_pretrained(BASE,torch_dtype=torch.bfloat16)
            merged=PeftModel.from_pretrained(b2,str(d)).merge_and_unload()
            md=out/tag; merged.save_pretrained(str(md))
            shutil.copy(d/'score_head.pt',md/'score_head.pt'); tok.save_pretrained(str(md))
            del b2,merged; torch.cuda.empty_cache()
            print(f'[main] model {tag} -> {md}',flush=True)
        # Inference loads the ROOT <output> directory (predict_scorehead_dev/blind).
        # Promote the best checkpoint there: merged weights + tokenizer +
        # score_head.pt, then check that everything that will be loaded is present.
        src_best = out/'best'
        if src_best.is_dir():
            for fp in src_best.iterdir():
                if fp.is_file(): shutil.copy(fp, out/fp.name)
            json.dump({'selected':'best','val_loss':float(best),'base':BASE,
                       'hidden':H,'marker_id':MARK_ID},
                      open(out/'selection.json','w'),indent=2)
            need=['config.json','score_head.pt']
            miss=[n for n in need if not (out/n).exists()]
            sf=list(out.glob('*.safetensors'))
            if miss or not sf:
                raise SystemExit(f'[main] PROMOTION FAILED: missing={miss} safetensors={len(sf)}')
            ck=torch.load(out/'score_head.pt',map_location='cpu')
            assert ck['hidden']==H and ck['marker_id']==MARK_ID, 'score_head.pt is inconsistent'
            print(f'[main] best val_loss={best:.4f} | reloadable final model -> {out}',flush=True)
        else:
            raise SystemExit(f'[main] FAILED: {src_best} is missing, the root {out} '
                             'holds no model that inference can load')
    acc.wait_for_everyone()
if __name__=='__main__': main()
