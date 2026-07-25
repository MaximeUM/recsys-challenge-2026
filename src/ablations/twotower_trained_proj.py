"""ASYMMETRIC two-tower model: fine-tuned 4B query encoder (LoRA, as in train_dense_encoder.py) -> FROZEN track tower
made from the six provided multimodal embeddings plus a small trainable track projection.

This gives the query side the same capacity as the dense retriever by training the 4B; only the track side remains
frozen because raw audio and images are unavailable.

Because the track tower is frozen and track_proj is tiny, project the ENTIRE 47k catalog at each step
and use it as negatives with full-softmax InfoNCE. This yields thousands of negatives without DDP
gathering; each GPU processes its own batch. The expensive 4B processes queries only.

Run (4 GPUs):
    accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 src/ablations/twotower_trained_proj.py
"""
import os, argparse, warnings, glob, random
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore'); torch.manual_seed(42); random.seed(42)

DATA=Path('data')
TASK='Given a music chat conversation, retrieve the track the user wants to listen to next'
def query_prompt(t): return f'Instruct: {TASK}\nQuery: {t}'

ap=argparse.ArgumentParser()
ap.add_argument('--base', default='Qwen/Qwen3-Embedding-4B')
ap.add_argument('--output', default='models/qwen3_qmm')
ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--max_seq_len', type=int, default=1024)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--proj_lr', type=float, default=1e-3)
ap.add_argument('--lora_r', type=int, default=16)
ap.add_argument('--dim', type=int, default=512)
ap.add_argument('--temp', type=float, default=0.05)
ap.add_argument('--dev_n', type=int, default=200)
args=ap.parse_args()

acc=Accelerator(mixed_precision='bf16', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
def log(m):
    if acc.is_main_process: print(f'[main] {m}', flush=True)

# Data.
train=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
tm=pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
for c in ['track_name','artist_name','album_name']:
    tm[c]=tm[c].apply(lambda x:x[0] if isinstance(x,(list,np.ndarray)) and len(x)>0 else x).astype(str)
tids=tm['track_id'].tolist(); tidx={t:i for i,t in enumerate(tids)}; N=len(tids); lk=tm.set_index('track_id')
def meta(tid):
    if tid not in lk.index: return tid
    r=lk.loc[tid]; return f"track_id: {tid}, track_name: {r['track_name']}, artist_name: {r['artist_name']}, album_name: {r['album_name']}"
def build_full_query(convs,tt):
    L=[]
    for t in convs:
        if t['turn_number']>=tt: break
        ro,co=t['role'],t['content']
        if ro=='music': ro,co='assistant',meta(co)
        L.append(f'{ro}: {co}')
    for t in convs:
        if t['turn_number']==tt and t['role']=='user': L.append(f"user: {t['content']}"); break
    return '\n'.join(L)
def build_pairs(df):
    anc,gt=[],[]
    for _,s in df.iterrows():
        gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
        for tn in range(1,9):
            g=gtbt.get(tn)
            if g is None or g not in tidx: continue
            anc.append(query_prompt(build_full_query(s['conversations'],tn).lower())); gt.append(tidx[g])
    return anc,gt
anchors,gts=build_pairs(train)
log(f'{len(anchors):,} train pairs')

# Frozen track tower (normalized multimodal concatenation).
emb=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA/'TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet')))],ignore_index=True).set_index('track_id')
MODS=['audio-laion_clap','image-siglip2','cf-bpr','attributes-qwen3_embedding_0.6b','lyrics-qwen3_embedding_0.6b','metadata-qwen3_embedding_0.6b']
def mat(col):
    dim=next(len(r) for r in emb[col] if isinstance(r,(list,np.ndarray)) and len(r)>0)
    M=np.zeros((N,dim),np.float32)
    for tid,row in emb[col].items():
        if tid in tidx and isinstance(row,(list,np.ndarray)) and len(row)==dim: M[tidx[tid]]=np.asarray(row,np.float32)
    return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
MM=np.concatenate([mat(m) for m in MODS],1); Dmm=MM.shape[1]
MMt=torch.tensor(MM, device=acc.device, dtype=torch.bfloat16)   # Frozen.
log(f'track tower: {MM.shape}')

# Models.
st=SentenceTransformer(args.base, model_kwargs={'torch_dtype':torch.bfloat16})
st.max_seq_length=args.max_seq_len; st.tokenizer.truncation_side='left'
st.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_r, lora_alpha=2*args.lora_r,
                          lora_dropout=0.05, target_modules=['q_proj','k_proj','v_proj','o_proj']))
try: st[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
except Exception:
    try: st[0].auto_model.gradient_checkpointing_enable()
    except Exception: pass
class Proj(nn.Module):
    def __init__(s,din,d): super().__init__(); s.net=nn.Sequential(nn.Linear(din,1024),nn.GELU(),nn.Linear(1024,d))
    def forward(s,x): z=s.net(x); return z/(z.norm(dim=-1,keepdim=True)+1e-9)
qproj=Proj(st.get_sentence_embedding_dimension(),args.dim)
tproj=Proj(Dmm,args.dim)

class DS(Dataset):
    def __init__(s,a,g): s.a=a; s.g=g
    def __len__(s): return len(s.a)
    def __getitem__(s,i): return s.a[i], s.g[i]
def collate(b):
    texts=[x[0] for x in b]; gt=torch.tensor([x[1] for x in b])
    feats=st.tokenize(texts); return feats, gt   # st stays the raw SentenceTransformer object (it has .tokenize)
dl=DataLoader(DS(anchors,gts), batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True)

opt=torch.optim.AdamW([
    {'params':[p for p in st.parameters() if p.requires_grad], 'lr':args.lr},
    {'params':list(qproj.parameters())+list(tproj.parameters()), 'lr':args.proj_lr},
], weight_decay=1e-4)

model, qp_d, tp_d, opt, dl = acc.prepare(st, qproj, tproj, opt, dl)  # Keep raw st for tokenization/evaluation.

# Training.
model.train(); qp_d.train(); tp_d.train()
steps=len(dl)*args.epochs; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
gstep=0
for ep in range(args.epochs):
    for feats, gt in dl:
        feats={k:(v.to(acc.device) if torch.is_tensor(v) else v) for k,v in feats.items()}; gt=gt.to(acc.device)
        q=model(feats)['sentence_embedding']                    # (B,2560) with gradients.
        zq=qp_d(q)                                              # (B,d) normalized.
        zt=tp_d(MMt)                                            # (N,d) projected catalog.
        logits=zq@zt.T/args.temp                                # (B,N) full-softmax
        loss=nn.functional.cross_entropy(logits, gt)
        acc.backward(loss); opt.step(); sched.step(); opt.zero_grad()
        gstep+=1
        if acc.is_main_process and gstep%50==0:
            log(f'step {gstep}/{steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}')

acc.wait_for_everyone()
# ---------- save ----------
if acc.is_main_process:
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    u_st=acc.unwrap_model(model); u_q=acc.unwrap_model(qp_d); u_t=acc.unwrap_model(tp_d)
    u_st.save_pretrained(str(out/'encoder'))   # Query encoder with adapter.
    torch.save(u_q.state_dict(), out/'qproj.pt')
    torch.save(u_t.state_dict(), out/'tproj.pt')
    np.save(out/'mm.npy', MM)
    log(f'saved -> {out}')

    # Inline dev evaluation (200).
    dev=pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
    items=[]
    for _,s in dev.iterrows():
        gtbt={t['turn_number']:t['content'] for t in s['conversations'] if t['role']=='music'}
        for tn in range(1,9):
            g=gtbt.get(tn)
            if g is None or g not in tidx: continue
            items.append((query_prompt(build_full_query(s['conversations'],tn).lower()), tidx[g]))
    if args.dev_n and args.dev_n<len(items): items=random.Random(0).sample(items,args.dev_n)
    u_st.eval(); u_q.eval(); u_t.eval()
    with torch.no_grad():
        qe=u_st.encode([x[0] for x in items], batch_size=16, convert_to_tensor=True, device=acc.device,
                        normalize_embeddings=False).to(torch.bfloat16)
        zq=u_q(qe); zt=u_t(MMt)
        sims=zq@zt.T
        gtd=torch.tensor([x[1] for x in items], device=acc.device)
        r=((sims>sims.gather(1,gtd[:,None])).sum(1)+1).cpu().numpy()
    ndcg=np.mean([1/np.log2(x+1) if x<=20 else 0 for x in r])
    log(f'DEV(n={len(items)}) LEARNED query->mm : nDCG@20 {ndcg:.4f}  R@20 {np.mean(r<=20):.4f}  R@50 {np.mean(r<=50):.4f}  R@500 {np.mean(r<=500):.4f}')
    log('(reference: dense ctx1024 baseline on 200 candidates, nDCG@20 0.1592)')
