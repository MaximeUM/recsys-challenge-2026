"""Combine picks (exp/picks/picks_<name>_*.parquet) and compute full-dev nDCG@20:
Qwen alone, Llama alone, Qwen+Llama vote. Single-GT -> nDCG@20 = 1/log2(GT_rank+1).

Strict by default, like eval_scorehead_ndcg.py: each name must contribute the
expected number of shards and exactly 8,000 distinct dev keys with no
duplicates, so a partial or stale run cannot print a plausible score. Relax with
--expect_shards 0 --expect_rows 0 for ad-hoc subsets.

    python src/reranking/eval_firstpos_ndcg.py --names qwen,llama
"""
import argparse, glob, sys
from collections import Counter
import numpy as np, pandas as pd
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--names', default='qwen,llama')
ap.add_argument('--expect_shards', type=int, default=4, help='required shard count per name (0 = no check)')
ap.add_argument('--expect_rows', type=int, default=8000, help='required unique dev keys per name (0 = no check)')
args=ap.parse_args()
NAMES=[n for n in args.names.split(',') if n]
P=Path('exp/picks')

# pick per (name, key)
picks={n:{} for n in NAMES}; meta={}
for n in NAMES:
    fs=sorted(glob.glob(str(P/f'picks_{n}_*.parquet')))
    if not fs:
        sys.exit(f'no shard matches {P}/picks_{n}_*.parquet')
    if args.expect_shards and len(fs)!=args.expect_shards:
        sys.exit(f'{n}: expected {args.expect_shards} shards, found {len(fs)}:\n  '
                 +'\n  '.join(fs)+'\nclear exp/picks/ and rerun, or pass --expect_shards')
    df=pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    nkey=df['key'].nunique()
    if nkey!=len(df):
        dup=df['key'].value_counts(); dup=dup[dup>1]
        sys.exit(f'{n}: {len(df)-nkey} duplicate key(s) across shards '
                 f'(e.g. {list(dup.index[:3])}); a stale shard is probably mixed in')
    if args.expect_rows and nkey!=args.expect_rows:
        sys.exit(f'{n}: expected {args.expect_rows} unique dev keys, found {nkey}; '
                 'the shards do not cover the full dev split')
    for _,r in df.iterrows():
        picks[n][r['key']]=int(r['pick'])
        meta[r['key']]=(int(r['gt_pos']), int(r['n_cand']))
    print(f'{n}: {len(df)} picks ({len(fs)} shards)', flush=True)

keys=[k for k in meta if all(k in picks[n] for n in NAMES)]
if args.expect_rows and len(keys)!=args.expect_rows:
    sys.exit(f'only {len(keys)} keys are common to {NAMES}, expected {args.expect_rows}')
print(f'{len(keys)} common turns | GT in pool: {np.mean([meta[k][0]>=0 for k in keys]):.3f}', flush=True)

def ndcg(scheme):
    tot=0.0
    for k in keys:
        gt_pos,n=meta[k]
        if gt_pos<0: continue
        votes=Counter()
        for nm in scheme:
            p=picks[nm][k]
            if 0<=p<n: votes[p]+=1
        placed=[p for p,_ in sorted(votes.items(), key=lambda kv:(-kv[1],kv[0]))]
        seen=set(placed)
        for p in range(n):
            if p not in seen: placed.append(p)
        rank=placed.index(gt_pos)+1
        if rank<=20: tot+=1.0/np.log2(rank+1)
    return tot/len(keys)

print('\n=== nDCG@20 (FULL-DEV) ===')
for n in NAMES: print(f'  {n+" alone":<16}: {ndcg([n]):.4f}')
print(f'  {"Vote "+"+".join(NAMES):<16}: {ndcg(NAMES):.4f}')
