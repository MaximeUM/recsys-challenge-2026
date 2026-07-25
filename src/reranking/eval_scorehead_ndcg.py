"""Combine scoring-head picks (exp/picks/picks_<name>_*.parquet) and compute full-dev nDCG@20
in TWO modes:
  - TOP-1: argmax pick first + remainder in pool order (comparable to the generative firstpos reranker).
  - TOP-20: 20 recommendations from the COMPLETE model-score ranking (GT rank = gt_rank_full).
Single-GT -> nDCG@20 = 1/log2(GT_rank+1).

The combiner is strict by default: it requires the expected number of shards and
exactly 8,000 distinct dev keys with no duplicates, so a partial run or a stale
shard left over from a previous evaluation cannot silently print a plausible
score. Relax with --expect_shards 0 --expect_rows 0 for ad-hoc subsets.

    python src/reranking/eval_scorehead_ndcg.py --name scorehead
"""
import argparse, glob, sys
import numpy as np, pandas as pd
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--name',default='scorehead')
ap.add_argument('--expect_shards',type=int,default=4,help='required shard count (0 = no check)')
ap.add_argument('--expect_rows',type=int,default=8000,help='required unique dev keys (0 = no check)')
args=ap.parse_args()
P=Path('exp/picks')
fs=sorted(glob.glob(str(P/f'picks_{args.name}_*.parquet')))
if not fs:
    sys.exit(f'no shard matches {P}/picks_{args.name}_*.parquet')
if args.expect_shards and len(fs)!=args.expect_shards:
    sys.exit(f'expected {args.expect_shards} shards, found {len(fs)}:\n  '
             +'\n  '.join(fs)+'\nclear exp/picks/ and rerun, or pass --expect_shards')
df=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
N=len(df); nkey=df['key'].nunique()
if nkey!=N:
    dup=df['key'].value_counts(); dup=dup[dup>1]
    sys.exit(f'{N-nkey} duplicate key(s) across shards (e.g. {list(dup.index[:3])}); '
             'a stale shard is probably mixed in - clear exp/picks/ and rerun')
if args.expect_rows and nkey!=args.expect_rows:
    sys.exit(f'expected {args.expect_rows} unique dev keys, found {nkey}; '
             'the shards do not cover the full dev split')
ginp=int((df['gt_pos']>=0).sum())
# Denominator = ALL turns (GT outside pool -> zero contribution).
print(f'{args.name}: {N} turns | GT in pool {ginp} ({ginp/N*100:.1f}%) | {len(fs)} shards',flush=True)
def dcg(rank): return 1.0/np.log2(rank+1) if rank<=20 else 0.0
# TOP-1: pick first, remainder in pool order -> GT rank.
def rank_top1(r):
    # List = [pick] + (pool order without pick). GT is at gt_pos.
    if r['gt_pos']<0: return 999
    if r['pick']==r['gt_pos']: return 1
    return r['gt_pos']+1 if r['pick']<r['gt_pos'] else r['gt_pos']+2
nd1=np.mean([dcg(rank_top1(r)) for _,r in df.iterrows()])
nd20=np.mean([dcg(int(r['gt_rank_full'])) if r['gt_pos']>=0 else 0.0 for _,r in df.iterrows()])
print('\n=== nDCG@20 (FULL-DEV) - scoring head ===')
print(f'  TOP-1  (pick + pool order) : {nd1:.4f}')
print(f'  TOP-20 (full score ranking): {nd20:.4f}')
print(f'\n  (generative reference: intent+mm 0.1718 | previous 0.1681)')
