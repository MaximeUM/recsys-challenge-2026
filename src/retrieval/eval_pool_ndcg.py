"""POOL-ONLY nDCG@20 (without reranking) on dev: the reranking table baseline.
Single-GT -> nDCG@20 = 1/log2(GT_rank+1), or 0 when the GT is outside the top 20.
The denominator includes ALL turns (GT outside the pool -> zero contribution).

    python src/retrieval/eval_pool_ndcg.py
    python src/retrieval/eval_pool_ndcg.py --pool exp/combined_pool_ctx1024_dev.parquet
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--pool', default='exp/combined_pool_ctx1024_dev.parquet')
a = ap.parse_args()

df = pd.read_parquet(a.pool)
N = len(df)

def rank_gt(r):
    pool = json.loads(r['pool'])
    try:
        return pool.index(r['gt_idx']) + 1
    except ValueError:
        return 999

ranks = df.apply(rank_gt, axis=1)
ndcg = np.where(ranks <= 20, 1.0 / np.log2(ranks + 1), 0.0).mean()

print(f'{a.pool}: {N} turns')
print(f'  GT in pool  : {(ranks < 999).mean()*100:.1f}%')
print(f'  GT in top 20: {(ranks <= 20).mean()*100:.1f}%')
print(f'  nDCG@20 (pool order, without reranker) = {ndcg:.4f}')
