"""Emit the paper's central TOP-1 / TOP-20 table as one versioned CSV.

`reproduce_paper.sh eval` prints each row as it is computed; this collects all
four into a single dated file so a run can be diffed against the published
numbers instead of read off a log.

Same arithmetic as eval_firstpos_ndcg.py and eval_scorehead_ndcg.py, and the
same strictness: every row must come from exactly four shards covering 8,000
distinct dev keys, or the script fails rather than reporting a partial figure.

    python src/reranking/table1_summary.py
    python src/reranking/table1_summary.py --out exp/table1_rerun.csv
"""
import argparse
import datetime
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PICKS = Path('exp/picks')

# (label, run name, kind) - kind picks the TOP-1/TOP-20 convention
ROWS = [
    ('Generative firstpos (intent+mm)', 'intentmm', 'firstpos'),
    ('Scoring head Llama-3.2-3B, top-50', 'sh_llama_top50', 'scorehead'),
    ('Scoring head Llama-3.2-3B, top-200', 'sh_llama_top200', 'scorehead'),
    ('Scoring head Qwen3-8B, top-200', 'shqwen8b', 'scorehead'),
]

# Rounded value printed in the paper, and the exact figure behind it. Deltas are
# taken against the exact one; rounding the reference first would make a run look
# 0.0008 off when it is 0.0005 off.
PUBLISHED = {
    'intentmm':        (0.172, 0.172, 0.17179910388520142, 0.17179910388520142),
    'sh_llama_top50':  (0.175, 0.202, 0.17485280165364536, 0.20185241256833059),
    'sh_llama_top200': (0.178, 0.216, 0.17765772398890173, 0.21593342797602416),
    'shqwen8b':        (0.177, 0.219, 0.17697462589275176, 0.21859846122644389),
}
TOL = 0.001  # bf16 near-ties move picks by about this much between runs


def dcg(rank):
    return 1.0 / np.log2(rank + 1) if rank <= 20 else 0.0


def load(name, expect_shards, expect_rows):
    fs = sorted(glob.glob(str(PICKS / f'picks_{name}_*.parquet')))
    if not fs:
        sys.exit(f'{name}: no shard matches {PICKS}/picks_{name}_*.parquet')
    if expect_shards and len(fs) != expect_shards:
        sys.exit(f'{name}: expected {expect_shards} shards, found {len(fs)}')
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if df['key'].nunique() != len(df):
        sys.exit(f'{name}: duplicate keys across shards - clear {PICKS}/ and rerun')
    if expect_rows and len(df) != expect_rows:
        sys.exit(f'{name}: expected {expect_rows} unique dev keys, found {len(df)}')
    return df, len(fs)


def rank_top1(r):
    """[pick] followed by the pool order minus the pick."""
    if r['gt_pos'] < 0:
        return 999
    if r['pick'] == r['gt_pos']:
        return 1
    return r['gt_pos'] + 1 if r['pick'] < r['gt_pos'] else r['gt_pos'] + 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None,
                    help='default: exp/table1_<today>.csv')
    ap.add_argument('--expect_shards', type=int, default=4)
    ap.add_argument('--expect_rows', type=int, default=8000)
    args = ap.parse_args()

    out = Path(args.out or
               f'exp/table1_{datetime.date.today().isoformat()}.csv')

    rows = []
    for label, name, kind in ROWS:
        df, nshards = load(name, args.expect_shards, args.expect_rows)
        # Denominator = ALL turns; a GT outside the pool contributes zero.
        top1 = float(np.mean([dcg(rank_top1(r)) for _, r in df.iterrows()]))
        if kind == 'scorehead':
            top20 = float(np.mean([dcg(int(r['gt_rank_full'])) if r['gt_pos'] >= 0 else 0.0
                                   for _, r in df.iterrows()]))
        else:
            # A single-pick reranker leaves positions 2..20 in pool order, so
            # TOP-20 is TOP-1 by construction.
            top20 = top1
        nan = float('nan')
        pub1, pub20, ex1, ex20 = PUBLISHED.get(name, (nan, nan, nan, nan))
        rows.append({'reranker': label, 'run_name': name, 'n_turns': len(df),
                     'n_shards': nshards,
                     'gt_in_pool': float((df['gt_pos'] >= 0).mean()),
                     'top1': top1, 'top20': top20,
                     'published_top1': pub1, 'published_top20': pub20,
                     'exact_top1': ex1, 'exact_top20': ex20,
                     'delta_top1': top1 - ex1, 'delta_top20': top20 - ex20})

    table = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f'\nSaved -> {out}')

    off = table[(table['delta_top1'].abs() > TOL) | (table['delta_top20'].abs() > TOL)]
    if len(off):
        print(f'\nWARNING - rows further than {TOL} from the exact published figure:')
        print(off[['reranker', 'top1', 'top20', 'exact_top1', 'exact_top20',
                   'delta_top1', 'delta_top20']].to_string(index=False))
    else:
        print(f'\nAll rows within {TOL} of the exact published figures.')


if __name__ == '__main__':
    main()
