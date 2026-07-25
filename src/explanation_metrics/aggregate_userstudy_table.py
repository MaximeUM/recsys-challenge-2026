"""Aggregate the user-study judge panel into the published table.

Reads the raw per-turn judge scores produced by userstudy_dimensions.py, applies
the published protocol (full dev split, minus the 91 `Unknown message`
placeholder turns -> 7909 paired turns), averages the thinking-mode seeds, and
emits both a canonical CSV and the Markdown table used in the documentation.

The documentation table must never be transcribed by hand: regenerate it here.

    python src/explanation_metrics/aggregate_userstudy_table.py
    python src/explanation_metrics/aggregate_userstudy_table.py --keep_unknown
    python src/explanation_metrics/aggregate_userstudy_table.py --out_csv exp/userstudy_dimensions_fulldev.csv
"""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd

# questionnaire item -> dimension (two items each for transparency/effectiveness)
DIMS = {'A': 'Transparency', 'B': 'Transparency', 'C': 'Effectiveness',
        'D': 'Effectiveness', 'E': 'Persuasion', 'F': 'Trust', 'G': 'Satisfaction'}
ORDER = ['Transparency', 'Effectiveness', 'Persuasion', 'Trust', 'Satisfaction']
JUDGES = {'gemma4': 'gemma-4-E2B', 'qwen3': 'Qwen3-8B', 'llama': 'Llama-3.2-3B'}
# column order of the published table
COLS = [('gemma4', 'direct'), ('gemma4', 'think'), ('qwen3', 'direct'),
        ('qwen3', 'think'), ('llama', 'direct')]

ap = argparse.ArgumentParser()
ap.add_argument('--exp_dir', default='exp', help='directory holding intrs_dims_fulldev_* runs')
ap.add_argument('--dev_parquet',
                default='data/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
ap.add_argument('--out_csv', default='exp/userstudy_dimensions_fulldev.csv')
ap.add_argument('--out_md', default=None, help='also write the Markdown table here')
ap.add_argument('--keep_unknown', action='store_true',
                help='diagnostic variant: keep the 91 placeholder turns (8000 turns)')
args = ap.parse_args()

# --- the 91 placeholder turns excluded by the published protocol -------------
placeholders = set()
dev = pd.read_parquet(args.dev_parquet)
for _, r in dev.iterrows():
    for t in r['conversations']:
        if str(t.get('content', '')).strip() == 'Unknown message':
            placeholders.add((str(r['session_id']), int(t['turn_number'])))
print(f'placeholder turns found: {len(placeholders)}')

# --- aggregate every judge run ----------------------------------------------
rows = []
for d in sorted(glob.glob(f'{args.exp_dir}/intrs_dims_fulldev_*')):
    shards = sorted(glob.glob(f'{d}/scores_[0-9]*.parquet'))
    if not shards:
        continue
    tag = Path(d).name.replace('intrs_dims_fulldev_', '')
    m = re.match(r'^(gemma4|qwen3|llama)(_think(?:_s(\d+))?)?$', tag)
    if not m:
        print(f'  skipping unrecognised run: {tag}')
        continue
    judge, mode, seed = m.group(1), ('think' if m.group(2) else 'direct'), m.group(3)

    df = pd.concat([pd.read_parquet(f) for f in shards])
    ok = df[df['score'] > 0].assign(dim=lambda x: x['item'].map(DIMS))
    piv = ok.pivot_table(index=['session_id', 'turn', 'cond'],
                         columns='dim', values='score').reset_index()
    if not args.keep_unknown:
        key = list(zip(piv['session_id'].astype(str), piv['turn'].astype(int)))
        piv = piv[[k not in placeholders for k in key]]

    n = int((piv['cond'] == 'gen').sum())
    mean = piv.groupby('cond')[ORDER].mean()
    for dim in ORDER:
        rows.append({'judge': JUDGES[judge], 'judge_key': judge, 'mode': mode,
                     'seed': seed or '', 'dim': dim, 'n_turns': n,
                     'ours': mean.loc['gen', dim], 'gold': mean.loc['gold', dim],
                     'delta': mean.loc['gen', dim] - mean.loc['gold', dim]})
    print(f'  {tag:24} n={n} turns/condition')

raw = pd.DataFrame(rows)
if raw.empty:
    raise SystemExit(f'no runs found under {args.exp_dir}/intrs_dims_fulldev_*')

# thinking mode: average across seeds
agg = (raw.groupby(['judge', 'judge_key', 'mode', 'dim'], as_index=False)
          .agg(ours=('ours', 'mean'), gold=('gold', 'mean'),
               delta=('delta', 'mean'), n_turns=('n_turns', 'max'),
               n_seeds=('seed', 'nunique')))

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
agg.sort_values(['judge_key', 'mode', 'dim']).to_csv(args.out_csv, index=False,
                                                     float_format='%.4f')
print(f'\nwrote {args.out_csv}')

# --- Markdown table ----------------------------------------------------------
idx = {(r.judge_key, r.mode, r.dim): r for r in agg.itertuples()}
head = ('| Dimension | ' + ' | '.join(
    f'{JUDGES[j]}{" think" if m == "think" else " direct"}' for j, m in COLS) + ' |')
lines = [head, '|---' * (len(COLS) + 1) + '|']
for dim in ORDER:
    cells = []
    for j, m in COLS:
        r = idx.get((j, m, dim))
        cells.append(f'**{r.ours:.2f}** / {r.gold:.2f}' if r else '—')
    lines.append(f'| {dim} | ' + ' | '.join(cells) + ' |')
md = '\n'.join(lines)
print('\n' + md)

n_all = sorted(agg['n_turns'].unique())
print(f'\nturns per condition: {n_all}   (published protocol: 7909)')
print(f'all deltas favour ours: {bool((agg["delta"] > 0).all())} '
      f'| min {agg["delta"].min():.3f} | max {agg["delta"].max():.3f}')
print(f'gold range: {agg["gold"].min():.2f} - {agg["gold"].max():.2f}')

if args.out_md:
    Path(args.out_md).write_text(md + '\n')
    print(f'wrote {args.out_md}')
