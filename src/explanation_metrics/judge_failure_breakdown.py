"""Track scoring failures by judge, split between generated responses and gold responses.
This is intended for the appendix. Two levels are available depending on run output:
  - ALL judges (scores_*.parquet, with a 'cond' column from the start): number of zero-score
    rows remaining in the FINAL file for each condition.
  - Judges run with retry-on-failure (diag_*.parquet, added July 16): also report how many
    evaluations needed at least one retry and how many remain unrecoverable after three attempts.
    This exposes the REAL pre-correction failure rate, whereas scores_*.parquet only shows failures
    remaining after correction.

    python src/explanation_metrics/judge_failure_breakdown.py
"""
import glob
import pandas as pd

rows = []
for d in sorted(glob.glob('exp/intrs_dims_fulldev_*')):
    tag = d.split('exp/intrs_dims_fulldev_')[-1]
    sfiles = sorted(glob.glob(f'{d}/scores_[0-9]*.parquet'))
    if not sfiles:
        continue
    sc = pd.concat([pd.read_parquet(f) for f in sfiles], ignore_index=True)
    # One row per (session, turn, condition, item); a zero item means that evaluation failed.
    # Summarize at (session, turn, condition): failure if at least one item is zero.
    ev = sc.groupby(['session_id', 'turn', 'cond'])['score'].apply(lambda s: (s == 0).any()).reset_index()
    fail_gen = int(ev[(ev['cond'] == 'gen') & ev['score']].shape[0])
    fail_gold = int(ev[(ev['cond'] == 'gold') & ev['score']].shape[0])
    n_gen = int((ev['cond'] == 'gen').sum()); n_gold = int((ev['cond'] == 'gold').sum())
    row = {'judge': tag, 'n_gen': n_gen, 'final_failures_gen': fail_gen,
           'n_gold': n_gold, 'final_failures_gold': fail_gold}

    dfiles = sorted(glob.glob(f'{d}/diag_[0-9]*.parquet'))
    if dfiles:
        dg = pd.concat([pd.read_parquet(f) for f in dfiles], ignore_index=True)
        for cond in ['gen', 'gold']:
            sub = dg[dg['cond'] == cond]
            row[f'retried_{cond}'] = int((sub['n_retries'] > 0).sum())
            row[f'unrecoverable_{cond}'] = int(sub['unrecoverable'].sum())
    rows.append(row)

df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
df.to_csv('exp/judge_failure_breakdown.csv', index=False)
print('\nSaved -> exp/judge_failure_breakdown.csv')
