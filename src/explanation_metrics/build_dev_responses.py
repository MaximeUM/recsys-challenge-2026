"""Build a dev sample for explanation metrics as submission-format JSON files.
Use the selected Qwen3-8B scorehead top-1 pick from exp/picks/picks_shqwen8b_* over N_SESS randomly
sampled dev sessions (fixed seed), split into four shards for four-GPU generation through
src/response/convbestof20_diverse.py with --blind_parquet pointing to the dev Parquet file.

    python src/explanation_metrics/build_dev_responses.py --n_sess 50
"""
import argparse, glob, json, random
import numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--n_sess', type=int, default=50)
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--nshards', type=int, default=4)
ap.add_argument('--prefix', default='shqwen8b_respsample', help='output filename prefix')
ap.add_argument('--picks_name', default='shqwen8b',
                help="name passed to predict_scorehead_dev.py --name (reads exp/picks/picks_<name>_*.parquet)")
a = ap.parse_args()

DATA = Path('data')
dev = pd.read_parquet(DATA/'TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
tids = tm['track_id'].tolist()

_pat = f'exp/picks/picks_{a.picks_name}_*.parquet'
_files = sorted(glob.glob(_pat))
if not _files:
    raise SystemExit(f'no pick shards found for {_pat!r}; run predict_scorehead_dev.py '
                     f'with --name {a.picks_name}, or adjust --picks_name')
picks = pd.concat([pd.read_parquet(f) for f in _files], ignore_index=True)
picks = picks.set_index('key')
pool = pd.read_parquet('exp/combined_pool_ctx1024_dev.parquet')
pool['key'] = pool['session_id'] + '|' + pool['turn'].astype(str)
pool = pool.set_index('key')

rng = random.Random(a.seed)
sess_ids = sorted(dev['session_id'].tolist())
sample = rng.sample(sess_ids, a.n_sess)
uid = dev.set_index('session_id')['user_id']

shards = [[] for _ in range(a.nshards)]
missing = 0
for si, sid in enumerate(sample):
    for t in range(1, 9):
        k = f'{sid}|{t}'
        if k not in picks.index or k not in pool.index:
            missing += 1; continue
        cand = json.loads(pool.loc[k, 'pool'])[:int(picks.loc[k, 'n_cand'])]
        top1 = tids[cand[int(picks.loc[k, 'pick'])]]
        shards[si % a.nshards].append({
            'session_id': sid, 'user_id': uid.loc[sid], 'turn_number': t,
            'predicted_track_ids': [top1], 'predicted_response': ''})

OUT = Path('exp/inference/devset')
OUT.mkdir(parents=True, exist_ok=True)
tot = 0
for i, sh in enumerate(shards):
    p = OUT/f'{a.prefix}_shard{i}.json'
    json.dump(sh, open(p, 'w'), ensure_ascii=False, indent=1)
    print(f'{p}: {len(sh)} turns'); tot += len(sh)
print(f'total {tot} turns | missing {missing} | sessions {a.n_sess} (seed {a.seed})')
