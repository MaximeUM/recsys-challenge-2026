"""How often does the ground truth share an artist with an already-played track?

This is the statistic that motivates the artist-expansion channel in the paper.
It is reported there as 46.0% of turns that have history, 40.3% overall, and it
is computed here so the claim is reproducible from the repository rather than
quoted from a notebook.

"Shares an artist" means a non-empty intersection between the two tracks' sets of
credited artists - the same rule the artist channel applies in
`build_pool_dev.py --artist_match shared`. That definition matters, because a
track credits a LIST of artists: 11.2% of the catalog credits more than one, and
on those rows `artist_name` and `artist_id` are not even the same length, so
"the artist of a track" is not well defined by taking one element.

`--compare` additionally prints the two weaker definitions, to document why the
numbers moved:

  first artist_name       taking element [0] of the name array only
  artist_id array string  exact equality of str(artist_id), which treats a
                          feature as a different artist from the soloist

    python src/retrieval/artist_overlap_stat.py
    python src/retrieval/artist_overlap_stat.py --compare
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('data')


def as_list(x):
    return list(x) if isinstance(x, (list, np.ndarray)) else [x]


def build_turns(dev, lookup):
    """(target track, history so far) for every turn with a ground truth."""
    turns = []
    for _, session in dev.iterrows():
        gt_by_turn = {int(t['turn_number']): t['content']
                      for t in session['conversations'] if t['role'] == 'music'}
        history = []
        for tn in sorted(gt_by_turn):
            target = gt_by_turn[tn]
            if target in lookup.index:
                turns.append((target, [h for h in history if h in lookup.index]))
            history.append(gt_by_turn[tn])
    return turns


def count_overlap(turns, key, intersect):
    hits = 0
    for target, history in turns:
        if intersect:
            if any(key[target] & key[h] for h in history):
                hits += 1
        elif any(key[target] == key[h] for h in history):
            hits += 1
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compare', action='store_true',
                    help='also print the two weaker definitions')
    ap.add_argument('--split', default='test-00000-of-00001.parquet',
                    help='dataset split file under TalkPlayData-Challenge-Dataset/data/')
    args = ap.parse_args()

    dev = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Dataset/data' / args.split)
    tm = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
    lookup = tm.set_index('track_id')

    n_multi = sum(1 for x in tm['artist_id'] if isinstance(x, (list, np.ndarray)) and len(x) > 1)
    print(f'catalog: {len(tm):,} tracks, {n_multi:,} ({100 * n_multi / len(tm):.1f}%) '
          f'crediting more than one artist')

    turns = build_turns(dev, lookup)
    # Turn 1 has no history, so it can never contribute a match. One session
    # contributes one such turn.
    with_history = sum(1 for _, h in turns if h)
    print(f'dev: {len(turns):,} turns, {with_history:,} with history\n')

    shared = {t: set(as_list(v)) for t, v in lookup['artist_id'].items()}
    rows = [('shared artist (paper, and --artist_match shared)',
             count_overlap(turns, shared, intersect=True))]

    if args.compare:
        first_name = {t: as_list(v)[0] for t, v in lookup['artist_name'].items()}
        array_str = {t: str(v) for t, v in lookup['artist_id'].items()}
        rows += [('first artist_name only', count_overlap(turns, first_name, False)),
                 ('artist_id array as a string', count_overlap(turns, array_str, False))]

    # Same order as the paper: turns that have history first, then overall.
    width = max(len(label) for label, _ in rows)
    print(f'{"definition":<{width}} {"turns":>7} {"with history":>13} {"overall":>9}')
    for label, hits in rows:
        print(f'{label:<{width}} {hits:>7,} {100 * hits / with_history:>12.1f}% '
              f'{100 * hits / len(turns):>8.1f}%')


if __name__ == '__main__':
    main()