"""Carve a small, self-consistent copy of the challenge data out of data/.

`scripts/reproduce_paper.sh` on the real data is a multi-hour job: 47k tracks to
encode, 60k reranker records, four LoRA trainings. This script writes a
miniature dataset with the *same directory layout*, so the whole chain can be
exercised on the same models and the same hardware in a fraction of the time. It
answers "does every stage still run and hand the right artifact to the next
one", never "are the numbers right".

What makes the sample representative rather than merely small:

  * sessions are sampled whole - all turns of a conversation stay together, so
    the co-occurrence channel, the session-level train/val split and the
    turn-level evaluation see the structure they see on the full data;
  * every track a kept session references (ground truth *and* listening history)
    stays in the catalog, otherwise retrieval recall would be structurally 0 and
    every downstream stage would train and evaluate on unreachable targets;
  * the catalog is then padded with random tracks up to --n_tracks, so the
    retriever has distractors to rank against (a catalog made only of ground
    truths makes recall@200 trivially 1.0 and the reranker's job vacuous);
  * track embeddings are filtered to that same track set, so the descriptor
    builders find the audio/lyrics vectors they expect.

Known and accepted distortion: build_content_descriptors.py only keeps tags with
a document frequency of 30 or more, so a small catalog produces sparser
`sound:` / `themes:` descriptors than the real one. That degrades descriptor
quality, not whether the stage runs. Bigger --n_tracks, denser descriptors.

    python scripts/make_mini_dataset.py                       # data/ -> data_mini/
    python scripts/make_mini_dataset.py --n_tracks 8000 --n_train 200
    python scripts/make_mini_dataset.py --src data --out /tmp/mini --seed 7

Used by scripts/smoke_test.sh, which runs the paper pipeline against the result.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASET   = 'TalkPlayData-Challenge-Dataset'
BLIND_A   = 'TalkPlayData-Challenge-Blind-A'
BLIND_B   = 'TalkPlayData-Challenge-Blind-B'
TRACK_MD  = 'TalkPlayData-Challenge-Track-Metadata'
TRACK_EMB = 'TalkPlayData-Challenge-Track-Embeddings'
USER_MD   = 'TalkPlayData-Challenge-User-Metadata'
USER_EMB  = 'TalkPlayData-Challenge-User-Embeddings'


def data_dir(root, repo):
    return Path(root) / repo / 'data'


def read_parquet(root, repo, name):
    """Read <root>/<repo>/data/<name>, or None when that split was not downloaded."""
    p = data_dir(root, repo) / name
    return pd.read_parquet(p) if p.exists() else None


def write_parquet(df, root, repo, name):
    p = data_dir(root, repo) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def played_track_ids(sessions):
    """Every track id a set of sessions references, in history or as ground truth."""
    ids = set()
    if sessions is None:
        return ids
    for convs in sessions['conversations']:
        for turn in convs:
            if turn['role'] == 'music' and turn['content']:
                ids.add(turn['content'])
    return ids


def sample_sessions(df, n, seed, label):
    """Whole sessions, reproducibly, keeping the file's original row order."""
    if df is None:
        return None
    if n >= len(df):
        print(f'  {label:<8} {len(df)} sessions (asked {n}, kept all)')
        return df.reset_index(drop=True)
    out = df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)
    print(f'  {label:<8} {len(out)} / {len(df)} sessions')
    return out


def gt_coverage(sessions, keep_ids):
    """Share of music turns whose track survived into the mini catalog."""
    total = hit = 0
    for convs in sessions['conversations']:
        for turn in convs:
            if turn['role'] == 'music' and turn['content']:
                total += 1
                hit += turn['content'] in keep_ids
    return hit, total


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='data', help='full dataset root (default: data)')
    ap.add_argument('--out', default='data_mini', help='mini dataset root (default: data_mini)')
    ap.add_argument('--n_train',  type=int, default=60, help='train sessions to keep')
    ap.add_argument('--n_dev',    type=int, default=40, help='dev (Dataset/test) sessions to keep')
    ap.add_argument('--n_blind',  type=int, default=20, help='Blind-A and Blind-B sessions to keep')
    ap.add_argument('--n_tracks', type=int, default=4000,
                    help='total catalog size: all referenced tracks, then random padding')
    ap.add_argument('--seed',     type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if src.resolve() == out.resolve():
        sys.exit('--src and --out must differ; this script never writes over the real data')
    if not (data_dir(src, DATASET) / 'train-00000-of-00001.parquet').exists():
        sys.exit(f'no dataset under {src}/ - run: python scripts/download_data.py')
    rng = np.random.default_rng(a.seed)

    # --- sessions ------------------------------------------------------------
    print(f'Sampling sessions from {src}/')
    train = sample_sessions(read_parquet(src, DATASET, 'train-00000-of-00001.parquet'),
                            a.n_train, a.seed, 'train')
    dev   = sample_sessions(read_parquet(src, DATASET, 'test-00000-of-00001.parquet'),
                            a.n_dev, a.seed, 'dev')
    blind = {}
    for repo, label in ((BLIND_A, 'blind-A'), (BLIND_B, 'blind-B')):
        df = sample_sessions(read_parquet(src, repo, 'test-00000-of-00001.parquet'),
                             a.n_blind, a.seed, label)
        if df is not None:
            blind[repo] = df

    write_parquet(train, out, DATASET, 'train-00000-of-00001.parquet')
    write_parquet(dev,   out, DATASET, 'test-00000-of-00001.parquet')
    for repo, df in blind.items():
        write_parquet(df, out, repo, 'test-00000-of-00001.parquet')

    # --- catalog: referenced tracks first, then random distractors -----------
    referenced = set()
    for df in [train, dev, *blind.values()]:
        referenced |= played_track_ids(df)

    catalog = read_parquet(src, TRACK_MD, 'all_tracks-00000-of-00001.parquet')
    if catalog is None:
        sys.exit(f'missing {TRACK_MD} - run: python scripts/download_data.py --only track-metadata')
    all_ids = catalog['track_id'].tolist()
    in_catalog = set(all_ids)
    keep = referenced & in_catalog
    orphans = referenced - in_catalog          # referenced but absent from the catalog
    padding = [t for t in all_ids if t not in keep]
    n_pad = max(0, a.n_tracks - len(keep))
    if n_pad:
        keep |= set(rng.choice(padding, size=min(n_pad, len(padding)), replace=False).tolist())
    catalog_mini = catalog[catalog['track_id'].isin(keep)].reset_index(drop=True)
    keep = set(catalog_mini['track_id'])       # authoritative, in catalog order
    write_parquet(catalog_mini, out, TRACK_MD, 'all_tracks-00000-of-00001.parquet')

    test_md = read_parquet(src, TRACK_MD, 'test_tracks-00000-of-00001.parquet')
    if test_md is not None:
        write_parquet(test_md[test_md['track_id'].isin(keep)].reset_index(drop=True),
                      out, TRACK_MD, 'test_tracks-00000-of-00001.parquet')

    print(f'\nCatalog: {len(catalog_mini):,} / {len(catalog):,} tracks '
          f'({len(referenced & in_catalog):,} referenced by the kept sessions, '
          f'{len(catalog_mini) - len(referenced & in_catalog):,} random distractors)')
    if orphans:
        print(f'  note: {len(orphans)} referenced ids are absent from the full catalog too '
              f'(same gap as on the real data)')

    # --- embeddings: same track set, collapsed into a single shard -----------
    # The descriptor builders glob all_tracks-*.parquet, so shard count is free.
    shards = sorted(glob.glob(str(data_dir(src, TRACK_EMB) / 'all_tracks-*.parquet')))
    if shards:
        parts = []
        for f in shards:                        # one shard at a time: 756 MB in total
            e = pd.read_parquet(f)
            parts.append(e[e['track_id'].isin(keep)])
            print(f'  embeddings {Path(f).name}: +{len(parts[-1]):,} rows')
        emb = pd.concat(parts, ignore_index=True)
        write_parquet(emb, out, TRACK_EMB, 'all_tracks-00000-of-00001.parquet')
        print(f'Embeddings: {len(emb):,} rows '
              f'({100 * len(emb) / max(1, len(catalog_mini)):.1f}% of the mini catalog)')
        test_emb = read_parquet(src, TRACK_EMB, 'test_tracks-00000-of-00001.parquet')
        if test_emb is not None:
            write_parquet(test_emb[test_emb['track_id'].isin(keep)].reset_index(drop=True),
                          out, TRACK_EMB, 'test_tracks-00000-of-00001.parquet')
    else:
        print('Embeddings: none found - the descriptor stages will not run '
              '(python scripts/download_data.py --only track-embeddings)')

    # --- users: filtered when they carry a user_id, copied otherwise ---------
    uids = set()
    for df in [train, dev, *blind.values()]:
        if df is not None and 'user_id' in df.columns:
            uids |= set(df['user_id'])
    for repo in (USER_MD, USER_EMB):
        d = data_dir(src, repo)
        if not d.exists():
            continue
        for f in sorted(d.glob('*.parquet')):
            df = pd.read_parquet(f)
            if 'user_id' in df.columns and uids:
                df = df[df['user_id'].isin(uids)].reset_index(drop=True)
            write_parquet(df, out, repo, f.name)

    # --- provenance + sanity check ------------------------------------------
    hit, total = gt_coverage(dev, keep)
    (out / 'MINI_INFO.json').write_text(json.dumps({
        'source': str(src), 'seed': a.seed,
        'n_train': len(train), 'n_dev': len(dev),
        'n_blind': {r: len(d) for r, d in blind.items()},
        'n_tracks': len(catalog_mini), 'referenced_tracks': len(referenced & in_catalog),
        'dev_music_turns': total, 'dev_turns_reachable': hit,
    }, indent=2) + '\n')

    print(f'\nDev music turns whose track is in the mini catalog: {hit}/{total} '
          f'({100 * hit / max(1, total):.1f}%)')
    if hit != total:
        print('  WARNING: not 100% - retrieval recall cannot reach 1.0 on this sample')
    print(f'Written to {out}/ ({dir_size_mb(out):.1f} MB)')
    print(f'Next: bash scripts/smoke_test.sh   (runs the paper pipeline against {out}/)')


if __name__ == '__main__':
    main()
