"""Download the official RecSys Challenge 2026 datasets into data/.

All data comes from the `talkpl-ai` organisation on Hugging Face. We use no
external resource beyond these. Total footprint: ~875 MB.

    python scripts/download_data.py                  # everything (default)
    python scripts/download_data.py --only inference # just what Blind-B inference needs
    python scripts/download_data.py --only blind-b track-metadata
    python scripts/download_data.py --check          # verify an existing download
    python scripts/download_data.py --evaluator      # also clone the official evaluator

Re-running is safe: snapshot_download resumes and skips what is already there.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

# short name -> (HF repo, approx size, expected file under <dir>/data/, why we need it)
DATASETS = {
    'dataset': (
        'TalkPlayData-Challenge-Dataset', '91 MB',
        ['train-00000-of-00001.parquet', 'test-00000-of-00001.parquet'],
        'train conversations + dev split (our held-out test set)'),
    'blind-a': (
        'TalkPlayData-Challenge-Blind-A', '476 KB',
        ['test-00000-of-00001.parquet'],
        'first blind phase'),
    'blind-b': (
        'TalkPlayData-Challenge-Blind-B', '132 KB',
        ['test-00000-of-00001.parquet'],
        'final blind phase (the submitted run)'),
    'track-metadata': (
        'TalkPlayData-Challenge-Track-Metadata', '20 MB',
        ['all_tracks-00000-of-00001.parquet'],
        'the 47k-track catalog'),
    'track-embeddings': (
        'TalkPlayData-Challenge-Track-Embeddings', '756 MB',
        [f'all_tracks-0000{i}-of-00004.parquet' for i in range(4)],
        'audio / lyrics embeddings, distilled into the text descriptors'),
    'user-metadata': (
        'TalkPlayData-Challenge-User-Metadata', '384 KB',
        [], 'user profiles'),
    'user-embeddings': (
        'TalkPlayData-Challenge-User-Embeddings', '7.5 MB',
        [], 'BPR user factors (evaluated, then dropped - see the ablations)'),
}

# minimum needed to run scripts/reproduce_blindB_submission.sh
INFERENCE = ['dataset', 'blind-b', 'track-metadata', 'track-embeddings']

ORG = 'talkpl-ai'
EVALUATOR = 'https://github.com/nlp4musa/music-crs-evaluator'


def expected_files(name):
    repo, _, files, _ = DATASETS[name]
    return [DATA / repo / 'data' / f for f in files]


def check(names):
    missing = []
    for n in names:
        for p in expected_files(n):
            if not p.exists():
                missing.append(p.relative_to(ROOT))
    if missing:
        print(f'MISSING {len(missing)} file(s):')
        for p in missing:
            print(f'  {p}')
        return False
    print(f'OK - all expected files present for: {", ".join(names)}')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='NAME',
                    help="subset to download: 'inference' or any of: "
                         + ', '.join(DATASETS))
    ap.add_argument('--check', action='store_true',
                    help='only verify what is already on disk')
    ap.add_argument('--evaluator', action='store_true',
                    help='also clone the official music-crs-evaluator at the repo root')
    args = ap.parse_args()

    if args.only:
        names = INFERENCE if args.only == ['inference'] else args.only
        unknown = [n for n in names if n not in DATASETS]
        if unknown:
            sys.exit(f'unknown dataset(s): {unknown}\navailable: {list(DATASETS)}')
    else:
        names = list(DATASETS)

    if args.check:
        sys.exit(0 if check(names) else 1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit('huggingface_hub is missing - run: bash setup_env.sh')

    DATA.mkdir(exist_ok=True)
    print(f'Downloading {len(names)} dataset(s) into {DATA.relative_to(ROOT)}/\n')
    for i, n in enumerate(names, 1):
        repo, size, _, why = DATASETS[n]
        print(f'[{i}/{len(names)}] {repo} ({size}) - {why}')
        snapshot_download(f'{ORG}/{repo}', repo_type='dataset',
                          local_dir=str(DATA / repo))

    print()
    ok = check(names)

    if args.evaluator:
        dest = ROOT / 'music-crs-evaluator'
        if dest.exists():
            print(f'\nevaluator already present: {dest.relative_to(ROOT)}/')
        else:
            print(f'\nCloning {EVALUATOR}')
            subprocess.run(['git', 'clone', EVALUATOR, str(dest)], check=True)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
