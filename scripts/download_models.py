"""Download the fine-tuned weights and derived artifacts from Hugging Face.

Companion to scripts/download_data.py. Everything lands where the pipeline
expects it: models/ for weights, cache/ and exp/ for the derived artifacts.

    python scripts/download_models.py                   # everything (~41 GB)
    python scripts/download_models.py --only inference  # only what Blind-B needs
    python scripts/download_models.py --check           # verify an existing download
    python scripts/download_models.py --check --checksums   # ... and hash every file

Every transfer is pinned to the revision recorded in scripts/artifact_manifest.json
and verified against the SHA-256 values in the same file, so you get
bit-identical inputs to the published results. `--no_pin` downloads the branch
head instead, and `--write_manifest` regenerates the file after publishing new
weights.

`--only inference` means "everything the Blind-B path reads and nothing else":
the retriever, the submitted scoring head, the descriptor cache, and the Blind-B
pool. It deliberately excludes the dev pool, which only the paper's dev tables
need. Use `--no_artifacts` to skip the derived artifacts entirely and rebuild
them locally instead.

Re-running is safe: transfers resume and complete files are skipped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / 'scripts' / 'artifact_manifest.json'

# short name -> (local destination, approx size, expected files, role)
COMPONENTS = {
    'retriever': (
        'models/qwen3_ft_dualencoder_4b_ctx1024', '7.6 GB',
        ['config.json', 'model.safetensors', 'modules.json'],
        'dense dual-encoder, the dense channel of the RRF pool'),
    'scorehead': (
        'models/qwen3_8b_scorehead_top200', '15 GB',
        # score_head.pt is NOT optional: without it the checkpoint is just a
        # backbone with no scoring head and the reranker cannot run.
        ['config.json', 'model.safetensors', 'score_head.pt'],
        'listwise scoring-head reranker, the submitted Blind-B model'),
    # comparison arms of the paper's reranking table
    'scorehead-llama-top200': (
        'models/llama32_3b_scorehead_top200', '6.1 GB',
        ['config.json', 'model.safetensors', 'score_head.pt'],
        'scoring head on Llama-3.2-3B, top-200 pool'),
    'scorehead-llama-top50': (
        'models/llama32_3b_scorehead_ctx1024', '6.1 GB',
        ['config.json', 'model.safetensors', 'score_head.pt'],
        'scoring head on Llama-3.2-3B, top-50 pool'),
    'firstpos': (
        'models/llama32_3b_intent_mm_ctx1024', '6.1 GB',
        ['config.json', 'model.safetensors', 'generation_config.json'],
        'generative single-pick reranker (intent+mm)'),
}

# Derived artifacts live in one dataset repo but land in several places.
# `inference` marks the ones the Blind-B path actually reads; the dev pool is
# only needed to reproduce the paper's dev tables.
ARTIFACTS = [
    # (remote path, local destination, needed for inference)
    ('content_desc_full.json', 'cache/content_desc_full.json', True),
    ('pools/combined_pool_ctx1024_blindB.parquet', 'exp/combined_pool_ctx1024_blindB.parquet', True),
    ('pools/combined_pool_ctx1024_dev.parquet', 'exp/combined_pool_ctx1024_dev.parquet', False),
]

INFERENCE = ['retriever', 'scorehead']


def load_manifest():
    if not MANIFEST_PATH.exists():
        sys.exit(f'missing {MANIFEST_PATH.relative_to(ROOT)} - '
                 'regenerate it with --write_manifest')
    return json.loads(MANIFEST_PATH.read_text())['components']


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def artifacts_for(inference_only):
    return [a for a in ARTIFACTS if a[2] or not inference_only]


def check(names, wanted_artifacts, manifest, checksums):
    """Presence + recorded size for everything; SHA-256 as well when asked."""
    problems = []

    def verify(local, rel_to_manifest, entry):
        if not local.exists():
            problems.append(f'missing      {local.relative_to(ROOT)}')
            return
        rec = (entry or {}).get('files', {}).get(rel_to_manifest)
        if rec is None:
            return  # not a checksummed file (small config/tokenizer text files)
        if local.stat().st_size != rec['size']:
            problems.append(f'wrong size   {local.relative_to(ROOT)} '
                            f'({local.stat().st_size} != {rec["size"]})')
            return
        if checksums and sha256(local) != rec['sha256']:
            problems.append(f'bad checksum {local.relative_to(ROOT)}')

    for n in names:
        dest, _, files, _ = COMPONENTS[n]
        entry = manifest.get(n)
        for f in files:
            verify(ROOT / dest / f, f, entry)

    if wanted_artifacts:
        entry = manifest.get('artifacts')
        for remote, dest, _ in wanted_artifacts:
            verify(ROOT / dest, remote, entry)

    if problems:
        print(f'FAILED - {len(problems)} problem(s):')
        for p in problems:
            print(f'  {p}')
        return False
    scope = ', '.join(names) + (' + artifacts' if wanted_artifacts else '')
    print(f'OK - {scope}' + (' (checksums verified)' if checksums else
                             ' (sizes verified; add --checksums to hash)'))
    return True


def write_manifest():
    """Re-record revisions and checksums from the hub after publishing weights."""
    from huggingface_hub import HfApi
    api = HfApi()
    repos = {n: (COMPONENTS[n], None) for n in COMPONENTS}
    out = {}
    known = load_manifest() if MANIFEST_PATH.exists() else {}
    for name in list(COMPONENTS) + ['artifacts']:
        repo = known.get(name, {}).get('repo')
        kind = known.get(name, {}).get('repo_type', 'model')
        if not repo:
            sys.exit(f'no repo recorded for {name}; add it to the manifest first')
        info = (api.model_info if kind == 'model' else api.dataset_info)(
            repo, files_metadata=True)
        files = {s.rfilename: {'sha256': s.lfs.sha256, 'size': s.size}
                 for s in info.siblings if s.lfs and getattr(s.lfs, 'sha256', None)}
        # keep any manually recorded non-LFS checksums
        for f, rec in known.get(name, {}).get('files', {}).items():
            files.setdefault(f, rec)
        out[name] = {'repo': repo, 'repo_type': kind, 'revision': info.sha,
                     'files': dict(sorted(files.items()))}
        print(f'{name:24s} {info.sha} ({len(files)} checksummed files)')
    body = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    body['components'] = out
    MANIFEST_PATH.write_text(json.dumps(body, indent=2) + '\n')
    print(f'\nwritten -> {MANIFEST_PATH.relative_to(ROOT)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='NAME',
                    help="subset: 'inference' (retriever + scoring head + the "
                         "Blind-B artifacts) or any of: " + ', '.join(COMPONENTS))
    ap.add_argument('--no_artifacts', action='store_true',
                    help='skip the descriptor cache and pools (you can rebuild them)')
    ap.add_argument('--check', action='store_true', help='verify what is on disk')
    ap.add_argument('--checksums', action='store_true',
                    help='with --check, also verify SHA-256 (slow: hashes every file)')
    ap.add_argument('--no_pin', action='store_true',
                    help='download the branch head instead of the pinned revision')
    ap.add_argument('--write_manifest', action='store_true',
                    help='re-record revisions and checksums from the hub')
    args = ap.parse_args()

    if args.write_manifest:
        write_manifest()
        return

    manifest = load_manifest()

    inference_only = args.only == ['inference']
    if args.only:
        names = INFERENCE if inference_only else args.only
        unknown = [n for n in names if n not in COMPONENTS]
        if unknown:
            sys.exit(f'unknown component(s): {unknown}\navailable: {list(COMPONENTS)}')
    else:
        names = list(COMPONENTS)
    wanted_artifacts = [] if args.no_artifacts else artifacts_for(inference_only)

    if args.check:
        sys.exit(0 if check(names, wanted_artifacts, manifest, args.checksums) else 1)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.errors import (GatedRepoError, RepositoryNotFoundError,
                                            RevisionNotFoundError)
    except ImportError:
        sys.exit('huggingface_hub is missing - run: bash setup_env.sh')

    try:
        for i, n in enumerate(names, 1):
            dest, size, _, role = COMPONENTS[n]
            entry = manifest[n]
            rev = None if args.no_pin else entry['revision']
            print(f'[{i}/{len(names)}] {entry["repo"]}@{rev or "HEAD"} ({size}) - {role}')
            snapshot_download(entry['repo'], repo_type=entry['repo_type'],
                              revision=rev, local_dir=str(ROOT / dest))

        if wanted_artifacts:
            entry = manifest['artifacts']
            rev = None if args.no_pin else entry['revision']
            print(f'[+] {entry["repo"]}@{rev or "HEAD"} - derived artifacts')
            for remote, dest, _ in wanted_artifacts:
                p = hf_hub_download(entry['repo'], remote, repo_type=entry['repo_type'],
                                    revision=rev)
                out = ROOT / dest
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(Path(p).read_bytes())
                print(f'    {dest}')
    except RevisionNotFoundError as e:
        sys.exit(f'\n{type(e).__name__}: {e}\n\n'
                 'The pinned revision no longer exists. Re-record the manifest with\n'
                 '  python scripts/download_models.py --write_manifest\n'
                 'or bypass the pin for a one-off run with --no_pin.')
    except (GatedRepoError, RepositoryNotFoundError) as e:
        sys.exit(f'\n{type(e).__name__}: {e}\n\n'
                 'Authenticate with\n'
                 '  huggingface-cli login\n'
                 'or rebuild the artifacts locally (see docs/02_reranking.md).')

    print()
    sys.exit(0 if check(names, wanted_artifacts, manifest, args.checksums) else 1)


if __name__ == '__main__':
    main()
