"""Dense retriever with the truncation fix: Qwen3-Embedding-4B, seq 1024, truncation_side='left'.

Produces `models/qwen3_ft_dualencoder_4b_ctx1024`. LoRA recipe (attention, n_hard=0,
MultipleNegativesRankingLoss gather); only long-query handling changes:

  Reason: 76% of dev queries exceed 256 tokens (median 671), and the final user
  request appears LAST. Default right truncation discarded the final intent for
  all turns >=3. Here:
    - truncation_side='left' -> keep recent turns and the final request
    - max_seq_len=1024       -> retain about 85% of queries in full (otherwise left-truncate)

Run (3 GPUs):
    accelerate launch --multi_gpu --num_processes 3 --gpu_ids 0,1,2 src/retrieval/train_dense_encoder.py

Note: the associated encoding/pool script MUST also set max_seq_length=1024 and
tokenizer.truncation_side='left' to avoid a train/inference mismatch.

Text format (--text_format):
  published  the rendering that produced the released weights. History tracks are
             written `assistant: track_id: ..., track_name: ...` and the current
             request `user: ...`, and only `list` metadata values are unwrapped.
             This does NOT match what build_pool_dev.py / build_pool_blind.py feed
             the model at inference time; it is kept as the default so the paper's
             retriever stays exactly reproducible.
  aligned    src/common/conv_render.py, i.e. byte-identical to what the pool
             builders emit (`assistant_played: Name - Artist`, `user (REQUEST):`),
             with `numpy.ndarray` metadata unwrapped as well. Use this when
             training a retriever to actually deploy.

tests/test_retriever_text_parity.py asserts the `aligned` equality and documents
the `published` divergence.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from src.common import conv_render

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import bm25s
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
    BatchSamplers,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from peft import LoraConfig, TaskType
from accelerate import PartialState

DATA = Path('data')
TASK = 'Given a music chat conversation, retrieve the track the user wants to listen to next'


def query_prompt(text: str) -> str:
    return f'Instruct: {TASK}\nQuery: {text}'


def track_text(row) -> str:
    parts = [
        f"track_name: {row['track_name']}",
        f"artist_name: {row['artist_name']}",
        f"album_name: {row['album_name']}",
    ]
    if isinstance(row['tag_list'], (list, np.ndarray)) and len(row['tag_list']) > 0:
        parts.append(f"tags: {', '.join(row['tag_list'])}")
    return '\n'.join(parts)


def track_id_to_metadata(tid, lookup) -> str:
    if tid not in lookup.index:
        return tid
    row = lookup.loc[tid]
    return (
        f"track_id: {tid}, track_name: {row['track_name']}, "
        f"artist_name: {row['artist_name']}, album_name: {row['album_name']}"
    )


def build_full_query(conversations, target_turn, lookup, text_format='published') -> str:
    """Render one training query. See the module docstring for the two formats."""
    if text_format == 'aligned':
        return conv_render.render_conversation(conversations, target_turn, lookup)
    lines = []
    for turn in conversations:
        if turn['turn_number'] >= target_turn:
            break
        role, content = turn['role'], turn['content']
        if role == 'music':
            role, content = 'assistant', track_id_to_metadata(content, lookup)
        lines.append(f'{role}: {content}')
    for turn in conversations:
        if turn['turn_number'] == target_turn and turn['role'] == 'user':
            lines.append(f"user: {turn['content']}")
            break
    return '\n'.join(lines)


def build_bm25_corpus(df):
    docs = []
    for _, row in df.iterrows():
        parts = [
            f"track_name: {row['track_name']}",
            f"artist_name: {row['artist_name']}",
            f"album_name: {row['album_name']}",
        ]
        if isinstance(row['tag_list'], (list, np.ndarray)):
            parts.append(f"tags: {', '.join(row['tag_list'])}")
        docs.append('\n'.join(parts))
    return docs


def build_pairs_and_mine(train_df, track_meta, track_lookup, n_hard, log, text_format='published'):
    log('Building (anchor, positive) pairs...')
    raw_queries, prompted_anchors, positive_tids = [], [], []
    for _, session in train_df.iterrows():
        convs = session['conversations']
        gt_by_turn = {t['turn_number']: t['content'] for t in convs if t['role'] == 'music'}
        for turn_number in range(1, 9):
            gt = gt_by_turn.get(turn_number)
            if gt is None or gt not in track_lookup.index:
                continue
            q = build_full_query(convs, turn_number, track_lookup, text_format).lower()
            raw_queries.append(q)
            prompted_anchors.append(query_prompt(q))
            positive_tids.append(gt)
    Q = len(raw_queries)
    log(f'{Q:,} pairs built.')

    if n_hard == 0:
        log('n_hard=0 -> no mining (v1 recipe).')
        return prompted_anchors, positive_tids, []

    log('Building BM25 index over catalog...')
    track_ids_list = track_meta['track_id'].tolist()
    corpus = build_bm25_corpus(track_meta)
    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    log('Tokenizing train queries + retrieving top-50...')
    q_tokens = bm25s.tokenize(raw_queries, show_progress=False)
    res = retriever.retrieve(q_tokens, k=50, return_as='tuple')

    log(f'Mining {n_hard} hard negatives per query...')
    rng = np.random.default_rng(42)
    N = len(track_ids_list)
    hard_negs = []
    for i in range(Q):
        pos_tid = positive_tids[i]
        candidates = [track_ids_list[int(idx)] for idx in res.documents[i]]
        negs = [c for c in candidates if c != pos_tid][:n_hard]
        if len(negs) < n_hard:
            needed = n_hard - len(negs)
            existing = set(negs) | {pos_tid}
            while needed > 0:
                rand_idx = int(rng.integers(0, N))
                cand = track_ids_list[rand_idx]
                if cand not in existing:
                    negs.append(cand)
                    existing.add(cand)
                    needed -= 1
        hard_negs.append(negs)
    log('Hard negative mining done.')
    return prompted_anchors, positive_tids, hard_negs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',        default='Qwen/Qwen3-Embedding-4B')
    parser.add_argument('--output',      default='models/qwen3_ft_dualencoder_4b_ctx1024')
    parser.add_argument('--batch_size',  type=int,   default=4)    # Sequence 1024 requires a smaller batch.
    parser.add_argument('--epochs',      type=int,   default=1)
    parser.add_argument('--max_seq_len', type=int,   default=1024) # truncation fix
    parser.add_argument('--n_hard',      type=int,   default=0)   # v1 recipe
    parser.add_argument('--text_format', choices=['published', 'aligned'], default='published',
                        help="'published' reproduces the released weights; 'aligned' matches "
                             "what the pool builders feed the model at inference time")
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--lora_r',      type=int,   default=16)
    parser.add_argument('--truncation_side', default='left')       # Keep the final request and recent turns.
    args = parser.parse_args()

    state = PartialState()
    is_main = state.is_main_process

    def log(msg):
        if is_main:
            print(f'[main] {msg}', flush=True)

    log(f'num_processes={state.num_processes}, device={state.device}')
    log(f'args: {vars(args)}')

    log('Loading data...')
    train_df = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet')
    track_meta = pd.read_parquet(DATA / 'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
    if args.text_format == 'aligned':
        conv_render.normalize_track_metadata(track_meta)
    else:
        # Released-weights behaviour: only `list` values are unwrapped, so an
        # ndarray column keeps its brackets ("['With Rainy Eyes']"). Preserved
        # deliberately - see --text_format.
        for col in ['track_name', 'artist_name', 'album_name']:
            track_meta[col] = track_meta[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    track_lookup = track_meta.set_index('track_id')
    log(f'text_format = {args.text_format}')

    prompted_anchors, positive_tids, hard_negs = build_pairs_and_mine(
        train_df, track_meta, track_lookup, args.n_hard, log, args.text_format,
    )

    Q = len(prompted_anchors)
    log('Building HF Dataset...')
    data = {
        'anchor':   prompted_anchors,
        'positive': [track_text(track_lookup.loc[t]) for t in positive_tids],
    }
    for k in range(args.n_hard):
        data[f'negative_{k+1}'] = [track_text(track_lookup.loc[hard_negs[i][k]]) for i in range(Q)]
    train_ds = Dataset.from_dict(data)
    log(f'Dataset built: {train_ds.column_names}, size={len(train_ds):,}')

    log(f'Loading {args.base} + LoRA (attention, r={args.lora_r})...')
    model = SentenceTransformer(
        args.base,
        model_kwargs={'torch_dtype': torch.bfloat16},
    )
    model.max_seq_length = args.max_seq_len
    # Truncation fix: the user request is LAST, and 76% of queries exceed 256 tokens.
    # Right truncation discarded the final request (the intent). Left truncation keeps
    # recent turns and the request. Positive documents are short, so this has no effect on them.
    model.tokenizer.truncation_side = args.truncation_side
    log(f'truncation_side = {model.tokenizer.truncation_side}, max_seq_len = {args.max_seq_len}')
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model.add_adapter(lora)

    try:
        loss = MultipleNegativesRankingLoss(model, gather_across_devices=True)
        log('Loss: gather_across_devices=True enabled')
    except TypeError:
        loss = MultipleNegativesRankingLoss(model)
        log('Loss: gather_across_devices not available, fallback')

    targs = SentenceTransformerTrainingArguments(
        output_dir='models/_ft_checkpoints_4b',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        ddp_find_unused_parameters=False,
        logging_steps=50,
        save_strategy='no',
        report_to=[],
        dataloader_num_workers=4,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_ds, loss=loss,
    )
    log('Training...')
    trainer.train()

    if is_main:
        import shutil
        import gc
        from peft import PeftModel

        tmp_dir = Path(str(args.output) + '.tmp_adapter')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)

        log(f'[1/4] Saving adapter to temp: {tmp_dir}')
        model.save_pretrained(str(tmp_dir))

        log('[2/4] Freeing training model from VRAM...')
        del trainer, loss, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log('[3/4] Reloading base + applying adapter + merging...')
        base = SentenceTransformer(
            args.base,
            model_kwargs={'torch_dtype': torch.bfloat16},
        )
        base[0].auto_model = PeftModel.from_pretrained(
            base[0].auto_model, str(tmp_dir)
        ).merge_and_unload()
        log('Merge done.')

        final_dir = Path(args.output)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        log(f'[4/4] Saving merged model to {final_dir}...')
        base.save_pretrained(str(final_dir))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log('Done.')
    state.wait_for_everyone()


if __name__ == '__main__':
    main()
