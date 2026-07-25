"""Canonical conversation rendering for the dense retriever.

Single source of truth for the text the retriever sees, shared by the pool
builders (`build_pool_dev.py`, `build_pool_blind.py`) and, via
`--text_format aligned`, by `train_dense_encoder.py`.

Why this module exists: the released retriever was trained on a rendering that
differs from the one the pool builders use at inference time - history tracks
rendered as `assistant: track_id: ..., track_name: ...` instead of
`assistant_played: Name - Artist`, and the current request labelled `user:`
instead of `user (REQUEST):`. The published numbers stand, because the pool
builders have always been self-consistent, but a model retrained from the
repository was not being trained on the format it would later be served.

The trainer therefore keeps `--text_format published` as its default (bit-exact
with the released weights) and offers `--text_format aligned`, which routes
through this module. `tests/test_retriever_text_parity.py` asserts that the
aligned rendering is byte-identical to what the pool builders emit.

Both sides lowercase the query at encode time, so this module returns the
untouched casing and leaves `.lower()` to the caller, exactly as before.
"""
import numpy as np

# Instruction prefix expected by Qwen3-Embedding, identical on both sides.
TASK = 'Given a music chat conversation, retrieve the track the user wants to listen to next'


def unwrap_metadata_value(value):
    """Catalog string columns arrive as one-element arrays; take the element.

    Parquet hands these back as a `list` in some readers and a `numpy.ndarray`
    in others, so both are unwrapped. Handling only `list` is what let
    `['With Rainy Eyes']` reach the training text.
    """
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        return value[0]
    return value


def normalize_track_metadata(track_meta, columns=('track_name', 'artist_name', 'album_name')):
    """Unwrap the catalog's one-element string columns, in place-safe fashion."""
    for col in columns:
        track_meta[col] = track_meta[col].apply(unwrap_metadata_value).astype(str)
    return track_meta


def track_id_to_short(track_id, lookup):
    """`Name - Artist`, the form the pool builders put in the conversation text."""
    if track_id not in lookup.index:
        return track_id
    row = lookup.loc[track_id]
    return f"{row['track_name']} - {row['artist_name']}"


def render_conversation(conversations, target_turn, lookup):
    """History turns before `target_turn`, then that turn's user request.

    Played tracks become `assistant_played: Name - Artist`; the current request
    is labelled `user (REQUEST):` so the retriever can tell it from history.
    """
    lines = []
    for turn in conversations:
        if turn['turn_number'] >= target_turn:
            break
        role, content = turn['role'], turn['content']
        if role == 'music':
            role, content = 'assistant_played', track_id_to_short(content, lookup)
        lines.append(f'{role}: {content}')
    for turn in conversations:
        if turn['turn_number'] == target_turn and turn['role'] == 'user':
            lines.append(f"user (REQUEST): {turn['content']}")
            break
    return '\n'.join(lines)


def track_text(row):
    """Document side of the dual encoder: the catalog fields plus tags."""
    parts = [
        f"track_name: {row['track_name']}",
        f"artist_name: {row['artist_name']}",
        f"album_name: {row['album_name']}",
    ]
    if isinstance(row['tag_list'], (list, np.ndarray)) and len(row['tag_list']) > 0:
        parts.append(f"tags: {', '.join(row['tag_list'])}")
    return '\n'.join(parts)


def query_prompt(text, task=TASK):
    return f'Instruct: {task}\nQuery: {text}'
