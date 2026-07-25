"""The retriever must be trained on the text it is served.

`--text_format aligned` has to produce byte-identical output to what
build_pool_dev.py and build_pool_blind.py feed the encoder. These tests pin that
equality, and pin the fact that the default `published` format does NOT match -
so the divergence behind the released weights stays visible instead of being
rediscovered later.

No data, weights or GPU needed:

    python tests/test_retriever_text_parity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import conv_render  # noqa: E402


CATALOG = pd.DataFrame({
    'track_id': ['t1', 't2'],
    # deliberately mixed: one plain string, one ndarray-wrapped value, which is
    # what Parquet hands back for these columns
    'track_name': ['Ramble On', np.array(['With Rainy Eyes'], dtype=object)],
    'artist_name': ['Led Zeppelin', np.array(['Toe'], dtype=object)],
    'album_name': ['Led Zeppelin II', np.array(['For Long Tomorrow'], dtype=object)],
    'tag_list': [np.array(['rock', '70s'], dtype=object), np.array([], dtype=object)],
})

CONVERSATION = [
    {'turn_number': 1, 'role': 'user', 'content': 'something bluesy'},
    {'turn_number': 1, 'role': 'music', 'content': 't1'},
    {'turn_number': 1, 'role': 'assistant', 'content': 'Try this one.'},
    {'turn_number': 2, 'role': 'user', 'content': 'now something by other artists'},
]


def lookup(normalize):
    """Catalog indexed by track_id, with or without the ndarray unwrap."""
    cat = CATALOG.copy()
    if normalize:
        conv_render.normalize_track_metadata(cat)
    else:
        for col in ['track_name', 'artist_name', 'album_name']:
            cat[col] = cat[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    return cat.set_index('track_id')


# --- reference implementations, copied verbatim from the two pool builders ----

def pool_dev_conv_text(convs, target_turn, track_lookup):
    """build_pool_dev.py :: conv_text"""
    def track_id_to_short(tid):
        if tid not in track_lookup.index:
            return tid
        r = track_lookup.loc[tid]
        return f"{r['track_name']} - {r['artist_name']}"
    lines = []
    for t in convs:
        if t['turn_number'] >= target_turn:
            break
        role, content = t['role'], t['content']
        if role == 'music':
            role, content = 'assistant_played', track_id_to_short(content)
        lines.append(f'{role}: {content}')
    for t in convs:
        if t['turn_number'] == target_turn and t['role'] == 'user':
            lines.append(f"user (REQUEST): {t['content']}")
            break
    return '\n'.join(lines)


def pool_blind_conv(cs, tt, lk):
    """build_pool_blind.py :: conv"""
    def short(t):
        if t not in lk.index:
            return t
        r = lk.loc[t]
        return f"{r['track_name']} - {r['artist_name']}"
    L = []
    for t in cs:
        if t['turn_number'] >= tt:
            break
        ro, co = t['role'], t['content']
        if ro == 'music':
            ro, co = 'assistant_played', short(co)
        L.append(f'{ro}: {co}')
    for t in cs:
        if t['turn_number'] == tt and t['role'] == 'user':
            L.append(f"user (REQUEST): {t['content']}")
            break
    return '\n'.join(L)


# --- tests -------------------------------------------------------------------

def test_aligned_matches_both_pool_builders():
    lk = lookup(normalize=True)
    shared = conv_render.render_conversation(CONVERSATION, 2, lk)
    assert shared == pool_dev_conv_text(CONVERSATION, 2, lk), 'aligned != build_pool_dev'
    assert shared == pool_blind_conv(CONVERSATION, 2, lk), 'aligned != build_pool_blind'
    assert 'assistant_played: Ramble On - Led Zeppelin' in shared
    assert 'user (REQUEST): now something by other artists' in shared


def test_published_format_diverges():
    """Documents the gap the released retriever was trained with."""
    sys.path.insert(0, str(ROOT / 'src' / 'retrieval'))
    from train_dense_encoder import build_full_query

    lk = lookup(normalize=True)
    published = build_full_query(CONVERSATION, 2, lk, 'published')
    aligned = build_full_query(CONVERSATION, 2, lk, 'aligned')
    assert aligned == conv_render.render_conversation(CONVERSATION, 2, lk)
    assert published != aligned, 'the published format is supposed to differ'
    assert 'assistant: track_id: t1' in published
    assert published.endswith('user: now something by other artists')


def test_ndarray_metadata_is_unwrapped_when_aligned():
    """The published path leaves brackets in the text; aligned does not."""
    assert "['With Rainy Eyes']" in str(lookup(normalize=False).loc['t2', 'track_name'])
    assert lookup(normalize=True).loc['t2', 'track_name'] == 'With Rainy Eyes'


def test_track_text_matches_pool_builders():
    lk = lookup(normalize=True)
    text = conv_render.track_text(lk.loc['t1'])
    assert text == ('track_name: Ramble On\nartist_name: Led Zeppelin\n'
                    'album_name: Led Zeppelin II\ntags: rock, 70s')
    # empty tag_list -> no tags line, same as both builders
    assert 'tags:' not in conv_render.track_text(lk.loc['t2'])


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_'):
            continue
        try:
            fn()
            print(f'PASS {name}')
        except AssertionError as e:
            failures += 1
            print(f'FAIL {name}: {e}')
    sys.exit(1 if failures else 0)
