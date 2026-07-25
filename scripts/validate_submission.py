"""Validate that a Blind-B submission passes the official scorer (nlp4musa/music-crs-evaluator).
Replicate the checks that otherwise cause the evaluation to fail:
  - match (session_id, turn_number) against the 80 Parquet keys -> otherwise IndexError (.iloc[0])
  - unique predicted_track_ids                                  -> otherwise ValueError (_has_duplicates)
as well as the eligibility rules: >=20 IDs, valid catalog IDs, and required columns.

    python scripts/validate_submission.py exp/inference/blindset_B/<file>.json
"""
import sys, json
import pandas as pd
from pathlib import Path

SUB = sys.argv[1] if len(sys.argv) > 1 else 'exp/inference/blindset_B/firstpos_scorehead_qwen_blindB_convbestofN.json'
DATA = Path('data')
tm = pd.read_parquet(DATA/'TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
catalog = set(tm['track_id'].astype(str))
b = pd.read_parquet(DATA/'TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet')
# The target turn is the last turn_number in the conversation.
gt_keys = {(str(r['session_id']), max(int(t['turn_number']) for t in r['conversations'])) for _, r in b.iterrows()}

preds = json.load(open(SUB))
df = pd.DataFrame(preds)
req = {'session_id', 'user_id', 'turn_number', 'predicted_track_ids', 'predicted_response'}
sub_keys = {(str(x['session_id']), int(x['turn_number'])) for x in preds}

errs = []
print(f"== {SUB} ==")
print(f"entries: {len(preds)} | columns: {list(df.columns)}")
if not req.issubset(df.columns): errs.append(f"missing columns: {req - set(df.columns)}")
miss, extra = gt_keys - sub_keys, sub_keys - gt_keys
print(f"matching (session,turn): {len(sub_keys & gt_keys)}/{len(gt_keys)} | missing={len(miss)} extra={len(extra)}")
if miss:  errs.append(f"{len(miss)} missing GT keys -> IndexError")
if extra: errs.append(f"{len(extra)} extra keys")

dup = nlt20 = badid = 0
for _, r in df.iterrows():
    ids = list(r['predicted_track_ids'])
    if len(ids) > len(set(ids)): dup += 1
    if len(ids) < 20: nlt20 += 1
    if any(str(t) not in catalog for t in ids): badid += 1
print(f"duplicates (-> scorer ValueError): {dup} | <20 IDs: {nlt20} | outside catalog: {badid}")
if dup:   errs.append(f"{dup} entries contain duplicates -> ValueError")
if badid: errs.append(f"{badid} entries contain IDs outside the catalog")
if nlt20: errs.append(f"{nlt20} entries contain <20 IDs")

empty_resp = sum(1 for x in preds if not str(x.get('predicted_response', '')).strip())
print(f"empty predicted_response values: {empty_resp}/{len(preds)}")

print("VERDICT:", "PASS" if not errs else "FAIL: " + " ; ".join(errs))
sys.exit(0 if not errs else 1)
