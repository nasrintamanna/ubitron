#!/usr/bin/env python3
"""Timelines for the 56 known users from the UPGRADED classifier (out-of-fold).

Written to upgraded_pipeline/timelines/; the original ../timelines/ is untouched.
  python3 build_timelines.py 00EABED2 9DC38D04     # or --all
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("users", nargs="*"); ap.add_argument("--all", action="store_true")
a = ap.parse_args()
prefixes = a.users
if a.all:
    prefixes = sorted({u for f in range(5) for u in json.load(open(
        predict.ROOT / f"balanced_folds/fold_{f}/report.json"))["users"]["test"]})
if not prefixes:
    ap.error("give user prefixes or --all")
for p in prefixes:
    tl = predict.known_user_timeline(p)
    predict.T.save_timeline(tl, str(HERE / "timelines" / f"{tl['user']}.json"))
    print(f"  {tl['user'][:8]}  {len(tl['intervals']):5d} intervals  (fold {tl['source']['fold']})")
