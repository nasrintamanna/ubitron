#!/usr/bin/env python3
"""Timelines from the IMPROVED model, written to Try_increase_accuracy/timelines/.

Same procedure as ../build_timelines.py, but reading final_model/predictions.npz
instead of ../rf_results/predictions.npz. The original timelines are untouched.

  python3 build_timelines_improved.py 9DC38D04          # from inside this folder
  python3 build_timelines_improved.py --all

Then ask questions from the project root with:
  python3 ask.py --timeline Try_increase_accuracy/timelines/<uuid>.json "..."
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from common import HERE, ROOT

sys.path.insert(0, str(ROOT))
import activity_timeline as T           # noqa: E402  (read-only import)

OUT = HERE / "timelines"
PRED = HERE / "final_model" / "predictions.npz"


def users_of(fold):
    return json.load(open(ROOT / f"balanced_folds/fold_{fold}/report.json"))["users"]["test"]


def build(fold, only=None):
    users = users_of(fold)
    u = np.load(ROOT / f"balanced_folds/fold_{fold}/u_test.npy")
    w = np.load(ROOT / f"balanced_folds/fold_{fold}/w_test.npy")
    X = np.load(ROOT / f"balanced_folds/fold_{fold}/X_test.npy", mmap_mode="r")
    pr = np.load(PRED)
    yp, yt = pr[f"y_pred_{fold}"], pr[f"y_true_{fold}"]
    OUT.mkdir(exist_ok=True)
    for j, uid in enumerate(users):
        if only and uid not in only:
            continue
        ss = pd.read_parquet(ROOT / f"segmented_4s/{uid}.parquet",
                             columns=["seg_start"])["seg_start"].to_numpy()[::128]
        seg_idx = ss[ss % 128 == 0] // 128
        m = np.flatnonzero(u == j)
        assert len(seg_idx) == len(m), f"{uid}: segment/prediction count mismatch"
        tl = T.build_timeline(T.build_segment_table(w[m], seg_idx, yp[m], yt[m]),
                              T.TimelineConfig(), X=X[m], user=uid)
        tl["source"] = {"fold": fold, "classifier": "random_forest + time of day + tuned thresholds",
                        "predictions": "Try_increase_accuracy/final_model/predictions.npz"}
        T.save_timeline(tl, str(OUT / f"{uid}.json"))
        print(f"  fold {fold}  {uid[:8]}  {len(tl['intervals']):5d} intervals")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("users", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for f in range(5):
            build(f)
    else:
        for pre in a.users:
            f = next((f for f in range(5) if any(x.startswith(pre) for x in users_of(f))), None)
            if f is None:
                sys.exit(f"{pre!r} is not a test user in any fold")
            build(f, {x for x in users_of(f) if x.startswith(pre)})
