#!/usr/bin/env python3
"""Build activity timelines from saved classifier predictions.

Each test user in a cross-validation fold gets one timeline JSON in timelines/,
built from the Random Forest's out-of-fold predictions (rf_results/) and the
signal in balanced_folds/, so the query engine can answer questions about them.

  python3 build_timelines.py 9DC38D04 4FC32141      # specific users (UUID prefixes)
  python3 build_timelines.py --all                  # every test user, all 5 folds
  python3 build_timelines.py --fold 2               # every test user of one fold
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import activity_timeline as T

OUT_DIR = "timelines"


def fold_users(fold: int) -> list[str]:
    return json.load(open(f"balanced_folds/fold_{fold}/report.json"))["users"]["test"]


def locate(prefix: str) -> tuple[int, str]:
    for f in range(5):
        for u in fold_users(f):
            if u.startswith(prefix):
                return f, u
    raise SystemExit(f"{prefix!r} is not a test user in any fold "
                     f"(only test users have out-of-fold predictions)")


def build_fold(fold: int, only: set[str] | None = None, cfg=None) -> list[str]:
    cfg = cfg or T.TimelineConfig()
    users = fold_users(fold)
    u = np.load(f"balanced_folds/fold_{fold}/u_test.npy")
    w = np.load(f"balanced_folds/fold_{fold}/w_test.npy")
    X = np.load(f"balanced_folds/fold_{fold}/X_test.npy", mmap_mode="r")
    pred = np.load("rf_results/predictions.npz", allow_pickle=True)
    yp, yt = pred[f"y_pred_{fold}"], pred[f"y_true_{fold}"]
    os.makedirs(OUT_DIR, exist_ok=True)
    done = []
    for j, uid in enumerate(users):
        if only and uid not in only:
            continue
        # segment position inside its window, in the order balanced_folds used
        ss = pd.read_parquet(f"segmented_4s/{uid}.parquet",
                             columns=["seg_start"])["seg_start"].to_numpy()[::128]
        seg_idx = ss[ss % 128 == 0] // 128
        m = np.flatnonzero(u == j)
        if len(seg_idx) != len(m):
            raise SystemExit(f"{uid}: {len(seg_idx)} segments vs {len(m)} predictions - "
                             "balanced_folds and segmented_4s are out of sync")
        df = T.build_segment_table(w[m], seg_idx, yp[m], yt[m])
        tl = T.build_timeline(df, cfg, X=X[m], user=uid)
        tl["source"] = {"fold": fold, "classifier": "random_forest",
                        "predictions": "rf_results/predictions.npz"}
        T.save_timeline(tl, os.path.join(OUT_DIR, f"{uid}.json"))
        done.append(uid)
        print(f"  fold {fold}  {uid[:8]}  {len(tl['intervals']):5d} intervals  "
              f"{tl['recording_span_s']:>10,.0f} s")
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("users", nargs="*", help="UUIDs or unique prefixes")
    p.add_argument("--all", action="store_true", help="every test user of every fold")
    p.add_argument("--fold", type=int, help="every test user of one fold")
    a = p.parse_args()
    if a.all:
        for f in range(5):
            build_fold(f)
    elif a.fold is not None:
        build_fold(a.fold)
    elif a.users:
        by_fold: dict[int, set[str]] = {}
        for pre in a.users:
            f, uid = locate(pre)
            by_fold.setdefault(f, set()).add(uid)
        for f, s in sorted(by_fold.items()):
            build_fold(f, only=s)
    else:
        p.error("give user prefixes, --fold N, or --all")


if __name__ == "__main__":
    main()
