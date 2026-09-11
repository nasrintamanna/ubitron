"""Train the ONE deployable upgraded classifier on all 56 users, and save it.

Cross-validation produced five fold models and kept only their predictions. A
pipeline that must answer questions about an unseen recording needs a single
saved model. This trains it exactly like the evaluated configuration:

  training set   segmented_4s, same rule as random_forest_model.ipynb: capped at
                 120,000 segments per class, allocated equally across users, picks
                 spread evenly over each user's session - now over all 56 users
  features       the notebook's 213 features + time of day (sin, cos of local hour)
  model          RandomForest with the notebook's exact hyper-parameters
  thresholds     per-class weights = geometric mean of the five weight vectors
                 tuned on validation in Try_increase_accuracy (they agreed closely
                 across folds, so their mean is a stable single setting)

  python3 train_final_model.py            # the upgraded model (with time of day)
  python3 train_final_model.py --no-time  # fallback for recordings without clock time
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import features as F                                   # verbatim notebook extractor
from timefeat import hour_features

SEG_DIR = ROOT / "segmented_4s"
TIA = ROOT / "Try_increase_accuracy"
SEG_LEN, N_CLASSES, CAP = 128, 7, 120_000
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=4, max_features="sqrt",
                 class_weight="balanced", n_jobs=24, random_state=0)
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def water_fill(avail, target):          # verbatim from the RF notebook
    avail = np.asarray(avail, dtype=np.int64)
    take = np.zeros_like(avail)
    remaining, active = int(min(target, avail.sum())), avail > 0
    while remaining > 0 and active.any():
        share = max(1, remaining // int(active.sum()))
        give = np.minimum(share, avail - take)
        give[~active] = 0
        if give.sum() == 0:
            break
        if give.sum() > remaining:
            for j in np.flatnonzero(give):
                give[j] = min(give[j], max(remaining, 0))
                remaining -= give[j]
            take += give
            break
        take += give
        remaining -= int(give.sum())
        active = (avail - take) > 0
    return take


def all_users():
    us = set()
    for f in range(5):
        us |= set(json.load(open(ROOT / f"balanced_folds/fold_{f}/report.json"))["users"]["test"])
    return sorted(us)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--leaf", type=int, default=4)
    a = ap.parse_args()
    use_time = not a.no_time
    users = all_users()
    t0 = time.time()
    labs = {u: pd.read_parquet(SEG_DIR / f"{u}.parquet", columns=["Activity_Labels"])[
        "Activity_Labels"].to_numpy()[::SEG_LEN] - 1 for u in users}
    avail = np.stack([np.bincount(labs[u], minlength=N_CLASSES) for u in users])
    quota = np.stack([water_fill(avail[:, k], CAP) for k in range(N_CLASSES)], axis=1)
    Fs, ys = [], []
    for j, u in enumerate(users):
        d = pd.read_parquet(SEG_DIR / f"{u}.parquet", columns=["window"] + CHANNELS)
        n = len(d) // SEG_LEN
        X = d[CHANNELS].to_numpy(np.float32).reshape(n, SEG_LEN, 6)
        w = d["window"].to_numpy()[::SEG_LEN]
        y = labs[u]
        pick = []
        for k in range(N_CLASSES):
            q = int(quota[j, k])
            if q <= 0:
                continue
            idx = np.flatnonzero(y == k)
            if len(idx) > q:
                idx = idx[np.linspace(0, len(idx) - 1, q).round().astype(int)]
            pick.append(idx)
        pick = np.sort(np.concatenate(pick))
        f = F.extract_chunked(X[pick])
        if use_time:
            f = np.hstack([f, hour_features(w[pick])])
        Fs.append(f); ys.append(y[pick])
        print(f"  {j + 1:2d}/56 {u[:8]}  {len(pick):7,} training segments", flush=True)
    Xtr, ytr = np.concatenate(Fs), np.concatenate(ys)
    print(f"training set {Xtr.shape} built in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    rf = RandomForestClassifier(**{**RF_PARAMS, "min_samples_leaf": a.leaf}).fit(Xtr, ytr)
    fit_s = time.time() - t0
    src = "time__thresholds.json" if use_time else "baseline__thresholds.json"
    W = np.exp(np.mean(np.log(json.load(open(TIA / "results" / src))["weights_per_fold"]), axis=0))
    name = f"rf_{'time' if use_time else 'notime'}_leaf{a.leaf}"
    path = HERE / "models" / f"{name}.joblib"
    joblib.dump({"model": rf, "weights": W, "use_time": use_time,
                 "n_features": Xtr.shape[1]}, path, compress=3)
    meta = {"name": name, "use_time": use_time, "n_features": int(Xtr.shape[1]),
            "n_train": int(len(ytr)), "train_per_class": np.bincount(ytr, minlength=7).tolist(),
            "rf_params": {**RF_PARAMS, "min_samples_leaf": a.leaf}, "class_weights": W.tolist(),
            "weights_from": f"Try_increase_accuracy/results/{src} (geometric mean of 5 folds)",
            "fit_seconds": fit_s, "file_mb": path.stat().st_size / 2**20,
            "n_nodes_total": int(sum(e.tree_.node_count for e in rf.estimators_))}
    json.dump(meta, open(HERE / "models" / f"{name}.json", "w"), indent=1)
    print(f"fit {fit_s:.0f}s | saved {path.name}: {meta['file_mb']:,.0f} MB, "
          f"{meta['n_nodes_total']:,} tree nodes | weights {np.round(W, 2).tolist()}", flush=True)


if __name__ == "__main__":
    main()
