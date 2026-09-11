"""Recover the window id of every cached TRAINING segment.

random_forest_model.ipynb picks its training segments deterministically
(equal per-user quotas, evenly spaced picks - the rng it is handed is never
used) but discards the window ids. Replaying the same selection over the label
and window columns only recovers them. The recovered labels and user indices are
checked against the cached ones, so a mismatch fails loudly.
"""
import numpy as np
import pandas as pd

from common import CACHE, FEAT_DIR, N_CLASSES, SEG_DIR, SEG_LEN, N_FOLDS, fold_users

TRAIN_CAP_PER_CLASS = 120_000         # as in the RF notebook


def water_fill(avail, target):        # verbatim from the RF notebook
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


def user_yw(u):
    d = pd.read_parquet(SEG_DIR / f"{u}.parquet", columns=["window", "Activity_Labels"])
    return (d["Activity_Labels"].to_numpy()[::SEG_LEN] - 1,
            d["window"].to_numpy()[::SEG_LEN])


for i in range(N_FOLDS):
    users = fold_users(i, "train")
    yw = {u: user_yw(u) for u in users}
    avail = np.stack([np.bincount(yw[u][0], minlength=N_CLASSES) for u in users])
    quota = np.stack([water_fill(avail[:, k], TRAIN_CAP_PER_CLASS)
                      for k in range(N_CLASSES)], axis=1)
    ys, us, ws = [], [], []
    for j, u in enumerate(users):
        y, w = yw[u]
        pick = []
        for k in range(N_CLASSES):
            q = int(quota[j, k])
            if q <= 0:
                continue
            idx = np.flatnonzero(y == k)
            if len(idx) > q:
                idx = idx[np.linspace(0, len(idx) - 1, q).round().astype(int)]
            pick.append(idx)
        if not pick:
            continue
        pick = np.sort(np.concatenate(pick))
        ys.append(y[pick]); us.append(np.full(len(pick), j, np.int16)); ws.append(w[pick])
    y_r, u_r, w_r = np.concatenate(ys), np.concatenate(us), np.concatenate(ws)

    z = np.load(FEAT_DIR / f"fold_{i}.npz")
    ok_y = np.array_equal(y_r, z["ytr"]); ok_u = np.array_equal(u_r, z["utr"])
    if not (ok_y and ok_u):
        raise SystemExit(f"fold {i}: replay does NOT match the cache (y {ok_y}, u {ok_u})")
    np.save(CACHE / f"wtr_fold_{i}.npy", w_r)
    print(f"fold {i}: {len(w_r):,} training windows recovered - labels and users match the cache exactly")
