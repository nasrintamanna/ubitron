"""Shared helpers for the accuracy experiments.

Everything here READS from the parent project and WRITES only inside
Try_increase_accuracy/. The existing models, datasets and results are never
modified - in particular ../rf_features/ is opened read-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # the original project
FEAT_DIR = ROOT / "rf_features"                      # read-only cache
SEG_DIR = ROOT / "segmented_4s"
FOLD_DIR = ROOT / "updated_cv_5_folds"
CACHE = HERE / "cache"
RESULTS = HERE / "results"

N_FOLDS, N_CLASSES, SEG_LEN = 5, 7, 128
CLASS_NAMES = ["Lying down", "Sitting", "Walking", "Running",
               "Bicycling", "Standing in place", "Standing and moving"]
STATIC, DYNAMIC = [0, 1, 5], [2, 3, 4, 6]

# identical to random_forest_model.ipynb, so the baseline reproduces exactly
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=4, max_features="sqrt",
                 class_weight="balanced", n_jobs=24, random_state=0)


def fold_users(i: int, part: str) -> list[str]:
    with open(FOLD_DIR / f"fold_{i}_{part}_uuids.txt") as fh:
        return [l.strip() for l in fh if l.strip()]


def load_fold(i: int) -> dict:
    """Cached features for one fold, read-only, plus recovered training windows."""
    z = np.load(FEAT_DIR / f"fold_{i}.npz")
    d = {k: z[k] for k in z.files}
    wtr = CACHE / f"wtr_fold_{i}.npy"
    if wtr.exists():
        d["wtr"] = np.load(wtr)
    return d


# ------------------------------------------------ metrics (same definitions) --
def confusion(y_true, y_pred, n=N_CLASSES):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def overall(cm) -> dict:
    tp = np.diag(cm).astype(float)
    sup, pred, tot = cm.sum(1), cm.sum(0), cm.sum()
    rec = np.where(sup > 0, tp / np.maximum(sup, 1), np.nan)
    prec = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
    r0 = np.nan_to_num(rec)
    f1 = np.where(prec + r0 > 0, 2 * prec * r0 / np.maximum(prec + r0, 1e-12), 0.0)
    f1 = np.where(sup > 0, f1, np.nan)
    acc = tp.sum() / tot
    pe = float((sup * pred).sum()) / (tot * tot)
    return {"accuracy": float(acc), "macro_f1": float(np.nanmean(f1)),
            "balanced_accuracy": float(np.nanmean(rec)),
            "kappa": float((acc - pe) / (1 - pe)) if pe < 1 else 0.0,
            "per_class_f1": [None if np.isnan(v) else float(v) for v in f1],
            "per_class_recall": [None if np.isnan(v) else float(v) for v in rec]}


def score(y_true, y_pred) -> dict:
    return overall(confusion(y_true, y_pred))


# ----------------------------------------------------------- extra features --
def hour_features(windows: np.ndarray) -> np.ndarray:
    """Local hour of day as (sin, cos), so 23:00 and 01:00 sit close together.

    ExtraSensory was recorded around UC San Diego; the minute-window id is a UTC
    epoch, converted with daylight saving via the tz database.
    """
    import pandas as pd
    t = pd.to_datetime(np.asarray(windows), unit="s", utc=True).tz_convert("America/Los_Angeles")
    h = (t.hour + t.minute / 60.0).to_numpy(dtype=np.float64)
    ang = 2 * np.pi * h / 24.0
    return np.column_stack([np.sin(ang), np.cos(ang)]).astype(np.float32)


def per_user_stats(n_feat: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean/std of every feature over each user's FULL natural recording.

    Every user is a test user in exactly one fold, and the test features cover
    all of that user's non-overlapping segments - so this is each user's whole
    recording, no labels involved. Using the same source for every user keeps
    training and test users normalised identically.
    """
    path = CACHE / "per_user_stats.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["stats"].item()
    stats = {}
    for i in range(N_FOLDS):
        z = np.load(FEAT_DIR / f"fold_{i}.npz")
        Xte, ute = z["Xte"], z["ute"]
        for j, u in enumerate(fold_users(i, "test")):
            F = Xte[ute == j].astype(np.float64)
            stats[u] = (F.mean(0).astype(np.float32),
                        np.maximum(F.std(0), 1e-6).astype(np.float32))
    np.savez(path, stats=np.array(stats, dtype=object))
    return stats


def normalise_per_user(F, uidx, users, stats):
    out = np.empty_like(F, dtype=np.float32)
    for j, u in enumerate(users):
        m = uidx == j
        mu, sd = stats[u]
        out[m] = (F[m] - mu) / sd
    return out


def save_result(name: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.json", "w") as fh:
        json.dump(payload, fh, indent=1)
