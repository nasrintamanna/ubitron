"""Run the accuracy experiments, one per suggestion, on identical folds.

  python3 run_experiments.py baseline leaf20 leaf50 leaf100 hgb peruser time twostage
  python3 run_experiments.py --list

Each experiment trains once per fold on the cached RF features (read-only from
../rf_features), predicts probabilities for validation and test, and saves them
to cache/proba/ so decision-threshold tuning can later be applied on top of any
of them. Results go to results/<name>.json.

Protocol: choices are made on VALIDATION macro-F1 only. Test metrics are
reported for every experiment, but never used to pick one.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from common import (CACHE, N_CLASSES, N_FOLDS, RF_PARAMS, STATIC, DYNAMIC,
                    fold_users, hour_features, load_fold, normalise_per_user,
                    per_user_stats, save_result, score)

PROBA = CACHE / "proba"
PROBA.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- feature sets --
def feats_raw(d, i):
    return d["Xtr"], d["Xva"], d["Xte"]


def feats_time(d, i):
    return (np.hstack([d["Xtr"], hour_features(d["wtr"])]),
            np.hstack([d["Xva"], hour_features(d["wva"])]),
            np.hstack([d["Xte"], hour_features(d["wte"])]))


def _norm(d, i):
    st = per_user_stats(d["Xtr"].shape[1])
    return (normalise_per_user(d["Xtr"], d["utr"], fold_users(i, "train"), st),
            normalise_per_user(d["Xva"], d["uva"], fold_users(i, "val"), st),
            normalise_per_user(d["Xte"], d["ute"], fold_users(i, "test"), st))


def feats_peruser(d, i):
    return _norm(d, i)


def feats_peruser_raw(d, i):
    a, b, c = _norm(d, i)
    return np.hstack([d["Xtr"], a]), np.hstack([d["Xva"], b]), np.hstack([d["Xte"], c])


def feats_peruser_raw_time(d, i):
    a, b, c = feats_peruser_raw(d, i)
    return (np.hstack([a, hour_features(d["wtr"])]),
            np.hstack([b, hour_features(d["wva"])]),
            np.hstack([c, hour_features(d["wte"])]))


# ------------------------------------------------------------------- models --
def rf(**over):
    return lambda: RandomForestClassifier(**{**RF_PARAMS, **over})


def hgb():
    return lambda: HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, class_weight="balanced", early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=0)


EXPERIMENTS = {
    # name: (description, feature builder, model factory)
    "baseline":   ("RF exactly as in random_forest_model.ipynb", feats_raw, rf()),
    "leaf20":     ("RF, min_samples_leaf=20", feats_raw, rf(min_samples_leaf=20)),
    "leaf50":     ("RF, min_samples_leaf=50", feats_raw, rf(min_samples_leaf=50)),
    "leaf100":    ("RF, min_samples_leaf=100", feats_raw, rf(min_samples_leaf=100)),
    "hgb":        ("Histogram gradient boosting", feats_raw, hgb()),
    "peruser":    ("RF on per-user normalised features", feats_peruser, rf()),
    "peruser_raw": ("RF on raw + per-user normalised features", feats_peruser_raw, rf()),
    "time":       ("RF on raw features + time of day", feats_time, rf()),
    "twostage":   ("Two-stage: still/moving, then within group", feats_raw, None),
    "time_leaf100": ("RF on raw features + time of day, min_samples_leaf=100",
                     feats_time, rf(min_samples_leaf=100)),
}


def _proba(model, X):
    P = model.predict_proba(X)
    full = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    full[:, model.classes_] = P
    return full


def two_stage(make, Xtr, ytr, Xs):
    """P(class) = P(group) x P(class | group), fitted as three separate forests."""
    g = np.isin(ytr, STATIC).astype(int)                     # 1 = still
    m_g = make(); m_g.fit(Xtr, g)
    m_s = make(); m_s.fit(Xtr[g == 1], ytr[g == 1])
    m_d = make(); m_d.fit(Xtr[g == 0], ytr[g == 0])
    outs = []
    for X in Xs:
        pg = m_g.predict_proba(X)                             # columns [moving, still]
        p_still = pg[:, list(m_g.classes_).index(1)]
        P = np.zeros((len(X), N_CLASSES), dtype=np.float32)
        P[:, m_s.classes_] = m_s.predict_proba(X) * p_still[:, None]
        P[:, m_d.classes_] = m_d.predict_proba(X) * (1 - p_still)[:, None]
        outs.append(P)
    return outs


def run(name: str) -> dict:
    desc, build, make = EXPERIMENTS[name]
    print(f"\n=== {name}: {desc} ===", flush=True)
    folds, yte_all, pte_all = [], [], []
    for i in range(N_FOLDS):
        d = load_fold(i)
        Xtr, Xva, Xte = build(d, i)
        t0 = time.time()
        if name == "twostage":
            Pva, Pte = two_stage(rf(), Xtr, d["ytr"], [Xva, Xte])
        else:
            m = make(); m.fit(Xtr, d["ytr"])
            Pva, Pte = _proba(m, Xva), _proba(m, Xte)
        fit_s = time.time() - t0
        np.savez(PROBA / f"{name}_fold_{i}.npz", Pva=Pva, Pte=Pte,
                 yva=d["yva"], yte=d["yte"])
        v = score(d["yva"], Pva.argmax(1)); t = score(d["yte"], Pte.argmax(1))
        folds.append({"fold": i, "fit_seconds": fit_s, "n_features": int(Xtr.shape[1]),
                      "val": v, "test": t})
        yte_all.append(d["yte"]); pte_all.append(Pte.argmax(1))
        print(f"  fold {i}: {fit_s:5.0f}s  {Xtr.shape[1]} feats | "
              f"VAL F1 {v['macro_f1']:.4f} | TEST acc {t['accuracy']:.4f} "
              f"F1 {t['macro_f1']:.4f} bal {t['balanced_accuracy']:.4f} "
              f"kappa {t['kappa']:.4f}", flush=True)
        del d, Xtr, Xva, Xte
    pooled = score(np.concatenate(yte_all), np.concatenate(pte_all))
    res = {"name": name, "description": desc, "folds": folds,
           "val_macro_f1_mean": float(np.mean([f["val"]["macro_f1"] for f in folds])),
           "pooled_test": pooled,
           "fold_mean_test": {k: float(np.mean([f["test"][k] for f in folds]))
                              for k in ("accuracy", "macro_f1", "balanced_accuracy", "kappa")},
           "fold_std_test": {k: float(np.std([f["test"][k] for f in folds], ddof=1))
                             for k in ("accuracy", "macro_f1", "balanced_accuracy", "kappa")}}
    save_result(name, res)
    print(f"  -> VAL F1 mean {res['val_macro_f1_mean']:.4f} | POOLED TEST "
          f"acc {pooled['accuracy']:.4f} F1 {pooled['macro_f1']:.4f} "
          f"bal {pooled['balanced_accuracy']:.4f} kappa {pooled['kappa']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.names:
        for k, (desc, *_) in EXPERIMENTS.items():
            print(f"  {k:12s} {desc}")
        sys.exit(0)
    for n in a.names:
        if n not in EXPERIMENTS:
            sys.exit(f"unknown experiment {n!r}")
        run(n)
