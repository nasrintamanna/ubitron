"""Suggestion 1: correct for the class balance, then tune decision weights.

  python3 tune_thresholds.py baseline          # on top of any saved experiment

The forest was trained with class_weight="balanced", so its probabilities
behave as if every class were equally common. Test data is not: Sitting is 44%,
Running 0.4%. Two corrections, neither retrains anything:

  prior   multiply each probability by the class's NATURAL frequency among the
          fold's training users (Bayes' rule for a shifted prior), then argmax.
          Aimed at accuracy. Uses no validation or test labels.
  tuned   per-class multipliers chosen on VALIDATION to maximise macro-F1,
          by coordinate ascent, then applied unchanged to test.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import CACHE, N_CLASSES, N_FOLDS, SEG_DIR, SEG_LEN, fold_users, save_result, score

PROBA = CACHE / "proba"
GRID = np.exp(np.linspace(np.log(0.1), np.log(10.0), 41))


def natural_prior(i: int) -> np.ndarray:
    path = CACHE / f"prior_fold_{i}.npy"
    if path.exists():
        return np.load(path)
    c = np.zeros(N_CLASSES)
    for u in fold_users(i, "train"):
        y = pd.read_parquet(SEG_DIR / f"{u}.parquet", columns=["Activity_Labels"])[
            "Activity_Labels"].to_numpy()[::SEG_LEN] - 1
        c += np.bincount(y, minlength=N_CLASSES)
    p = c / c.sum()
    np.save(path, p)
    return p


def macro_f1(y, p):
    return score(y, p)["macro_f1"]


def tune_weights(P, y, rounds=3):
    w = np.ones(N_CLASSES)
    best = macro_f1(y, (P * w).argmax(1))
    for _ in range(rounds):
        improved = False
        for k in range(N_CLASSES):
            for g in GRID:
                w2 = w.copy(); w2[k] = g
                f = macro_f1(y, (P * w2).argmax(1))
                if f > best + 1e-6:
                    best, w, improved = f, w2, True
        if not improved:
            break
    return w, best


def run(name: str) -> dict:
    out = {m: {"y": [], "p": [], "folds": []} for m in ("argmax", "prior", "tuned")}
    weights = []
    for i in range(N_FOLDS):
        z = np.load(PROBA / f"{name}_fold_{i}.npz")
        Pva, Pte, yva, yte = z["Pva"], z["Pte"], z["yva"], z["yte"]
        pri = natural_prior(i)
        w, _ = tune_weights(Pva, yva)
        weights.append(w.tolist())
        for m, pred in (("argmax", Pte.argmax(1)),
                        ("prior", (Pte * pri).argmax(1)),
                        ("tuned", (Pte * w).argmax(1))):
            out[m]["y"].append(yte); out[m]["p"].append(pred)
            vpred = {"argmax": Pva.argmax(1), "prior": (Pva * pri).argmax(1),
                     "tuned": (Pva * w).argmax(1)}[m]
            out[m]["folds"].append({"fold": i, "val": score(yva, vpred), "test": score(yte, pred)})
    res = {"base": name, "weights_per_fold": weights, "variants": {}}
    print(f"\n=== threshold tuning on top of '{name}' ===")
    for m in ("argmax", "prior", "tuned"):
        pooled = score(np.concatenate(out[m]["y"]), np.concatenate(out[m]["p"]))
        vmean = float(np.mean([f["val"]["macro_f1"] for f in out[m]["folds"]]))
        res["variants"][m] = {"val_macro_f1_mean": vmean, "pooled_test": pooled,
                              "folds": out[m]["folds"]}
        print(f"  {m:7s} VAL F1 {vmean:.4f} | POOLED TEST acc {pooled['accuracy']:.4f} "
              f"F1 {pooled['macro_f1']:.4f} bal {pooled['balanced_accuracy']:.4f} "
              f"kappa {pooled['kappa']:.4f}")
    save_result(f"{name}__thresholds", res)
    return res


if __name__ == "__main__":
    for n in sys.argv[1:] or ["baseline"]:
        run(n)
