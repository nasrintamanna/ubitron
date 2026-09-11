"""Package the configuration chosen on validation: RF + time of day + tuned thresholds.

Writes final_model/:
  predictions.npz        same keys as ../rf_results/predictions.npz
                         (y_true_i, y_pred_i, w_test_i) - a drop-in replacement
  metrics.json           per-fold and pooled metrics, mean +/- std over folds
  confusion_matrix.png   pooled, row-normalised, same design as the original

Nothing is retrained here: it combines the saved per-fold test probabilities of
the 'time' experiment with the class weights each fold tuned on its own
validation users.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import CACHE, CLASS_NAMES, FEAT_DIR, HERE, N_FOLDS, RESULTS, confusion, overall

OUT = HERE / "final_model"
OUT.mkdir(exist_ok=True)
BASE_EXP = "time"
weights = json.load(open(RESULTS / f"{BASE_EXP}__thresholds.json"))["weights_per_fold"]

preds, folds, Y, P = {}, [], [], []
for i in range(N_FOLDS):
    z = np.load(CACHE / "proba" / f"{BASE_EXP}_fold_{i}.npz")
    wte = np.load(FEAT_DIR / f"fold_{i}.npz")["wte"]
    yp = (z["Pte"] * np.asarray(weights[i])).argmax(1).astype(np.int8)
    yt = z["yte"].astype(np.int8)
    preds.update({f"y_true_{i}": yt, f"y_pred_{i}": yp, f"w_test_{i}": wte})
    folds.append({"fold": i, "class_weights": weights[i], "test": overall(confusion(yt, yp))})
    Y.append(yt); P.append(yp)
np.savez(OUT / "predictions.npz", **preds)

cm = confusion(np.concatenate(Y), np.concatenate(P))
pooled = overall(cm)
K = ("accuracy", "macro_f1", "balanced_accuracy", "kappa")
json.dump({"configuration": "RandomForest (as submitted) + time-of-day features "
                            "+ per-fold class weights tuned on validation macro-F1",
           "pooled_test": pooled, "confusion_matrix": cm.tolist(), "folds": folds,
           "fold_mean": {k: float(np.mean([f["test"][k] for f in folds])) for k in K},
           "fold_std": {k: float(np.std([f["test"][k] for f in folds], ddof=1)) for k in K}},
          open(OUT / "metrics.json", "w"), indent=1)

base = json.load(open(RESULTS / "baseline.json"))
print("TEST metrics per fold\n")
print(f"  {'fold':>4}" + "".join(f"{k:>19s}" for k in K))
for f in folds:
    print(f"  {f['fold']:>4}" + "".join(f"{f['test'][k]:19.4f}" for k in K))
m = {k: np.array([f['test'][k] for f in folds]) for k in K}
print(f"  {'mean':>4}" + "".join(f"{m[k].mean():19.4f}" for k in K))
print(f"  {'std':>4}" + "".join(f"{m[k].std(ddof=1):19.4f}" for k in K))
print(f"\nPOOLED vs submitted baseline\n")
for k in K:
    b = base["pooled_test"][k]
    print(f"  {k:18s} {b:.4f} -> {pooled[k]:.4f}   ({pooled[k] - b:+.4f})")
print(f"\n  {'class':22s} {'F1 before':>10s} {'F1 after':>10s} {'recall before':>14s} {'recall after':>13s}")
for c in range(7):
    print(f"  {CLASS_NAMES[c]:22s} {base['pooled_test']['per_class_f1'][c]:10.3f} "
          f"{pooled['per_class_f1'][c]:10.3f} {base['pooled_test']['per_class_recall'][c]:14.3f} "
          f"{pooled['per_class_recall'][c]:13.3f}")

# ---- confusion matrix, same design as rf_results/confusion_matrix.png -------
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
SHORT = ["Lying\ndown", "Sitting", "Walking", "Running", "Bicycling",
         "Standing\nin place", "Standing &\nmoving"]
cmr = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(10.2, 8.6))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
fig.subplots_adjust(top=0.845, left=0.20, right=0.94, bottom=0.11)
im = ax.imshow(cmr, cmap=LinearSegmentedColormap.from_list("b", SEQ), vmin=0, vmax=1)
ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
ax.grid(which="minor", color=SURF, linewidth=2.4); ax.tick_params(which="minor", length=0)
for i in range(7):
    for j in range(7):
        v = cmr[i, j]; col = INK if v < 0.45 else "#ffffff"
        ax.text(j, i - 0.11, f"{v * 100:.1f}%", ha="center", va="center", color=col,
                fontsize=11.5 if i == j else 10.5, fontweight="bold" if i == j else "normal")
        ax.text(j, i + 0.24, f"{cm[i, j]:,}", ha="center", va="center", color=col,
                fontsize=7.5, alpha=.6)
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(SHORT, fontsize=9, color=INK2, linespacing=1.35)
ax.set_yticklabels([f"{n}\n{s:,}" for n, s in zip(CLASS_NAMES, cm.sum(1))],
                   fontsize=9, color=INK2, linespacing=1.45)
ax.tick_params(length=0, pad=7)
ax.set_xlabel("Predicted", fontsize=10.5, color=INK2, labelpad=14)
ax.set_ylabel("Actual  ·  segments in class", fontsize=10.5, color=INK2, labelpad=14)
for s in ax.spines.values():
    s.set_visible(False)
fig.text(0.055, 0.955, "Improved Random Forest — pooled confusion matrix",
         fontsize=16, color=INK, fontweight="bold", va="top")
fig.text(0.055, 0.905, f"+ time of day + tuned thresholds · 5 folds · {cm.sum():,} test segments · "
         f"accuracy {pooled['accuracy']:.3f}, macro-F1 {pooled['macro_f1']:.3f}",
         fontsize=9.6, color=INK3, va="top")
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.035)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, length=0, colors=INK3)
cb.set_ticks([0, .25, .5, .75, 1]); cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
fig.savefig(OUT / "confusion_matrix.png", dpi=200, facecolor=SURF)
print(f"\nsaved final_model/predictions.npz, metrics.json, confusion_matrix.png")
