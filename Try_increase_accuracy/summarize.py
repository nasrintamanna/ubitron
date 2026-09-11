"""Collect every experiment into one comparison table and chart.

Writes results/summary.md and results/comparison.png. Configurations are listed
with their VALIDATION macro-F1 (the number used to choose between them) next to
the pooled TEST metrics (reported, never used for choosing).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS

KEYS = ("accuracy", "macro_f1", "balanced_accuracy", "kappa")


def row(label, val_f1, pooled, group):
    return {"label": label, "val": val_f1, "group": group, **{k: pooled[k] for k in KEYS}}


def load(name):
    p = RESULTS / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


rows = []
PLAN = [  # (experiment, label, group)
    ("baseline", "Baseline RF (as submitted)", "baseline"),
    ("leaf20", "RF min_samples_leaf=20", "regularise"),
    ("leaf50", "RF min_samples_leaf=50", "regularise"),
    ("leaf100", "RF min_samples_leaf=100", "regularise"),
    ("hgb", "Gradient boosting (HGB)", "model"),
    ("twostage", "Two-stage still/moving", "model"),
    ("peruser", "Per-user normalised", "normalise"),
    ("peruser_raw", "Raw + per-user normalised", "normalise"),
    ("time", "+ time of day", "time"),
    ("time_leaf100", "+ time of day, leaf=100", "time"),
]
for name, label, group in PLAN:
    r = load(name)
    if r:
        rows.append(row(label, r["val_macro_f1_mean"], r["pooled_test"], group))
    t = load(f"{name}__thresholds")
    if t:
        for v, suffix in (("prior", "prior-corrected"), ("tuned", "tuned thresholds")):
            vv = t["variants"][v]
            rows.append(row(f"{label} + {suffix}", vv["val_macro_f1_mean"],
                            vv["pooled_test"], group + "+thr"))

base = next(r for r in rows if r["label"].startswith("Baseline RF (as"))
best = max(rows, key=lambda r: r["val"])

lines = ["# Accuracy experiments — comparison", "",
         "Pooled over all 5 subject-wise folds (1,333,415 test segments, every user tested once).",
         "**Selection is by validation macro-F1**; test is reported but never used to choose.", "",
         "| configuration | val macro-F1 | accuracy | macro-F1 | balanced acc | kappa | Δ macro-F1 |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    d = r["macro_f1"] - base["macro_f1"]
    mark = " **←selected**" if r is best else ""
    lines.append(f"| {r['label']}{mark} | {r['val']:.4f} | {r['accuracy']:.4f} | "
                 f"{r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['kappa']:.4f} | "
                 f"{d:+.4f} |")
(RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- chart: macro-F1 and accuracy per configuration, one shared 0-1 axis ----
S1, S2 = "#2a78d6", "#eb6834"
SURF, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
show = [r for r in rows if not r["label"].endswith("prior-corrected")]
show.sort(key=lambda r: r["macro_f1"])
y = np.arange(len(show)); h = 0.38
fig, ax = plt.subplots(figsize=(11, 0.46 * len(show) + 2.2))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.barh(y + h / 2, [r["macro_f1"] for r in show], h * 0.92, color=S1, label="Macro-F1", zorder=3)
ax.barh(y - h / 2, [r["accuracy"] for r in show], h * 0.92, color=S2, label="Accuracy", zorder=3)
for yi, r in zip(y, show):
    ax.text(r["macro_f1"] + 0.006, yi + h / 2, f"{r['macro_f1']:.3f}", va="center",
            fontsize=8.5, color=INK2)
ax.axvline(base["macro_f1"], color=S1, lw=1, ls=(0, (4, 3)), alpha=.6, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([("★ " if r is best else "") + r["label"] for r in show], fontsize=9, color=INK2)
ax.set_xlim(0, 0.75); ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.tick_params(length=0, colors=INK3, labelsize=8.5)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
fig.text(0.01, 0.985, "Which changes improved the activity classifier?", fontsize=14,
         fontweight="bold", color=INK, va="top")
fig.text(0.01, 0.945, "Pooled test over 5 subject-wise folds · dashed line = submitted baseline "
         "macro-F1 · ★ = chosen on validation", fontsize=9, color=INK3, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(RESULTS / "comparison.png", dpi=180, facecolor=SURF)
print(f"\nsaved results/summary.md and results/comparison.png")
