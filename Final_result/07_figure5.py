"""Figure 5 - robustness curves, from tables/robustness.json (06_robustness.py)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fr_common as C

S1, S2 = "#2a78d6", "#eb6834"
SURF, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
R = json.load(open(C.TAB / "robustness.json"))

PANELS = [("noise", "Added Gaussian noise σ (g / rad/s)", lambda v: f"{v:g}", False),
          ("dropout", "Samples dropped per segment (%)", lambda v: f"{v:g}%", False),
          ("rate", "Sampling rate (Hz) — 32 Hz is native", lambda v: f"{v:g}", True)]
fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
fig.patch.set_facecolor(SURF)
fig.subplots_adjust(top=0.80, bottom=0.15, left=0.055, right=0.985, wspace=0.10)
for ax, (kind, xl, fmt, rev) in zip(axs, PANELS):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_visible(False)
    pts = R[kind]
    x = np.array([p["level"] for p in pts], dtype=float)
    labels = x.copy()
    if kind == "noise":                 # levels span two decades: space them evenly
        x = np.arange(len(x), dtype=float)
    for key, lab, col in (("qa_macro", "QA accuracy (macro over types)", S1),
                          ("clf_macro_f1", "Classifier macro-F1", S2)):
        y = np.array([p[key] for p in pts]) * 100
        ax.plot(x, y, color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=lab, zorder=3)
        # start value left of the first point, end value right of the last one;
        # offsets are in display units, so this holds on the reversed rate axis too
        ax.annotate(f"{y[0]:.1f}", (x[0], y[0]), xytext=(-8, 0), textcoords="offset points",
                    fontsize=8.8, color=INK2, ha="right", va="center")
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    fontsize=8.8, color=INK2, ha="left", va="center")
    ax.margins(x=0.13)
    if rev:
        ax.invert_xaxis()
    ax.axhline(25, color="#00000000")
    ax.set_xticks(x); ax.set_xticklabels([fmt(v) for v in labels], fontsize=8.8)
    ax.set_xlabel(xl, color=INK2, labelpad=8, fontsize=9.5)
    ax.set_ylim(0, 80); ax.set_yticks(range(0, 81, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 81, 20)], fontsize=8.8)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK3)
axs[0].set_ylabel("Score on the fixed fold-0 set", color=INK2, labelpad=8)
axs[0].legend(loc="lower left", frameon=False, fontsize=9)
n_q = len(json.load(open(C.TAB / "questions.json")))
fig.text(0.055, 0.955, "Figure 5 — Robustness to degraded sensor input", fontsize=15,
         fontweight="bold", color=INK)
fig.text(0.055, 0.905, "Each point reruns the whole pipeline on corrupted signal — feature extraction, "
         "recognition, timelines, answers — for fold 0's 12 held-out users and their fixed 371 questions. "
         "Degradation worsens left to right.", fontsize=9.5, color=INK3)
fig.savefig(C.FIG / "fig5_robustness.png", dpi=200, facecolor=SURF)
print("fig5 saved")
