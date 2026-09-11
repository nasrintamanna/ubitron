"""Figures 1-3: accuracy by question type, confusion matrix, accuracy vs strictness."""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import fr_common as C

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
SURF, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
sc = pickle.load(open(C.CACHE / "scored.pkl", "rb"))


def frame(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK3)


# ================================================================ FIGURE 1 ===
RULES = {"identification": "exact match", "verification": "exact yes/no",
         "duration": "within ±10%", "count": "within ±1", "comparison": "exact match",
         "grounding": "IoU≥0.5 + evidence", "open_world": "category match"}
# the two systems compared throughout; colour follows the system in figs 1 and 3
SYSTEMS = [("submitted", "Previous system — previous classifier + Qwen2.5-3B", S2),
           ("final", "Upgraded system — upgraded classifier + Qwen2.5-3B", S1)]
cats = C.TYPES + ["overall"]
fig, ax = plt.subplots(figsize=(13.5, 6.6)); fig.patch.set_facecolor(SURF); frame(ax)
fig.subplots_adjust(top=0.80, bottom=0.21, left=0.06, right=0.99)
x = np.arange(len(cats)).astype(float); x[-1] += 0.55              # set Overall apart
w = 0.36
vals = {}
for k, (key, lab, col) in enumerate(SYSTEMS):
    a = sc[key]["agg"]
    vals[key] = np.array([a["per_type"][t] for t in C.TYPES] + [a["macro"]]) * 100
    ax.bar(x + (k - 0.5) * w, vals[key], w * 0.92, color=col, label=lab, zorder=3)
# one label per group: the upgraded value, and its change from the previous system
for xi, up, pr in zip(x, vals["final"], vals["submitted"]):
    ax.text(xi, max(up, pr) + 1.4, f"{up:.1f}", ha="center", va="bottom",
            fontsize=9.2, color=INK2, fontweight="bold")
    ax.text(xi, max(up, pr) + 5.6, f"{up - pr:+.1f}", ha="center", va="bottom",
            fontsize=8.4, color=INK3)
ax.axvline(x[-2] + 0.78, color=GRID, lw=1.2, zorder=1)
n = sc["final"]["agg"]["n"]
ax.set_xticks(x)
ax.set_xticklabels([f"{C.TYPE_LABEL[t]}\n{RULES[t]}\nn={n[t]}" for t in C.TYPES]
                   + ["Overall\nmacro over\n7 types"], fontsize=8.8, color=INK2, linespacing=1.35)
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=8.8)
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.set_ylabel("Questions answered correctly", color=INK2, labelpad=8)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5,
          handlelength=1.1)
fig.text(0.06, 0.965, "Figure 1 — QA accuracy by question type: previous vs upgraded system",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.06, 0.918, f"Same SLM (Qwen2.5-3B) and the same {sum(n.values()):,} questions over all 56 "
         "users; only the activity classifier differs. Labels show the upgraded value (bold) and its "
         "change from the previous system.", fontsize=9.5, color=INK3)
fig.text(0.06, 0.035, "Rules: categorical answers by exact match; duration correct when within ±10% of "
         "the true total; count when within ±1 episode; grounding only when the yes/no answer is right,\n"
         "the cited onset interval reaches IoU ≥ 0.5 with the true one, and the cited modality and "
         "channels match. Overall = unweighted mean of the 7 per-type accuracies.",
         fontsize=8.6, color=INK3, linespacing=1.5)
fig.savefig(C.FIG / "fig1_accuracy_by_question_type.png", dpi=200, facecolor=SURF)
plt.close(fig)
print("fig1 saved")

# ================================================================ FIGURE 2 ===
m = json.load(open(C.ROOT / "Try_increase_accuracy/final_model/metrics.json"))
cm = np.array(m["confusion_matrix"]); cmr = cm / cm.sum(1, keepdims=True)
tp = np.diag(cm); rec = tp / cm.sum(1); prec = tp / np.maximum(cm.sum(0), 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SHORT = ["Lying\ndown", "Sitting", "Walking", "Running", "Bicycling",
         "Standing\nin place", "Standing &\nmoving"]
fig = plt.figure(figsize=(14.2, 8.6)); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[7, 3.0], left=0.15, right=0.985, top=0.83,
                      bottom=0.16, wspace=0.05)
ax = fig.add_subplot(gs[0]); frame(ax)
ax.imshow(cmr, cmap=LinearSegmentedColormap.from_list("b", SEQ), vmin=0, vmax=1)
ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
ax.grid(which="minor", color=SURF, linewidth=2.4); ax.tick_params(which="minor", length=0)
for i in range(7):
    for j in range(7):
        v = cmr[i, j]; col = INK if v < 0.45 else "#ffffff"
        ax.text(j, i - 0.11, f"{v * 100:.1f}%", ha="center", va="center", color=col,
                fontsize=10.5 if i == j else 9.5, fontweight="bold" if i == j else "normal")
        ax.text(j, i + 0.25, f"{cm[i, j]:,}", ha="center", va="center", color=col,
                fontsize=7, alpha=.6)
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(SHORT, fontsize=8.8, color=INK2, linespacing=1.3)
ax.set_yticklabels([f"{c}\n{s:,}" for c, s in zip(C.CLASS_NAMES, cm.sum(1))], fontsize=8.8,
                   color=INK2, linespacing=1.4)
ax.set_xlabel("Predicted activity", color=INK2, labelpad=10)
ax.set_ylabel("True activity  ·  test segments", color=INK2, labelpad=10)
# per-class precision / recall / F1, one row per heatmap row
tx = fig.add_subplot(gs[1]); tx.set_facecolor(SURF); tx.axis("off")
tx.set_xlim(0, 3); tx.set_ylim(6.5, -0.5)
for c, h in enumerate(["precision", "recall", "F1"]):
    tx.text(c + 0.5, -0.75, h, ha="center", va="bottom", fontsize=9.5, color=INK2, fontweight="bold")
for i in range(7):
    for c, v in enumerate([prec[i], rec[i], f1[i]]):
        tx.text(c + 0.5, i, f"{v:.3f}", ha="center", va="center", fontsize=10.5,
                color=INK, fontweight="bold" if c == 2 else "normal")
    tx.axhline(i + 0.5, color=GRID, lw=0.8)
P = m["pooled_test"]
tx.text(1.5, 6.95, f"macro-F1 {P['macro_f1']:.3f} · accuracy {P['accuracy']:.3f}\n"
        f"balanced acc {P['balanced_accuracy']:.3f} · kappa {P['kappa']:.3f}",
        ha="center", va="top", fontsize=9, color=INK2, linespacing=1.6)
fig.text(0.02, 0.965, "Figure 2 — Activity confusion matrix, recognition backbone",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.02, 0.918, f"Random Forest + time of day + tuned thresholds · pooled over 5 subject-wise "
         f"folds · {cm.sum():,} test segments · rows = true activity, normalised so each row sums to 100%",
         fontsize=9.5, color=INK3)
fig.savefig(C.FIG / "fig2_confusion_matrix.png", dpi=200, facecolor=SURF)
plt.close(fig)
print("fig2 saved")

# ================================================================ FIGURE 3 ===
def curve(rows, kind, grid):
    if kind == "dur":
        rs = [r for r in rows if r["type"] == "duration"]
        return [np.mean([r["rel_err"] is not None and r["rel_err"] <= g for r in rs]) for g in grid]
    if kind == "cnt":
        rs = [r for r in rows if r["type"] == "count"]
        return [np.mean([r["abs_err"] is not None and r["abs_err"] <= g for r in rs]) for g in grid]
    if kind == "onset":
        rs = [r for r in rows if r["type"] == "grounding" and r.get("answer_ok") is not None]
        out = []
        for g in grid:
            ok = [bool(r["answer_ok"]) if r["iou"] is None else
                  bool(r["answer_ok"] and r["iou"] >= g and r["mod_ok"]) for r in rs]
            out.append(np.mean(ok))
        return out
    rs = [r for r in rows if r["grounded"] is not None and r["iou"] is not None]
    return [np.mean([r["iou"] >= g for r in rs]) for g in grid]


PANELS3 = [("dur", np.linspace(0, 1.0, 51), "Duration answers — relative error tolerance",
            0.10, lambda v: f"{v * 100:.0f}%", "±10%", np.linspace(0, 1, 6), (0.16, 0.82)),
           ("cnt", np.arange(0, 16), "Count answers — absolute error tolerance (episodes)",
            1, lambda v: f"{v:.0f}", "±1", np.arange(0, 16, 3), (0.16, 0.82)),
           ("onset", np.linspace(0.1, 0.9, 17), "Temporal answers — onset interval, IoU threshold",
            0.5, lambda v: f"{v:.1f}", "IoU 0.5", np.linspace(0.1, 0.9, 9), (0.55, 0.82)),
           ("cited", np.linspace(0.1, 0.9, 17), "Cited evidence intervals — IoU threshold",
            0.5, lambda v: f"{v:.1f}", "IoU 0.5", np.linspace(0.1, 0.9, 9), (0.06, 0.12))]
fig, axs = plt.subplots(2, 2, figsize=(14, 10)); fig.patch.set_facecolor(SURF)
fig.subplots_adjust(top=0.835, bottom=0.07, left=0.065, right=0.985, wspace=0.14, hspace=0.38)
for ax, (kind, grid, title, op, fmt, oplab, ticks, spot) in zip(axs.flat, PANELS3):
    frame(ax)
    at = {}
    for key, lab, col in SYSTEMS:
        y = np.array(curve(sc[key]["rows"], kind, grid)) * 100
        ax.plot(grid, y, color=col, lw=2.2, label=lab, zorder=3)
        at[key] = float(np.interp(op, grid, y))
        ax.scatter([op], [at[key]], s=50, color=col, edgecolor=SURF, linewidth=2, zorder=4)
    ax.annotate(f"at {oplab}\nupgraded {at['final']:.1f}%  ·  previous {at['submitted']:.1f}%",
                (op, at["final"]), xytext=spot, textcoords="axes fraction", fontsize=9,
                color=INK2, zorder=6, ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8, shrinkA=2, shrinkB=4))
    ax.axvline(op, color=INK3, lw=0.9, ls=(0, (2, 3)), zorder=1)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=8.8)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.set_xticks(ticks); ax.set_xticklabels([fmt(v) for v in ticks], fontsize=8.8)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
for ax in axs[:, 0]:
    ax.set_ylabel("Answers accepted", color=INK2, labelpad=8)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.06, 0.925), ncol=2, frameon=False,
           fontsize=9.8, handlelength=1.6)
fig.text(0.065, 0.975, "Figure 3 — Accuracy versus strictness: previous vs upgraded system",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.065, 0.945, "How fast accuracy falls as the correctness rule tightens. A curve that rises "
         "steeply at loose tolerances means the misses are near; a flat one means they are wild. "
         "Dots mark the operating thresholds.", fontsize=9.5, color=INK3)
fig.savefig(C.FIG / "fig3_accuracy_vs_strictness.png", dpi=200, facecolor=SURF)
plt.close(fig)
print("fig3 saved")
