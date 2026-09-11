"""Figure 4 - accuracy versus overhead, with Pareto frontiers.

Every configuration answers the SAME 1,720 questions against the SAME improved
backbone; only the question-parsing SLM differs. Costs come from the
single-query measurements in 02_parse.py, all on one NVIDIA RTX A4500 (20 GB).
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fr_common as C

S = C.S
ORDER = ["rules", "qwen0.5b", "qwen1.5b", "qwen3b_4bit", "qwen3b_8bit", "qwen3b"]
LABEL = {"rules": "Rules only (no SLM)", "qwen0.5b": "Qwen2.5-0.5B fp16",
         "qwen1.5b": "Qwen2.5-1.5B fp16", "qwen3b": "Qwen2.5-3B fp16",
         "qwen3b_8bit": "Qwen2.5-3B 8-bit", "qwen3b_4bit": "Qwen2.5-3B 4-bit"}
ev = pickle.load(open(C.CACHE / "eval_set.pkl", "rb"))
QS, TL = ev["questions"], ev["timelines"]

rows = []
for name in ORDER:
    p = C.CACHE / f"parse_{name}.json"
    if not p.exists():
        print(f"  (skipping {name}: not parsed yet)"); continue
    d = json.load(open(p))
    sc = [C.score_one(q, S.resolve(it, TL[q["user"]]["improved"]), None)
          for q, it in zip(QS, d["intents"])]
    agg = C.aggregate(sc)
    agree = float(np.mean([it["intent"] == q["type"] for q, it in zip(QS, d["intents"])]))
    st = d["stats"]
    rows.append({"name": name, "label": LABEL[name], "acc": agg["macro"], "agree": agree,
                 "size": st["footprint_mb"], "lat": st["latency_median_s"] * 1000,
                 "vram": st["peak_vram_mb"], "energy": st["energy_j_per_query"],
                 "params": st["params"], "per_type": agg["per_type"]})

# 4-bit storage packs two weights per byte, so PyTorch's element count halves;
# a quantized model has the same parameters as its full-precision original.
_full = {r["name"]: r["params"] for r in rows}
for r in rows:
    if r["name"].startswith("qwen3b_") and "qwen3b" in _full:
        r["params"] = _full["qwen3b"]

L = ["# Accuracy versus overhead", "",
     "Same 1,720 questions, same improved backbone; only the question-parsing SLM changes. "
     "Measured one query at a time on an NVIDIA RTX A4500 (20 GB), full answer path "
     "(parse + resolve + explanation).", "",
     "| configuration | params | size (MB) | latency (ms, median) | peak VRAM (MB) | energy (J/query) | "
     "intent parsed correctly | overall QA accuracy |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['label']} | {r['params'] / 1e6:,.0f}M | {r['size']:,.0f} | {r['lat']:,.0f} | "
             f"{r['vram']:,.0f} | {r['energy']:.1f} | {100 * r['agree']:.1f}% | **{100 * r['acc']:.1f}%** |")
(C.TAB / "overhead.md").write_text("\n".join(L) + "\n")
print("\n".join(L))


def pareto(pts):
    """Indices of non-dominated points: nothing cheaper is at least as accurate."""
    idx = sorted(range(len(pts)), key=lambda i: (pts[i][0], -pts[i][1]))
    front, best = [], -1
    for i in idx:
        if pts[i][1] > best + 1e-12:
            front.append(i); best = pts[i][1]
    return front


ACC, MUTE = "#2a78d6", "#b9b8b2"
SURF, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
axes_def = [("size", "Model size in memory (MB)"), ("lat", "Single-query latency (ms, median)"),
            ("vram", "Peak GPU memory (MB)"), ("energy", "Energy per query (J, GPU estimate)")]
fig, axs = plt.subplots(1, 4, figsize=(17, 5.4), sharey=True)
fig.patch.set_facecolor(SURF)
fig.subplots_adjust(top=0.84, bottom=0.14, left=0.05, right=0.99, wspace=0.10)
accs = np.array([r["acc"] for r in rows]) * 100
lo, hi = accs.min(), accs.max()
pad = max(1.5, (hi - lo) * 0.35)
for ax, (key, xl) in zip(axs, axes_def):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_visible(False)
    xs = np.array([r[key] for r in rows])
    front = pareto(list(zip(xs, accs)))
    ax.plot(xs[front], accs[front], color=ACC, lw=2, zorder=2)
    for i, r in enumerate(rows):
        on = i in front
        ax.scatter(xs[i], accs[i], s=70 if on else 50, color=ACC if on else MUTE,
                   edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(r["label"].replace(" fp16", "").replace("Qwen2.5-", ""),
                    (xs[i], accs[i]),
                    xytext={"rules": (6, -15), "qwen3b": (6, -15), "qwen1.5b": (6, 7),
                            "qwen3b_8bit": (6, 7)}.get(r["name"], (6, -15)),
                    textcoords="offset points", fontsize=8.2, color=INK2 if on else INK3)
    ax.set_xlabel(xl, color=INK2, labelpad=8, fontsize=9.5)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK3, labelsize=8.8)
    ax.set_xlim(-0.06 * xs.max(), xs.max() * 1.32)
axs[0].set_ylabel("Overall QA accuracy (macro, %)", color=INK2, labelpad=8)
fig.text(0.05, 0.955, "Figure 4 — Accuracy versus overhead", fontsize=15, fontweight="bold", color=INK)
fig.text(0.05, 0.905, "Each point is one configuration of the question-parsing SLM; blue points are "
         "non-dominated (no cheaper option is as accurate) and are joined into the Pareto frontier. "
         "Measured on an NVIDIA RTX A4500.", fontsize=9.5, color=INK3)
fig.savefig(C.FIG / "fig4_accuracy_vs_overhead.png", dpi=200, facecolor=SURF)
print("\nfig4 saved")
