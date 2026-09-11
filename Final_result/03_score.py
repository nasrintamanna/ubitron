"""Step 3 - answer and score every question under three configurations.

  final      improved backbone + Qwen2.5-3B parsing       (the submitted system)
  submitted  submitted backbone + Qwen2.5-3B parsing      (before the upgrade)
  oracle     improved backbone + PERFECT parsing          (isolates parser error)

Writes cache/scored.pkl and the metric tables in tables/.
"""
import json
import pickle
import re

import numpy as np

import fr_common as C

S = C.S
ev = pickle.load(open(C.CACHE / "eval_set.pkl", "rb"))
QS, TL = ev["questions"], ev["timelines"]
intents = json.load(open(C.CACHE / "parse_qwen3b.json"))["intents"]


def oracle_intent(q):
    t, g, text = q["type"], q["gt"], q["query"]
    m = re.search(r"(\d+(?:\.\d+)?)\s*second", text)
    tsec = float(m.group(1)) if m else None
    if t == "identification":
        acts = []
    elif t == "comparison":
        acts = [a.strip() for a in g["activity"].split(",")]
    elif t == "open_world":
        acts = ([a for a in C.CLASS_NAMES if a.lower() in g["activity"].lower()]
                if g["activity"] != "Overall activity level" else [])
    else:
        acts = [g["activity"]]
    return {"intent": t, "activities": acts, "time_s": tsec if t in
            ("identification", "verification") else None, "query": text, "raw": ""}


def run(backbone, parse):
    rows, results = [], []
    for q, it in zip(QS, parse):
        res = S.resolve(it, TL[q["user"]][backbone])
        r = C.score_one(q, res, TL[q["user"]]["df"])
        r.update(id=q["id"], parsed=it["intent"]); rows.append(r); results.append(res)
    return rows, results


CONF = {"final": ("improved", intents),
        "submitted": ("submitted", intents),
        "oracle": ("improved", [oracle_intent(q) for q in QS])}
scored = {}
for name, (bb, parse) in CONF.items():
    rows, res = run(bb, parse)
    scored[name] = {"rows": rows, "results": res, "agg": C.aggregate(rows)}
pickle.dump(scored, open(C.CACHE / "scored.pkl", "wb"))

# ----------------------------------------------------------------- tables ----
def pct(x):
    return f"{100 * x:.1f}%"


agg = {k: v["agg"] for k, v in scored.items()}
L = ["# QA accuracy by question type", "",
     "Overall QA accuracy = fraction correct per question type, **macro-averaged over the "
     "7 types** so the abundant sedentary questions do not dominate.", "",
     "| question type | n | rule | final system | before upgrade | perfect parsing |",
     "|---|---|---|---|---|---|"]
RULE = {"identification": "exact match", "verification": "exact match (yes/no)",
        "duration": "within ±10%", "count": "within ±1", "comparison": "exact match",
        "grounding": "answer + IoU≥0.5 + modality/channels", "open_world": "categorical match"}
for t in C.TYPES:
    L.append(f"| {C.TYPE_LABEL[t]} | {agg['final']['n'][t]} | {RULE[t]} | "
             f"{pct(agg['final']['per_type'][t])} | {pct(agg['submitted']['per_type'][t])} | "
             f"{pct(agg['oracle']['per_type'][t])} |")
L.append(f"| **Overall (macro)** | {len(QS)} | | **{pct(agg['final']['macro'])}** | "
         f"{pct(agg['submitted']['macro'])} | {pct(agg['oracle']['macro'])} |")
L.append(f"| Overall (micro) | {len(QS)} | | {pct(agg['final']['micro'])} | "
         f"{pct(agg['submitted']['micro'])} | {pct(agg['oracle']['micro'])} |")

rows = scored["final"]["rows"]
by = lambda t: [r for r in rows if r["type"] == t]

# categorical detail: identification as a 7-class problem
idr = by("identification")
labs = C.CLASS_NAMES
yt = [labs.index(r["true_label"]) for r in idr]
yp = [labs.index(r["pred_label"]) if r["pred_label"] in labs else -1 for r in idr]
from sklearn.metrics import f1_score, balanced_accuracy_score
yp2 = [p if p >= 0 else (t + 1) % 7 for p, t in zip(yp, yt)]       # unparseable = wrong
L += ["", "## Categorical answers", "",
      "| metric | identification |", "|---|---|",
      f"| accuracy | {pct(np.mean(np.array(yt) == np.array(yp)))} |",
      f"| macro-F1 | {f1_score(yt, yp2, average='macro', labels=range(7), zero_division=0):.3f} |",
      f"| balanced accuracy | {balanced_accuracy_score(yt, yp2):.3f} |",
      "", f"Comparison accuracy {pct(agg['final']['per_type']['comparison'])}, "
          f"open-world categorical accuracy {pct(agg['final']['per_type']['open_world'])}."]

# verification: positive-class metrics
vr = by("verification")
tp = sum(r["pred_yn"] == "yes" and r["true_yn"] == "yes" for r in vr)
fp = sum(r["pred_yn"] == "yes" and r["true_yn"] == "no" for r in vr)
fn = sum(r["pred_yn"] != "yes" and r["true_yn"] == "yes" for r in vr)
tn = sum(r["pred_yn"] != "yes" and r["true_yn"] == "no" for r in vr)
pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
L += ["", "## Binary verification (positive class = Yes)", "",
      "| accuracy | precision | recall | F1 | specificity | TP | FP | FN | TN |",
      "|---|---|---|---|---|---|---|---|---|",
      f"| {pct((tp + tn) / len(vr))} | {pr:.3f} | {rc:.3f} | "
      f"{2 * pr * rc / max(pr + rc, 1e-9):.3f} | {tn / max(tn + fp, 1):.3f} | {tp} | {fp} | {fn} | {tn} |"]

# numeric answers
dr, cr = by("duration"), by("count")
de = [r["abs_err"] for r in dr if r["abs_err"] is not None]
dp = [r["rel_err"] for r in dr if r["rel_err"] is not None]
ce = [r["abs_err"] for r in cr if r["abs_err"] is not None]
L += ["", "## Numeric answers", "",
      "| type | accuracy within tolerance | tolerance | MAE | MAPE | median abs. error |",
      "|---|---|---|---|---|---|",
      f"| duration | {pct(np.mean([r['correct'] for r in dr]))} | ±10% | {np.mean(de):,.0f} s | "
      f"{100 * np.mean(dp):.1f}% | {np.median(de):,.0f} s |",
      f"| count | {pct(np.mean([r['correct'] for r in cr]))} | ±1 | {np.mean(ce):.1f} | — | "
      f"{np.median(ce):.0f} |"]

# temporal / evidence grounding
gr = [r for r in by("grounding")]
gy = [r for r in gr if r.get("answer_ok") is not None]
L += ["", "## Temporal answers and evidence grounding", "",
      "| measure | value |", "|---|---|",
      f"| grounding accuracy (answer + IoU≥0.5 + modality/channels) | "
      f"{pct(np.mean([r['correct'] for r in gr]))} |",
      f"| grounding questions, answer alone correct | {pct(np.mean([r['answer_ok'] for r in gy]))} |",
      f"| median IoU of cited onset interval | "
      f"{np.median([r['iou'] for r in gr if r['iou'] is not None]):.3f} |"]
ev_rows = [r for r in rows if r["grounded"] is not None]
L += ["", "**The price of demanding evidence** — answers whose reference has an interval to cite:", "",
      "| type | n | answer correct | grounded & correct | gap | grounding precision |",
      "|---|---|---|---|---|---|"]
def plain(r):
    """Answer correctness WITHOUT the evidence requirement. For grounding
    questions 'correct' already includes the interval rule, so use answer_ok."""
    return r["answer_ok"] if r["type"] == "grounding" else r["correct"]


for t in C.TYPES:
    e = [r for r in ev_rows if r["type"] == t]
    if not e:
        continue
    a = np.mean([plain(r) for r in e]); g = np.mean([r["grounded"] for r in e])
    gp = [r["gprec"] for r in e if r.get("gprec") is not None]
    L.append(f"| {C.TYPE_LABEL[t]} | {len(e)} | {pct(a)} | {pct(g)} | {100 * (a - g):.1f} pts | "
             f"{pct(np.mean(gp)) if gp else '—'} |")
a = np.mean([plain(r) for r in ev_rows]); g = np.mean([r["grounded"] for r in ev_rows])
gp = [r["gprec"] for r in ev_rows if r.get("gprec") is not None]
L.append(f"| **all** | {len(ev_rows)} | {pct(a)} | {pct(g)} | {100 * (a - g):.1f} pts | {pct(np.mean(gp))} |")

# parser diagnostics
agree = np.mean([r["parsed"] == r["type"] for r in rows])
L += ["", "## Question parsing (Qwen2.5-3B)", "",
      f"Parsed intent equals the question's true type for **{pct(agree)}** of questions."]
for t in C.TYPES:
    e = by(t)
    L.append(f"- {C.TYPE_LABEL[t]}: {pct(np.mean([r['parsed'] == t for r in e]))}")

(C.TAB / "qa_metrics.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
