"""Shared code for the final evaluation. Reads the project; writes only here.

Definitions used throughout, stated once so the report can quote them:

* Ground truth  - the ExtraSensory labels, NOT smoothed, then passed through the
                  same episode rule the system uses: an activity episode must
                  last at least 60 s (one full window). Truth and prediction are
                  therefore judged by one definition of "an episode".
* Time base     - seconds from the start of each user's recording.
* Backbones     - "submitted": ../rf_results/predictions.npz
                  "improved" : ../Try_increase_accuracy/final_model/predictions.npz
                  (RF + time of day + tuned thresholds; the final result)
"""
from __future__ import annotations

import json
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE, FIG, TAB = HERE / "cache", HERE / "figures", HERE / "tables"
sys.path.insert(0, str(ROOT))
import activity_timeline as T      # noqa: E402  read-only
import slm_query_engine as S       # noqa: E402  read-only

CLASS_NAMES = S.CLASS_NAMES
BACKBONES = {"submitted": ROOT / "rf_results/predictions.npz",
             "improved": ROOT / "Try_increase_accuracy/final_model/predictions.npz"}
TYPES = ["identification", "verification", "duration", "count",
         "comparison", "grounding", "open_world"]
TYPE_LABEL = {"identification": "Identification", "verification": "Verification",
              "duration": "Duration", "count": "Count", "comparison": "Comparison",
              "grounding": "Grounding", "open_world": "Open-world"}

# correctness rules - one per answer kind, exactly as the report states them
DUR_REL_TOL = 0.10        # duration correct within +/-10 %
COUNT_ABS_TOL = 1         # count correct within +/-1
IOU_THR = 0.5             # a cited interval must reach IoU 0.5
REF_MODALITY, REF_CHANNELS = "Accelerometer, Gyroscope", "All"

VERB = {"Lying down": "lie down", "Sitting": "sit", "Walking": "walk", "Running": "run",
        "Bicycling": "cycle", "Standing in place": "stand in place",
        "Standing and moving": "stand and move"}


def ger(a):
    return a.lower()


# ------------------------------------------------------------ data & timelines
def fold_test_users(f):
    return json.load(open(ROOT / f"balanced_folds/fold_{f}/report.json"))["users"]["test"]


def user_tables(rebuild=False):
    """Per user: windows, segment positions, truth, and both backbones' predictions."""
    path = CACHE / "user_tables.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    preds = {k: np.load(v) for k, v in BACKBONES.items()}
    out = {}
    for f in range(5):
        users = fold_test_users(f)
        u = np.load(ROOT / f"balanced_folds/fold_{f}/u_test.npy")
        w = np.load(ROOT / f"balanced_folds/fold_{f}/w_test.npy")
        yt = preds["submitted"][f"y_true_{f}"]
        assert np.array_equal(yt, preds["improved"][f"y_true_{f}"])
        for j, uid in enumerate(users):
            ss = pd.read_parquet(ROOT / f"segmented_4s/{uid}.parquet",
                                 columns=["seg_start"])["seg_start"].to_numpy()[::128]
            m = np.flatnonzero(u == j)
            seg = ss[ss % 128 == 0] // 128
            assert len(seg) == len(m)
            out[uid] = {"fold": f, "rows": m, "w": w[m], "seg": seg, "y_true": yt[m],
                        "pred": {k: p[f"y_pred_{f}"][m] for k, p in preds.items()}}
    pickle.dump(out, open(path, "wb"))
    return out


def truth_timeline(ut, uid):
    df = T.build_segment_table(ut["w"], ut["seg"], ut["y_true"], ut["y_true"])
    tl = T.build_timeline(df, T.TimelineConfig(smooth="none", min_interval_s=0),
                          user=uid, with_evidence=False)
    return S.clean_timeline(tl), df


def pred_timeline(ut, uid, y_pred, X=None):
    df = T.build_segment_table(ut["w"], ut["seg"], y_pred, ut["y_true"])
    tl = T.build_timeline(df, T.TimelineConfig(), X=X, user=uid, with_evidence=X is not None)
    return S.clean_timeline(tl)


# --------------------------------------------------------- question generation
def generate(tl, uid, rng):
    """Grammatical questions with paraphrases, answers from the TRUTH timeline."""
    iv, tot, cnt = tl["intervals"], tl["totals_s"], tl["counts"]
    if not iv:
        return []
    present = sorted(tot)
    absent = [a for a in CLASS_NAMES if a not in tot]
    Q = []

    def add(t, q, ans, act=None, spans=None, num=None):
        Q.append({"id": f"{uid[:8]}_{t}_{len(Q)}", "user": uid, "type": t, "query": q,
                  "gt": {"answer": ans, "activity": act, "spans": spans, "numeric": num}})

    for _ in range(5):                                            # identification
        x = rng.choice(iv); t = rng.uniform(x["start_s"], x["end_s"])
        q = rng.choice(["What activity is the user performing at {t:.0f} seconds?",
                        "What was the user doing at {t:.0f} seconds?",
                        "Which activity was happening at {t:.0f} seconds from the start?"])
        add("identification", q.format(t=t), x["activity"], x["activity"],
            [(x["start_s"], x["end_s"])])

    for i in range(6):                                            # verification
        x = rng.choice(iv); t = rng.uniform(x["start_s"], x["end_s"])
        a = x["activity"] if i % 2 == 0 else rng.choice([c for c in CLASS_NAMES if c != x["activity"]])
        q = rng.choice(["Is the user {g} at {t:.0f} seconds?", "Was the user {g} at {t:.0f} seconds?"])
        yes = a == x["activity"]
        add("verification", q.format(g=ger(a), t=t), "Yes" if yes else "No", a,
            [(x["start_s"], x["end_s"])] if yes else None)

    for a in rng.sample(present, min(4, len(present))):           # duration
        q = rng.choice(["How long was the user {g}?",
                        "What was the total time the user spent {g}?",
                        "How much time did the user spend {g} in total?"])
        add("duration", q.format(g=ger(a)), f"{tot[a]:.0f} seconds", a,
            [(x["start_s"], x["end_s"]) for x in iv if x["activity"] == a], float(tot[a]))

    for a in rng.sample(present, min(4, len(present))):           # count
        q = rng.choice(["How many separate times did the user {v}?",
                        "How many times did the user {v} during the recording?",
                        "How often did the user {v}?"])
        add("count", q.format(v=VERB[a]), str(cnt[a]), a, None, float(cnt[a]))

    pairs = [(a, b) for k, a in enumerate(present) for b in present[k + 1:] if tot[a] != tot[b]]
    for a, b in rng.sample(pairs, min(4, len(pairs))):           # comparison
        q = rng.choice(["Did the user spend more time {a} or {b}?",
                        "Which took longer overall, {a} or {b}?"])
        add("comparison", q.format(a=ger(a), b=ger(b)), a if tot[a] > tot[b] else b, f"{a}, {b}")

    for a in rng.sample(present, min(3, len(present))) + rng.sample(absent, min(1, len(absent))):
        q = rng.choice(["Did the user begin {g} at any point, and if so, when?",
                        "When did the user first start {g}?"]).format(g=ger(a))
        hits = [x for x in iv if x["activity"] == a]
        if hits:                                                   # grounding
            f = min(hits, key=lambda x: x["start_s"])
            add("grounding", q, "Yes", a, [(f["start_s"], f["end_s"])], float(f["start_s"]))
        else:
            add("grounding", q, "No", a, None)

    for a in rng.sample(["Lying down", "Sitting", "Walking", "Standing in place"], 2):
        long_ = [x for x in iv if x["activity"] == a and x["duration_s"] >= S.PROLONGED_S]
        q = rng.choice(["Did the user {v} for a prolonged period?",
                        "Was there a long stretch of {g}?"]).format(v=VERB[a], g=ger(a))
        add("open_world", q, "Yes" if long_ else "No", f"Prolonged {a.lower()}",
            [(x["start_s"], x["end_s"]) for x in long_] or None)
    cyc = [x for x in iv if x["activity"] == "Bicycling"]
    add("open_world", "Was the user using a wheeled or pedal-based mode of movement?",
        "Yes" if cyc else "No", "Bicycling", [(x["start_s"], x["end_s"]) for x in cyc] or None)
    s_ = sum(v for k, v in tot.items() if k in S.STATIC)
    d_ = sum(v for k, v in tot.items() if k in S.DYNAMIC)
    add("open_world", "Was the user mostly at rest or mostly physically active?",
        "Mostly at rest" if s_ >= d_ else "Mostly physically active", "Overall activity level")
    return Q


# ------------------------------------------------------------------ scoring --
def _num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def _yn(s):
    s = str(s).strip().lower()
    if s.startswith(("yes", "likely yes")):
        return "yes"
    if s.startswith("no"):
        return "no"
    return s.rstrip(".")


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def cited_iou(pred, ref):
    """Match each CITED interval to its best reference interval, then average."""
    if not pred or not ref or isinstance(pred[0], str):
        return None
    return float(np.mean([max(iou(p, r) for r in ref) for p in pred]))


def grounding_precision(pred_spans, activity, df):
    """Fraction of cited intervals whose true labels are mostly the named activity."""
    if not pred_spans or isinstance(pred_spans[0], str) or activity not in CLASS_NAMES:
        return None
    k = CLASS_NAMES.index(activity); ok = []
    for s, e in pred_spans:
        m = (df["t_start"] >= s) & (df["t_end"] <= e)
        if m.sum():
            ok.append(float((df.loc[m, "y_true"] == k).mean() >= 0.5))
    return float(np.mean(ok)) if ok else None


def score_one(q, res, df=None, dur_tol=DUR_REL_TOL, cnt_tol=COUNT_ABS_TOL, iou_thr=IOU_THR):
    g, t = q["gt"], q["type"]
    ans = res["answer"]
    r = {"type": t, "correct": False, "abs_err": None, "rel_err": None, "iou": None}
    ts = res.get("timestamps")
    r["iou"] = cited_iou(ts, g["spans"])
    r["mod_ok"] = (res.get("modality") == REF_MODALITY and res.get("channels") == REF_CHANNELS)

    if t == "identification":
        r["correct"] = str(ans).strip().lower() == g["answer"].lower()
        r["pred_label"], r["true_label"] = str(ans), g["answer"]
    elif t in ("verification", "open_world"):
        r["correct"] = _yn(ans) == _yn(g["answer"])
        r["pred_yn"], r["true_yn"] = _yn(ans), _yn(g["answer"])
    elif t == "comparison":
        r["correct"] = str(ans).strip().lower() == g["answer"].lower()
    elif t == "duration":
        p = _num(ans)
        if p is not None:
            r["abs_err"] = abs(p - g["numeric"])
            r["rel_err"] = r["abs_err"] / g["numeric"] if g["numeric"] else None
            r["correct"] = r["abs_err"] <= dur_tol * g["numeric"]
    elif t == "count":
        p = _num(ans)
        if p is not None:
            r["abs_err"] = abs(p - g["numeric"])
            r["correct"] = r["abs_err"] <= cnt_tol
    elif t == "grounding":                   # Task 3: answer + interval + evidence
        yn_ok = _yn(ans) == _yn(g["answer"])
        if g["answer"] == "No":
            r["correct"] = yn_ok
        else:
            r["correct"] = bool(yn_ok and r["iou"] is not None and r["iou"] >= iou_thr
                                and r["mod_ok"])
        r["answer_ok"] = yn_ok
    # the brief's combined rule, applied to every answer that cites evidence
    has_ref = bool(g["spans"]) and _yn(g["answer"]) != "no"
    r["grounded"] = bool(r["correct"] and has_ref and r["iou"] is not None
                         and r["iou"] >= iou_thr and r["mod_ok"]) if has_ref else None
    if df is not None and has_ref:
        r["gprec"] = grounding_precision(ts, g["activity"], df)
    return r


def aggregate(rows):
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r)
    per = {t: float(np.mean([x["correct"] for x in by[t]])) for t in TYPES if t in by}
    return {"per_type": per, "n": {t: len(by[t]) for t in per},
            "macro": float(np.mean(list(per.values()))),
            "micro": float(np.mean([r["correct"] for r in rows]))}
