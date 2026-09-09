"""Question bank and scorer for the "Ask the Sensors" challenge.

Two halves that share one answer schema:

* ``generate_questions`` builds question/answer pairs from a GROUND-TRUTH
  activity timeline, covering the seven question types the challenge scores -
  identification, verification, duration, count, comparison, grounding and
  open-world reasoning.
* ``score`` applies the challenge's own correctness rules, which differ by type:
  categorical answers by exact match, numeric answers within a tolerance,
  temporal answers by Intersection over Union, and grounded answers only when
  the answer, the cited interval and the cited modality/channels are all right.

The evaluation set is undisclosed, so this is how you measure yourself before
submitting. Generate a dev set from VALIDATION users and tune on that; keep a
set generated from test users for the final numbers.

Answer schema (mirrors the required output format)::

    {"answer": str, "activity_event": str,
     "timestamps": [(start_s, end_s), ...] | None,
     "modality": str, "channels": str, "explanation": str}

Time base: seconds from the start of the recording, matching the timelines.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np

STATIC = {"Lying down", "Sitting", "Standing in place"}
DYNAMIC = {"Walking", "Running", "Bicycling", "Standing and moving"}
NA = "N/A"

# what counts as "prolonged" for the open-world questions, in seconds
PROLONGED_S = 600.0


@dataclass
class Tolerance:
    """Correctness rules. Sweep these to produce the accuracy-vs-strictness figure."""
    duration_rel: float = 0.10     # a duration is correct within +/-10%
    duration_abs: float = 30.0     # ...or within 30 s, whichever is looser
    count_abs: int = 1             # a count is correct within +/-1
    iou: float = 0.5               # a cited interval must reach this IoU
    onset_abs: float = 60.0        # an onset time is correct within 60 s


def _blank(answer, activity=NA, timestamps=None, modality=NA, channels=NA, explanation=NA):
    return {"answer": answer, "activity_event": activity, "timestamps": timestamps,
            "modality": modality, "channels": channels, "explanation": explanation}


# ------------------------------------------------------------- generation ----
def generate_questions(tl: dict, n_per_type: int = 6, seed: int = 0) -> list[dict]:
    """Build questions with known answers from a ground-truth timeline."""
    rng = random.Random(seed)
    iv = tl["intervals"]
    if not iv:
        return []
    totals = tl["totals_s"]
    counts = tl["counts"]
    user = tl.get("user", "")
    present = sorted(totals)
    qs: list[dict] = []

    def add(qtype, query, gt, **extra):
        qs.append({"id": f"{user[:8]}_{qtype}_{len(qs)}", "user": user,
                   "type": qtype, "query": query, "gt": gt, **extra})

    # 1. identification - what is happening at a sampled instant
    for _ in range(n_per_type):
        it = rng.choice(iv)
        t = rng.uniform(it["start_s"], it["end_s"])
        add("identification",
            f"What activity is the user performing at {t:.0f} seconds?",
            _blank(it["activity"], it["activity"], [(it["start_s"], it["end_s"])]))

    # 2. verification - yes/no about an instant, balanced across both answers
    for i in range(n_per_type):
        it = rng.choice(iv)
        t = rng.uniform(it["start_s"], it["end_s"])
        if i % 2 == 0:
            asked = it["activity"]
        else:
            other = [a for a in present if a != it["activity"]]
            asked = rng.choice(other) if other else it["activity"]
        yes = asked == it["activity"]
        add("verification",
            f"Is the user {asked.lower()} at {t:.0f} seconds?",
            _blank("Yes" if yes else "No", asked,
                   [(it["start_s"], it["end_s"])] if yes else None))

    # 3. duration - total time in one activity
    for a in present[:n_per_type]:
        spans = [(x["start_s"], x["end_s"]) for x in iv if x["activity"] == a]
        add("duration",
            f"How long was the user {a.lower()}?",
            _blank(f"{totals[a]:.0f} seconds", a, spans),
            numeric=float(totals[a]))

    # 4. count - number of separate episodes
    for a in present[:n_per_type]:
        add("count",
            f"How many separate times did the user {a.lower()} during the recording?",
            _blank(str(counts[a]), a, None),
            numeric=float(counts[a]))

    # 5. comparison - which of two activities took longer
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]
             if totals[a] != totals[b]]
    rng.shuffle(pairs)
    for a, b in pairs[:n_per_type]:
        winner = a if totals[a] > totals[b] else b
        add("comparison",
            f"Did the user spend more time {a.lower()} or {b.lower()}?",
            _blank(winner, f"{a}, {b}", None))

    # 6. grounding - onset of an activity, evidence required and assessed
    for a in present[:n_per_type]:
        first = min((x for x in iv if x["activity"] == a), key=lambda x: x["start_s"])
        add("grounding",
            f"Did the user begin {a.lower()} at any point, and if so, when?",
            _blank(f"Yes, {a.lower()} began at {first['start_s']:.0f} seconds", a,
                   [(first["start_s"], first["end_s"])],
                   "Accelerometer, Gyroscope", "All"),
            numeric=float(first["start_s"]), evidence_required=True)

    # 7. open-world - behaviour rather than label, answer derivable from the timeline
    lying = [x for x in iv if x["activity"] == "Lying down" and x["duration_s"] >= PROLONGED_S]
    add("open_world", "Did the user lie down for a prolonged period?",
        _blank("Likely yes" if lying else "No", "Prolonged lying down",
               [(x["start_s"], x["end_s"]) for x in lying[:3]] if lying else None,
               "Accelerometer, Gyroscope", "All"),
        evidence_required=bool(lying))

    cyc = [x for x in iv if x["activity"] == "Bicycling" and x["duration_s"] >= 120]
    add("open_world", "Was the user using a wheeled or pedal-based mode of movement?",
        _blank("Yes" if cyc else "No",
               "Unknown outdoor physical activity, consistent with cycling" if cyc else NA,
               [(x["start_s"], x["end_s"]) for x in cyc[:3]] if cyc else None,
               "Accelerometer, Gyroscope", "All"),
        evidence_required=bool(cyc))

    static_s = sum(v for k, v in totals.items() if k in STATIC)
    dynamic_s = sum(v for k, v in totals.items() if k in DYNAMIC)
    add("open_world", "Was the user mostly at rest or mostly physically active?",
        _blank("Mostly at rest" if static_s >= dynamic_s else "Mostly physically active",
               "Overall activity level", None, "Accelerometer, Gyroscope", "All"))
    return qs


# ---------------------------------------------------------------- scoring ----
def _norm(s) -> str:
    return str(s).strip().lower().rstrip(".") if s is not None else ""


def _num(s):
    """First number in a string, so '700 seconds' and 700 both parse."""
    if isinstance(s, (int, float)):
        return float(s)
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def interval_iou(pred, true) -> float:
    """Match predicted intervals to true ones by best overlap, then average."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    return float(np.mean([max(iou(t, p) for p in pred) for t in true]))


def score_one(q: dict, ans: dict, tol: Tolerance) -> dict:
    """Correctness of one answer under the rule for its question type."""
    gt, t = q["gt"], q["type"]
    out = {"id": q["id"], "type": t, "correct": False, "abs_err": None, "iou": None}

    if t in ("identification", "verification", "comparison", "open_world"):
        out["correct"] = _norm(ans.get("answer")) == _norm(gt["answer"])

    elif t == "duration":
        p, g = _num(ans.get("answer")), q["numeric"]
        if p is not None:
            out["abs_err"] = abs(p - g)
            out["correct"] = out["abs_err"] <= max(tol.duration_abs, tol.duration_rel * g)

    elif t == "count":
        p, g = _num(ans.get("answer")), q["numeric"]
        if p is not None:
            out["abs_err"] = abs(p - g)
            out["correct"] = out["abs_err"] <= tol.count_abs

    elif t == "grounding":
        # answer, cited interval, modality and channels must ALL be right
        p = _num(ans.get("answer"))
        answer_ok = p is not None and abs(p - q["numeric"]) <= tol.onset_abs
        ov = interval_iou(ans.get("timestamps") or [], gt["timestamps"] or [])
        out["iou"] = ov
        mod_ok = _norm(ans.get("modality")) == _norm(gt["modality"])
        ch_ok = _norm(ans.get("channels")) == _norm(gt["channels"])
        out["abs_err"] = None if p is None else abs(p - q["numeric"])
        out["answer_ok"] = answer_ok
        out["correct"] = bool(answer_ok and ov >= tol.iou and mod_ok and ch_ok)

    if t != "grounding" and gt.get("timestamps") and ans.get("timestamps"):
        out["iou"] = interval_iou(ans["timestamps"], gt["timestamps"])
    return out


def score(questions: list[dict], answers: list[dict],
          tol: Tolerance | None = None) -> dict:
    """Per-type accuracy, the macro-average, and the error sizes the brief asks for."""
    tol = tol or Tolerance()
    rows = [score_one(q, a, tol) for q, a in zip(questions, answers)]
    by_type: dict[str, dict] = {}
    for r in rows:
        d = by_type.setdefault(r["type"], {"n": 0, "correct": 0, "errs": [], "ious": []})
        d["n"] += 1
        d["correct"] += int(r["correct"])
        if r["abs_err"] is not None:
            d["errs"].append(r["abs_err"])
        if r["iou"] is not None:
            d["ious"].append(r["iou"])

    per_type = {}
    for t, d in sorted(by_type.items()):
        per_type[t] = {
            "n": d["n"],
            "accuracy": d["correct"] / d["n"],
            "mae": float(np.mean(d["errs"])) if d["errs"] else None,
            "median_iou": float(np.median(d["ious"])) if d["ious"] else None,
        }
    macro = float(np.mean([v["accuracy"] for v in per_type.values()])) if per_type else 0.0
    overall = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    return {"per_type": per_type, "macro_accuracy": macro,
            "micro_accuracy": overall, "n": len(rows), "rows": rows}


def strictness_curve(questions, answers, kind="duration",
                     grid=None) -> list[tuple[float, float]]:
    """Accuracy as the correctness rule tightens - the accuracy-vs-strictness figure.

    ``kind='duration'`` sweeps the relative tolerance; ``kind='iou'`` sweeps the
    IoU threshold used for grounded answers.
    """
    pts = []
    if kind == "duration":
        grid = grid if grid is not None else np.arange(0.02, 0.52, 0.02)
        for g in grid:
            s = score(questions, answers, Tolerance(duration_rel=float(g), duration_abs=0.0))
            pts.append((float(g), s["per_type"].get("duration", {}).get("accuracy", 0.0)))
    else:
        grid = grid if grid is not None else np.arange(0.1, 0.95, 0.05)
        for g in grid:
            s = score(questions, answers, Tolerance(iou=float(g)))
            pts.append((float(g), s["per_type"].get("grounding", {}).get("accuracy", 0.0)))
    return pts


# ------------------------------------------------- deterministic answerer ----
def answer_from_timeline(q: dict, tl: dict) -> dict:
    """Answer a question directly from a PREDICTED timeline, no language model.

    This is the arithmetic half of the query engine, and the baseline the SLM
    pipeline must not fall below: it fixes the numbers, so the model is only
    ever responsible for reading the question and writing the explanation.
    """
    iv, totals, counts = tl["intervals"], tl["totals_s"], tl["counts"]
    t, gt = q["type"], q["gt"]
    ev_mod, ev_ch = "Accelerometer, Gyroscope", "All"

    def at(time_s):
        for x in iv:
            if x["start_s"] <= time_s <= x["end_s"]:
                return x
        return min(iv, key=lambda x: min(abs(x["start_s"] - time_s),
                                         abs(x["end_s"] - time_s))) if iv else None

    if t in ("identification", "verification"):
        ts = _num(q["query"].split(" at ")[-1])
        hit = at(ts) if ts is not None else None
        if t == "identification":
            a = hit["activity"] if hit else NA
            return _blank(a, a, [(hit["start_s"], hit["end_s"])] if hit else None,
                          ev_mod, ev_ch)
        asked = gt["activity_event"]
        yes = bool(hit and hit["activity"] == asked)
        return _blank("Yes" if yes else "No", asked,
                      [(hit["start_s"], hit["end_s"])] if yes and hit else None, ev_mod, ev_ch)

    if t in ("duration", "count"):
        a = gt["activity_event"]
        v = totals.get(a, 0.0) if t == "duration" else counts.get(a, 0)
        spans = [(x["start_s"], x["end_s"]) for x in iv if x["activity"] == a]
        return _blank(f"{v:.0f} seconds" if t == "duration" else str(v), a,
                      spans or None, ev_mod, ev_ch)

    if t == "comparison":
        a, b = [s.strip() for s in gt["activity_event"].split(",")]
        win = a if totals.get(a, 0.0) >= totals.get(b, 0.0) else b
        return _blank(win, f"{a}, {b}", None, ev_mod, ev_ch)

    if t == "grounding":
        a = gt["activity_event"]
        hits = [x for x in iv if x["activity"] == a]
        if not hits:
            return _blank("No", a, None, ev_mod, ev_ch)
        f = min(hits, key=lambda x: x["start_s"])
        return _blank(f"Yes, {a.lower()} began at {f['start_s']:.0f} seconds", a,
                      [(f["start_s"], f["end_s"])], ev_mod, ev_ch)

    # open-world
    ql = q["query"].lower()
    if "lie down" in ql:
        hits = [x for x in iv if x["activity"] == "Lying down" and x["duration_s"] >= PROLONGED_S]
        return _blank("Likely yes" if hits else "No", "Prolonged lying down",
                      [(x["start_s"], x["end_s"]) for x in hits[:3]] or None, ev_mod, ev_ch)
    if "wheeled" in ql or "pedal" in ql:
        hits = [x for x in iv if x["activity"] == "Bicycling" and x["duration_s"] >= 120]
        return _blank("Yes" if hits else "No",
                      "Unknown outdoor physical activity, consistent with cycling" if hits else NA,
                      [(x["start_s"], x["end_s"]) for x in hits[:3]] or None, ev_mod, ev_ch)
    s = sum(v for k, v in totals.items() if k in STATIC)
    d = sum(v for k, v in totals.items() if k in DYNAMIC)
    return _blank("Mostly at rest" if s >= d else "Mostly physically active",
                  "Overall activity level", None, ev_mod, ev_ch)
