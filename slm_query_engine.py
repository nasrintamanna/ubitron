"""SLM query engine: natural-language questions over an activity timeline.

The language model does two jobs and no others:

1. **Intent parsing** - turn a free-text question into a structured intent
   (which operation, which activities, which time range).
2. **Explanation** - write the Explanation field from evidence that has already
   been retrieved from the signal.

Everything numeric happens in Python, over the full timeline. The model never
sees the timeline and never does arithmetic, so a duration or a count cannot be
hallucinated: it is computed from the intervals the recognition layer produced.
This is what the challenge means by grounding - the language is tied to evidence
the earlier layers found, not generated freely.

    question ──▶ Qwen (parse) ──▶ intent
                                    │
                    timeline ──▶ Python resolve ──▶ answer + cited intervals
                                    │
                        evidence ──▶ Qwen (explain) ──▶ Explanation
                                    │
                                    ▼
                          formatted answer block

Answering an unsupported question returns N/A in every field rather than an
invented answer: the format allows N/A, and a fabricated answer scores worse
than a declined one.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

CLASS_NAMES = ["Lying down", "Sitting", "Walking", "Running",
               "Bicycling", "Standing in place", "Standing and moving"]
STATIC = {"Lying down", "Sitting", "Standing in place"}
DYNAMIC = {"Walking", "Running", "Bicycling", "Standing and moving"}
NA = "N/A"
MODALITY = "Accelerometer, Gyroscope"
CHANNELS = "All"

INTENTS = ["identification", "verification", "duration", "count",
           "comparison", "grounding", "open_world", "unsupported"]

# Verb and noun forms people actually type, so the rule fallback works without
# the model ("how many times did she walk?" names no activity literally).
VERB_FORMS = {
    "walk": "Walking", "stroll": "Walking",
    "run": "Running", "jog": "Running", "ran": "Running",
    "sit": "Sitting", "sat": "Sitting", "seated": "Sitting",
    "lie ": "Lying down", "lay ": "Lying down", "lying": "Lying down",
    "laid": "Lying down", "rest": "Lying down",
    "cycl": "Bicycling", "bike": "Bicycling", "biking": "Bicycling",
    "bicycl": "Bicycling", "pedal": "Bicycling",
    "stand": "Standing in place", "stood": "Standing in place",
}

# how many intervals of evidence to show the model when it writes an explanation
MAX_EVIDENCE_INTERVALS = 3

PARSE_SYSTEM = """You convert questions about a wearable-sensor recording into JSON.

Activities: Lying down, Sitting, Walking, Running, Bicycling, Standing in place, Standing and moving

Intents:
- identification: what activity is happening (optionally at a time)
- verification: yes/no, is the user doing a named activity
- duration: how long / how much total time in an activity
- count: how many times / how many separate episodes
- comparison: which of two activities took more time
- grounding: when did an activity start, or when did it happen
- open_world: behaviour not named by a label (prolonged rest, wheeled movement, overall activity level)
- unsupported: anything the sensors cannot answer (heart rate, location, mood, identity)

Reply with ONLY a JSON object:
{"intent": "<one intent>", "activities": ["<activity>", ...], "time_s": <number or null>}

Rules:
- "activities" uses the exact activity names above. Empty list if none named.
- "time_s" is a number of seconds only if the question names one, else null.
- comparison must list exactly two activities.
- No text outside the JSON."""

PARSE_EXAMPLES = [
    ("What activity is the user performing at 900 seconds?",
     '{"intent": "identification", "activities": [], "time_s": 900}'),
    ("Is the user running?",
     '{"intent": "verification", "activities": ["Running"], "time_s": null}'),
    ("How long did she walk in total?",
     '{"intent": "duration", "activities": ["Walking"], "time_s": null}'),
    ("How many times did the user go for a walk?",
     '{"intent": "count", "activities": ["Walking"], "time_s": null}'),
    ("Did the user spend more time walking or running?",
     '{"intent": "comparison", "activities": ["Walking", "Running"], "time_s": null}'),
    ("Did she begin cycling at any point, and when?",
     '{"intent": "grounding", "activities": ["Bicycling"], "time_s": null}'),
    ("Did the user lie down for a prolonged period?",
     '{"intent": "open_world", "activities": ["Lying down"], "time_s": null}'),
    ("What was her heart rate during the walk?",
     '{"intent": "unsupported", "activities": [], "time_s": null}'),
]

EXPLAIN_SYSTEM = """You write one short Explanation for a sensor question-answering system.

You are given the answer and the measured signal evidence behind it, as JSON.
Write ONE sentence, at most two, saying why the signal supports the answer.

Hard rules:
- Use ONLY numbers that appear verbatim in the JSON. Copy them exactly.
- Never describe a change, rise, fall or transition between two values unless
  BOTH values appear in the JSON. Describe what the values ARE, not how they moved.
- Never mention a feature that is absent from the JSON. If step_freq_hz is not
  given, do not mention step frequency at all.
- Do not restate the answer; explain the evidence for it.
Reply with the explanation sentence only, no preamble."""


# --------------------------------------------------------------- the model ---
class QwenEngine:
    """Wraps the SLM. Loaded once; both calls are short."""

    def __init__(self, model_id: str = MODEL_ID, dtype="float16", device_map="auto",
                 quantization: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"dtype": getattr(torch, dtype), "device_map": device_map}
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=torch.float16)
            kw.pop("dtype")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        self.quantization = quantization
        self.stats = {"parse_calls": 0, "explain_calls": 0, "tokens_in": 0,
                      "tokens_out": 0, "seconds": 0.0}

    def _chat(self, system: str, turns: list[tuple[str, str]], user: str,
              max_new_tokens: int = 96, temperature: float = 0.0) -> str:
        import torch
        msgs = [{"role": "system", "content": system}]
        for u, a in turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=temperature or None,
                pad_token_id=self.tok.eos_token_id)
        dt = time.perf_counter() - t0
        gen = out[0][enc["input_ids"].shape[1]:]
        self.stats["tokens_in"] += int(enc["input_ids"].shape[1])
        self.stats["tokens_out"] += int(gen.shape[0])
        self.stats["seconds"] += dt
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    # ---- job 1: question -> intent ----
    def parse_intent(self, query: str) -> dict:
        raw = self._chat(PARSE_SYSTEM, PARSE_EXAMPLES, query, max_new_tokens=80)
        self.stats["parse_calls"] += 1
        return normalise_intent(raw, query)

    # ---- job 2: evidence -> explanation ----
    def explain(self, answer: str, activity: str, facts: dict) -> str:
        payload = json.dumps({"answer": answer, "activity": activity, **facts},
                             separators=(",", ":"))
        self.stats["explain_calls"] += 1
        txt = self._chat(EXPLAIN_SYSTEM, [], payload, max_new_tokens=110)
        return " ".join(txt.split())


# ------------------------------------------------------- intent post-parse ---
def normalise_intent(raw: str, query: str = "") -> dict:
    """Parse the model's JSON, repair what is repairable, fall back on rules."""
    obj = {}
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            obj = {}

    intent = str(obj.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        intent = _rule_intent(query)

    acts, low = [], {c.lower(): c for c in CLASS_NAMES}
    for a in obj.get("activities") or []:
        c = low.get(str(a).strip().lower())
        if c and c not in acts:
            acts.append(c)
    if not acts:                                   # recover names from the text
        ql_ = query.lower()
        for c in CLASS_NAMES:
            if c.lower() in ql_ and c not in acts:
                acts.append(c)
    if not acts:                                   # then by verb form
        ql_ = query.lower()
        for stem, c in VERB_FORMS.items():
            if re.search(rf"\b{stem}", ql_) and c not in acts:
                acts.append(c)

    t = obj.get("time_s")
    try:
        t = float(t) if t is not None else None
    except (TypeError, ValueError):
        t = None
    if t is None:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*second", query.lower())
        if m2:
            t = float(m2.group(1))

    # Guardrail: these phrasings ask about behaviour, not about a label's onset.
    # The 3B model reliably mislabels them as "grounding", which produces an
    # answer of the wrong shape ("began at 6395 seconds" instead of "Likely yes").
    ql = query.lower()
    if any(k in ql for k in ("prolonged", "wheeled", "pedal-based", "pedal based",
                             "mostly at rest", "mostly active", "physically active")):
        intent = "open_world"

    if intent == "comparison" and len(acts) < 2:
        intent = "duration" if acts else "unsupported"
    return {"intent": intent, "activities": acts, "time_s": t, "raw": raw}


def _rule_intent(q: str) -> str:
    """Keyword fallback for when the model returns something unusable."""
    s = q.lower()
    if " or " in s and any(w in s for w in ("more", "longer", "spend")):
        return "comparison"
    if s.startswith(("how many", "how often")) or "times" in s:
        return "count"
    if s.startswith("how long") or "total time" in s or "how much time" in s:
        return "duration"
    if "when" in s or "begin" in s or "start" in s or "onset" in s:
        return "grounding"
    if s.startswith(("is ", "was ", "did ", "does ")):
        return "verification"
    if "what activity" in s or "what is" in s or "doing" in s:
        return "identification"
    return "unsupported"


# ------------------------------------------- deterministic resolution layer --
def _at(intervals, t):
    for x in intervals:
        if x["start_s"] <= t <= x["end_s"]:
            return x
    return None


def _spans(intervals, activity, limit=None):
    hits = [x for x in intervals if x["activity"] == activity]
    hits.sort(key=lambda x: -x["duration_s"])
    return hits[:limit] if limit else hits


def _facts(intervals_cited, extra=None):
    """Compact evidence for the explanation prompt - only real measurements."""
    f = dict(extra or {})
    ev = []
    for x in intervals_cited[:MAX_EVIDENCE_INTERVALS]:
        e = x.get("evidence") or {}
        ev.append({k: v for k, v in {
            "start_s": x["start_s"], "end_s": x["end_s"],
            "duration_s": x["duration_s"],
            "acc_mag_mean": e.get("acc_mag_mean"),
            "acc_mag_std": e.get("acc_mag_std"),
            "dynamic_acc_std": e.get("dynamic_acc_std"),
            "gyro_mag_mean": e.get("gyro_mag_mean"),
            "gyro_mag_std": e.get("gyro_mag_std"),
            "step_freq_hz": e.get("step_freq_hz"),
            "gravity_dir": e.get("gravity_dir"),
        }.items() if v is not None})
    if ev:
        f["intervals"] = ev
    return f


def resolve(intent: dict, tl: dict) -> dict:
    """Compute the answer from the timeline. No model, no arithmetic by the SLM."""
    iv, totals, counts = tl["intervals"], tl["totals_s"], tl["counts"]
    kind, acts, t = intent["intent"], intent["activities"], intent["time_s"]
    blank = {"answer": NA, "activity_event": NA, "timestamps": None,
             "modality": NA, "channels": NA, "facts": {}}

    if kind == "unsupported" or not iv:
        return blank

    if kind == "identification":
        hit = _at(iv, t) if t is not None else (
            max(totals, key=totals.get) if totals else None)
        if t is not None:
            if hit is None:
                return blank
            cited = [hit]
            a = hit["activity"]
        else:
            a = hit
            cited = _spans(iv, a, MAX_EVIDENCE_INTERVALS)
        return {"answer": a, "activity_event": a,
                "timestamps": [(x["start_s"], x["end_s"]) for x in cited],
                "modality": MODALITY, "channels": CHANNELS, "facts": _facts(cited)}

    if kind == "verification":
        if not acts:
            return blank
        a = acts[0]
        if t is not None:
            hit = _at(iv, t)
            yes = bool(hit and hit["activity"] == a)
            cited = [hit] if yes and hit else []
        else:
            cited = _spans(iv, a, MAX_EVIDENCE_INTERVALS)
            yes = bool(cited)
        return {"answer": "Yes" if yes else "No", "activity_event": a,
                "timestamps": [(x["start_s"], x["end_s"]) for x in cited] or None,
                "modality": MODALITY, "channels": CHANNELS,
                "facts": _facts(cited, {"total_s": totals.get(a, 0.0)})}

    if kind == "duration":
        if not acts:
            return blank
        a = acts[0]
        cited = _spans(iv, a)[:MAX_EVIDENCE_INTERVALS]   # cite the longest, not all
        return {"answer": f"{totals.get(a, 0.0):.0f} seconds", "activity_event": a,
                "timestamps": [(x["start_s"], x["end_s"]) for x in cited] or None,
                "modality": MODALITY, "channels": CHANNELS,
                "facts": _facts(cited,
                                {"total_s": totals.get(a, 0.0),
                                 "n_intervals": counts.get(a, 0)})}

    if kind == "count":
        if not acts:
            return blank
        a = acts[0]
        cited = _spans(iv, a, MAX_EVIDENCE_INTERVALS)
        return {"answer": str(counts.get(a, 0)), "activity_event": a,
                "timestamps": [(x["start_s"], x["end_s"]) for x in cited] or None,
                "modality": MODALITY, "channels": CHANNELS,
                "facts": _facts(cited, {"n_intervals": counts.get(a, 0),
                                        "total_s": totals.get(a, 0.0)})}

    if kind == "comparison":
        if len(acts) < 2:
            return blank
        a, b = acts[0], acts[1]
        ta, tb = totals.get(a, 0.0), totals.get(b, 0.0)
        win = a if ta >= tb else b
        return {"answer": win, "activity_event": f"{a}, {b}",
                "timestamps": [(f"{a} = {ta:.0f} seconds total, "
                                f"{b} = {tb:.0f} seconds total")],
                "modality": MODALITY, "channels": CHANNELS,
                "facts": {f"{a}_total_s": ta, f"{b}_total_s": tb}}

    if kind == "grounding":
        if not acts:
            return blank
        a = acts[0]
        hits = [x for x in iv if x["activity"] == a]
        if not hits:
            return {"answer": "No", "activity_event": a, "timestamps": None,
                    "modality": MODALITY, "channels": CHANNELS, "facts": {}}
        first = min(hits, key=lambda x: x["start_s"])
        return {"answer": f"Yes, {a.lower()} began at {first['start_s']:.0f} seconds",
                "activity_event": f"Onset of {a.lower()}",
                "timestamps": [(first["start_s"], first["end_s"])],
                "modality": MODALITY, "channels": CHANNELS,
                "facts": _facts([first], {"onset_s": first["start_s"]})}

    # open_world
    a = acts[0] if acts else None
    if a == "Lying down" or "lie down" in str(intent.get("raw", "")).lower():
        hits = [x for x in iv if x["activity"] == "Lying down" and x["duration_s"] >= 600]
        hits.sort(key=lambda x: -x["duration_s"])
        return {"answer": "Likely yes" if hits else "No",
                "activity_event": "Prolonged lying down",
                "timestamps": [(x["start_s"], x["end_s"]) for x in hits[:3]] or None,
                "modality": MODALITY, "channels": CHANNELS,
                "facts": _facts(hits[:3], {"longest_s": hits[0]["duration_s"] if hits else 0})}
    if a == "Bicycling":
        hits = [x for x in iv if x["activity"] == "Bicycling" and x["duration_s"] >= 120]
        hits.sort(key=lambda x: -x["duration_s"])
        return {"answer": "Yes" if hits else "No",
                "activity_event": ("Unknown outdoor physical activity, consistent with cycling"
                                   if hits else NA),
                "timestamps": [(x["start_s"], x["end_s"]) for x in hits[:3]] or None,
                "modality": MODALITY, "channels": CHANNELS, "facts": _facts(hits[:3])}
    s = sum(v for k, v in totals.items() if k in STATIC)
    d = sum(v for k, v in totals.items() if k in DYNAMIC)
    return {"answer": "Mostly at rest" if s >= d else "Mostly physically active",
            "activity_event": "Overall activity level", "timestamps": None,
            "modality": MODALITY, "channels": CHANNELS,
            "facts": {"static_total_s": round(s, 1), "dynamic_total_s": round(d, 1)}}


# ------------------------------------------------------------- formatting ---
def _fmt_ts(ts) -> str:
    if not ts:
        return NA
    if isinstance(ts[0], str):
        return ts[0]
    return ", ".join(f"{a:.0f} to {b:.0f}" for a, b in ts) + " (seconds from start)"


def format_answer(res: dict, explanation: str = NA) -> str:
    """The exact output block the challenge requires."""
    return (f"Answer: {res['answer']}\n"
            f"Activity/Event: {res['activity_event']}\n"
            f"Evidence:\n"
            f"    Timestamp(s): {_fmt_ts(res.get('timestamps'))}\n"
            f"    Sensor Modality: {res.get('modality', NA)}\n"
            f"    Sensor Channel(s): {res.get('channels', NA)}\n"
            f"Explanation: {explanation or NA}")


def answer_query(query: str, tl: dict, engine: QwenEngine | None = None,
                 explain: bool = True) -> dict:
    """Full pipeline for one question."""
    t0 = time.perf_counter()
    intent = engine.parse_intent(query) if engine else normalise_intent("", query)
    res = resolve(intent, tl)
    expl = NA
    if explain and engine and res["answer"] != NA:
        expl = engine.explain(res["answer"], res["activity_event"], res.get("facts", {}))
    return {"query": query, "intent": intent, "result": res, "explanation": expl,
            "text": format_answer(res, expl), "seconds": time.perf_counter() - t0}
