"""Segments -> activity predictions -> activity timeline, with the upgraded classifier.

Two sources of predictions:
  * a NEW recording: the final model trained on all 56 users (models/*.joblib)
  * one of the 56 KNOWN users: that user's out-of-fold upgraded predictions
    (Try_increase_accuracy/final_model/predictions.npz). Using the final model on
    a user it was trained on would be in-sample and flatter the answers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))
import activity_timeline as T                           # noqa: E402  read-only
import features as F                                    # noqa: E402
from ingest import ingest                               # noqa: E402
from timefeat import hour_features                      # noqa: E402

MODELS = HERE / "models"
OOF = ROOT / "Try_increase_accuracy/final_model/predictions.npz"
WALL_CLOCK_MIN = 946_684_800          # 2000-01-01: minute ids above this are wall-clock epochs
_cache = {}


def load_model(use_time: bool, leaf: int = 4) -> dict:
    name = f"rf_{'time' if use_time else 'notime'}_leaf{leaf}"
    if name not in _cache:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            raise SystemExit(f"{path} missing - run: python3 train_final_model.py"
                             + ("" if use_time else " --no-time"))
        _cache[name] = {**joblib.load(path), "name": name}
    return _cache[name]


def predict_segments(X, windows, bundle):
    Fx = F.extract_chunked(X)
    if bundle["use_time"]:
        Fx = np.hstack([Fx, hour_features(windows)])
    P = np.zeros((len(X), 7), np.float32)
    P[:, bundle["model"].classes_] = bundle["model"].predict_proba(Fx)
    return (P * bundle["weights"]).argmax(1).astype(np.int8), P


def recording_timeline(acc_dir, gyro_dir, label: str = "", clock: str = "auto") -> dict:
    """The full new-recording path: ingest, classify, build the timeline."""
    g = ingest(acc_dir, gyro_dir)
    if len(g["X"]) == 0:
        raise SystemExit(f"no usable segments: {g['report']}")
    wall = bool((g["window"] > WALL_CLOCK_MIN).all())
    use_time = wall if clock == "auto" else clock == "on"
    bundle = load_model(use_time)
    y, _ = predict_segments(g["X"], g["window"], bundle)
    df = T.build_segment_table(g["window"], g["seg_index"], y)
    tl = T.build_timeline(df, T.TimelineConfig(), X=g["X"], user=label)
    tl["source"] = {"kind": "new recording", "acc_dir": str(acc_dir), "gyro_dir": str(gyro_dir),
                    "classifier": bundle["name"], "time_of_day": use_time,
                    "wall_clock_detected": wall, "ingest": g["report"]}
    return tl


def known_user_timeline(prefix: str) -> dict:
    """A known user's timeline from the upgraded OUT-OF-FOLD predictions."""
    pr = np.load(OOF)
    for f in range(5):
        users = json.load(open(ROOT / f"balanced_folds/fold_{f}/report.json"))["users"]["test"]
        hits = [u for u in users if u.startswith(prefix)]
        if not hits:
            continue
        if len(hits) > 1:
            raise SystemExit(f"{prefix!r} matches {len(hits)} users; be more specific")
        uid, j = hits[0], users.index(hits[0])
        u = np.load(ROOT / f"balanced_folds/fold_{f}/u_test.npy")
        m = np.flatnonzero(u == j)
        w = np.load(ROOT / f"balanced_folds/fold_{f}/w_test.npy")[m]
        X = np.load(ROOT / f"balanced_folds/fold_{f}/X_test.npy", mmap_mode="r")[m]
        ss = pd.read_parquet(ROOT / f"segmented_4s/{uid}.parquet",
                             columns=["seg_start"])["seg_start"].to_numpy()[::128]
        seg = ss[ss % 128 == 0] // 128
        df = T.build_segment_table(w, seg, pr[f"y_pred_{f}"][m], pr[f"y_true_{f}"][m])
        tl = T.build_timeline(df, T.TimelineConfig(), X=np.asarray(X), user=uid)
        tl["source"] = {"kind": "known user", "fold": f,
                        "classifier": "RF + time of day + tuned thresholds (out-of-fold)",
                        "predictions": "Try_increase_accuracy/final_model/predictions.npz"}
        return tl
    raise SystemExit(f"{prefix!r} is not one of the 56 users")
