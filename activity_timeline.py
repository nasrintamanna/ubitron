"""Aggregation layer: per-window activity predictions -> an activity timeline.

This is the layer between the recognition backbone (Random Forest / CNN) and the
SLM query engine. The classifier emits one label per 4-second window; questions
like "how long was the user walking?" or "when did running begin?" need
intervals, durations, counts and transitions instead. This module produces them,
together with the signal evidence each interval rests on, so the SLM can cite
real measurements rather than inventing them.

Pipeline
--------
    per-segment predictions
      -> absolute time base          (seconds from the start of the recording)
      -> temporal smoothing          (mode filter, or Viterbi over an HMM)
      -> interval merging            (consecutive same-class runs, gap-tolerant)
      -> per-interval evidence       (signal features the explanation can cite)
      -> timeline + aggregates       (totals, counts, transitions)

Two properties of ExtraSensory shape the design:

1. The recording is *duty-cycled*: roughly 20 seconds of signal are captured per
   minute, not a continuous stream. So an interval has two different and equally
   legitimate durations. ``duration_s`` is the wall-clock span from the first to
   the last segment - the right answer to "how long was the user walking?",
   because the activity continued through the unsampled gaps. ``observed_s`` is
   the signal actually recorded (n_segments x 4 s). Both are reported; answers
   about elapsed time should use ``duration_s``, and evidence about the signal
   should use ``observed_s``.

2. Per-window predictions are noisy, so raw runs fragment into hundreds of
   one-segment flickers. Smoothing before merging matters more to the temporal
   answers than classifier accuracy does.

Time base: seconds from the start of the recording, anchored on the epoch of the
first window. State this convention wherever answers are reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- constants --
SEG_SECONDS = 4.0          # one segment = 128 samples at 32 Hz
SAMPLE_RATE = 32.0
SEG_LEN = 128
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CLASS_NAMES = ["Lying down", "Sitting", "Walking", "Running",
               "Bicycling", "Standing in place", "Standing and moving"]
N_CLASSES = len(CLASS_NAMES)

# gait band used when looking for a step / pedal frequency
GAIT_BAND_HZ = (0.5, 4.0)


@dataclass
class TimelineConfig:
    """Knobs for the aggregation layer. Tune on validation, never on test."""
    # Default is the mode filter: measured on fold 0 it gives the best temporal
    # IoU against true intervals (median 0.368 vs 0.303 raw and 0.278 for
    # Viterbi). Viterbi yields far fewer intervals but over-smooths, merging
    # short true intervals into long predicted ones that overshoot the boundary.
    # Fewest intervals is not the goal; matching real boundaries is.
    smooth: str = "mode"           # "mode" | "viterbi" | "none"
    mode_k: int = 5                # window (segments) for the mode filter, odd
    p_stay: float = 0.95           # HMM self-transition prior
    max_gap_s: float = 120.0       # merge same-class runs separated by <= this
    min_interval_s: float = 8.0    # drop intervals shorter than this
    emission: np.ndarray | None = field(default=None, repr=False)
    # emission[t, p] = P(predicted p | true t); supply from VALIDATION data


# -------------------------------------------------------------- time base ----
def build_segment_table(window: np.ndarray, seg_index: np.ndarray,
                        y_pred: np.ndarray, y_true: np.ndarray | None = None,
                        proba: np.ndarray | None = None) -> pd.DataFrame:
    """One row per segment, with an absolute time in seconds from recording start.

    ``window`` is the minute-aligned epoch id of the parent window, which is the
    real clock; ``seg_index`` locates the segment inside it. Times are therefore
    (window - first_window) + seg_index * 4 s.
    """
    t0 = int(np.min(window))
    df = pd.DataFrame({
        "window": window.astype(np.int64),
        "seg_index": seg_index.astype(np.int32),
        "y_pred": y_pred.astype(np.int8),
    })
    if y_true is not None:
        df["y_true"] = y_true.astype(np.int8)
    df["t_start"] = (df["window"] - t0) + df["seg_index"] * SEG_SECONDS
    df["t_end"] = df["t_start"] + SEG_SECONDS
    if proba is not None:
        df["confidence"] = proba.max(axis=1)
        for k in range(proba.shape[1]):
            df[f"p{k}"] = proba[:, k]
    df = df.sort_values("t_start", kind="stable").reset_index(drop=True)
    df.attrs["t0_epoch"] = t0
    return df


# -------------------------------------------------------------- smoothing ----
def smooth_mode(labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Majority filter over a centred window of k segments."""
    if k <= 1:
        return labels.copy()
    half = k // 2
    padded = np.pad(labels, half, mode="edge")
    out = np.empty_like(labels)
    for i in range(len(labels)):
        win = padded[i:i + k]
        counts = np.bincount(win, minlength=N_CLASSES)
        best = counts.max()
        # tie -> keep the current label if it is among the winners
        out[i] = labels[i] if counts[labels[i]] == best else int(counts.argmax())
    return out


def smooth_viterbi(labels: np.ndarray, emission: np.ndarray,
                   p_stay: float = 0.95, log_proba: np.ndarray | None = None) -> np.ndarray:
    """Most likely true-state sequence under a simple HMM.

    States are the seven activities. The transition matrix encodes only that
    activities persist (``p_stay`` on the diagonal, the rest spread uniformly) -
    a prior, not something estimated from the data being decoded. Emissions are
    P(predicted | true), which must come from validation, never from the
    sequence being smoothed.
    """
    n = len(labels)
    if n == 0:
        return labels.copy()

    trans = np.full((N_CLASSES, N_CLASSES), (1.0 - p_stay) / (N_CLASSES - 1))
    np.fill_diagonal(trans, p_stay)
    log_t = np.log(trans)
    log_e = np.log(np.clip(emission, 1e-9, None))

    # per-step log evidence: either the classifier's own posterior, or the
    # emission column for the hard label it predicted
    obs = log_proba if log_proba is not None else log_e[:, labels].T   # (n, C)

    delta = np.full((n, N_CLASSES), -np.inf)
    psi = np.zeros((n, N_CLASSES), dtype=np.int16)
    delta[0] = np.log(1.0 / N_CLASSES) + obs[0]
    for i in range(1, n):
        cand = delta[i - 1][:, None] + log_t          # (from, to)
        psi[i] = cand.argmax(axis=0)
        delta[i] = cand.max(axis=0) + obs[i]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(delta[-1].argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = psi[i, path[i]]
    return path


def apply_smoothing(df: pd.DataFrame, cfg: TimelineConfig) -> pd.DataFrame:
    """Smooth within contiguous stretches only - never across a long gap."""
    df = df.copy()
    gap = df["t_start"].diff().fillna(0.0) > cfg.max_gap_s
    df["run"] = gap.cumsum()

    out = np.empty(len(df), dtype=np.int8)
    for _, idx in df.groupby("run").groups.items():
        pos = df.index.get_indexer(idx)
        lab = df["y_pred"].to_numpy()[pos]
        if cfg.smooth == "mode":
            out[pos] = smooth_mode(lab, cfg.mode_k)
        elif cfg.smooth == "viterbi":
            if cfg.emission is None:
                raise ValueError("smooth='viterbi' needs cfg.emission "
                                 "(P(pred|true) estimated on validation data)")
            lp = None
            pcols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
            if pcols:
                lp = np.log(np.clip(df[pcols].to_numpy()[pos], 1e-9, None))
            out[pos] = smooth_viterbi(lab, cfg.emission, cfg.p_stay, lp)
        else:
            out[pos] = lab
    df["y_smooth"] = out
    return df


# ------------------------------------------------------- interval merging ----
def merge_intervals(df: pd.DataFrame, cfg: TimelineConfig,
                    label_col: str = "y_smooth") -> pd.DataFrame:
    """Collapse consecutive same-class segments into intervals.

    A run continues across a gap of at most ``max_gap_s`` - the recording is
    duty-cycled, so a one-minute gap between two walking windows is a pause in
    *sampling*, not in walking.
    """
    lab = df[label_col].to_numpy()
    t_s = df["t_start"].to_numpy()
    t_e = df["t_end"].to_numpy()

    new = np.empty(len(df), dtype=bool)
    new[0] = True
    new[1:] = (lab[1:] != lab[:-1]) | ((t_s[1:] - t_e[:-1]) > cfg.max_gap_s)
    gid = np.cumsum(new) - 1

    df = df.assign(_g=gid)
    rows = []
    for g, sub in df.groupby("_g", sort=True):
        i0 = int(sub.index[0])
        rows.append({
            "label": int(sub[label_col].iloc[0]),
            "activity": CLASS_NAMES[int(sub[label_col].iloc[0])],
            "start_s": float(sub["t_start"].iloc[0]),
            "end_s": float(sub["t_end"].iloc[-1]),
            "duration_s": float(sub["t_end"].iloc[-1] - sub["t_start"].iloc[0]),
            "observed_s": float(len(sub) * SEG_SECONDS),
            "n_segments": int(len(sub)),
            "row_lo": i0,
            "row_hi": int(sub.index[-1]) + 1,
            "confidence": float(sub["confidence"].mean()) if "confidence" in sub else float("nan"),
        })
    iv = pd.DataFrame(rows)
    if cfg.min_interval_s > 0 and len(iv):
        iv = iv[iv["duration_s"] >= cfg.min_interval_s].reset_index(drop=True)
    return iv


# ------------------------------------------------------ per-interval evidence -
def _dominant_frequency(sig: np.ndarray) -> tuple[float, float]:
    """Dominant frequency in the gait band and its share of in-band power.

    ``sig`` is (n_segments, SEG_LEN). Each segment is de-meaned (removing the
    gravity/DC term), transformed, and the magnitude spectra are averaged.
    """
    if sig.size == 0:
        return float("nan"), float("nan")
    x = sig - sig.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(sig.shape[1], d=1.0 / SAMPLE_RATE)
    band = (freqs >= GAIT_BAND_HZ[0]) & (freqs <= GAIT_BAND_HZ[1])
    if not band.any() or spec[band].sum() <= 0:
        return float("nan"), float("nan")
    k = int(np.flatnonzero(band)[spec[band].argmax()])
    return float(freqs[k]), float(spec[k] / spec[band].sum())


def interval_evidence(X: np.ndarray, lo: int, hi: int) -> dict:
    """Signal features an explanation can cite, for segments X[lo:hi].

    These are the quantities the challenge's own worked examples appeal to:
    acceleration magnitude and its variance, step frequency, gyroscope
    oscillation amplitude, and the orientation of the gravity vector.
    """
    seg = np.asarray(X[lo:hi], dtype=np.float64)      # (n, SEG_LEN, 6)
    acc, gyr = seg[:, :, :3], seg[:, :, 3:]

    acc_mag = np.linalg.norm(acc, axis=2)             # (n, SEG_LEN)
    gyr_mag = np.linalg.norm(gyr, axis=2)

    gravity = acc.reshape(-1, 3).mean(axis=0)         # body-frame "down"
    g_norm = float(np.linalg.norm(gravity))
    dynamic = acc - acc.mean(axis=1, keepdims=True)   # gravity removed per segment

    step_hz, step_share = _dominant_frequency(acc_mag)
    per_channel_std = seg.reshape(-1, 6).std(axis=0)

    return {
        "modality": "Accelerometer, Gyroscope",
        "channels": "All",
        "acc_mag_mean": round(float(acc_mag.mean()), 4),
        "acc_mag_std": round(float(acc_mag.std()), 4),
        "dynamic_acc_std": round(float(dynamic.reshape(-1, 3).std()), 4),
        "gyro_mag_mean": round(float(gyr_mag.mean()), 4),
        "gyro_mag_std": round(float(gyr_mag.std()), 4),
        "step_freq_hz": None if np.isnan(step_hz) else round(step_hz, 3),
        "step_freq_share": None if np.isnan(step_share) else round(step_share, 3),
        "gravity_dir": [round(float(v / g_norm), 3) for v in gravity] if g_norm > 1e-6 else None,
        "gravity_mag_g": round(g_norm, 4),
        "dominant_channel": CHANNELS[int(per_channel_std.argmax())],
        "channel_std": {c: round(float(s), 4) for c, s in zip(CHANNELS, per_channel_std)},
    }


# ------------------------------------------------------------- the timeline --
def build_timeline(seg_df: pd.DataFrame, cfg: TimelineConfig,
                   X: np.ndarray | None = None, user: str = "",
                   with_evidence: bool = True) -> dict:
    """Full aggregation: smoothing -> intervals -> evidence -> aggregates."""
    sm = apply_smoothing(seg_df, cfg)
    iv = merge_intervals(sm, cfg)

    intervals = []
    for _, r in iv.iterrows():
        item = {
            "activity": r["activity"],
            "label": int(r["label"]),
            "start_s": round(r["start_s"], 1),
            "end_s": round(r["end_s"], 1),
            "duration_s": round(r["duration_s"], 1),
            "observed_s": round(r["observed_s"], 1),
            "n_segments": int(r["n_segments"]),
        }
        if not np.isnan(r["confidence"]):
            item["confidence"] = round(r["confidence"], 3)
        if with_evidence and X is not None:
            item["evidence"] = interval_evidence(X, int(r["row_lo"]), int(r["row_hi"]))
        intervals.append(item)

    totals, counts, observed = {}, {}, {}
    for it in intervals:
        a = it["activity"]
        totals[a] = round(totals.get(a, 0.0) + it["duration_s"], 1)
        observed[a] = round(observed.get(a, 0.0) + it["observed_s"], 1)
        counts[a] = counts.get(a, 0) + 1

    transitions = [{"from": intervals[i]["activity"], "to": intervals[i + 1]["activity"],
                    "at_s": intervals[i + 1]["start_s"]}
                   for i in range(len(intervals) - 1)]

    return {
        "user": user,
        "t0_epoch": int(seg_df.attrs.get("t0_epoch", 0)),
        "time_base": "seconds from the start of the recording",
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": SEG_SECONDS,
        "recording_span_s": round(float(seg_df["t_end"].max() - seg_df["t_start"].min()), 1),
        "n_segments": int(len(seg_df)),
        "config": {"smooth": cfg.smooth, "mode_k": cfg.mode_k, "p_stay": cfg.p_stay,
                   "max_gap_s": cfg.max_gap_s, "min_interval_s": cfg.min_interval_s},
        "intervals": intervals,
        "totals_s": totals,
        "observed_s": observed,
        "counts": counts,
        "transitions": transitions,
    }


def save_timeline(timeline: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(timeline, fh, indent=1)
