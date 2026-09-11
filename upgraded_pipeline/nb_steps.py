"""Functions and constants extracted VERBATIM from data_processing.ipynb.

Generated mechanically with Python's ast module - never hand-edited - so the
pipeline resamples, checks units and merges sensors with exactly the logic that
built the training data. Only definitions are taken; the cells' whole-dataset
loops are left behind.

  cell 1  resample to 32 Hz (all of it)
  cell 3  unit-rule constants: G, BAND, CONSISTENCY_MIN
  cell 5  AR extrapolation + gyro_on_acc_grid
"""

# ======================================================================
# from data_processing.ipynb, cell 1
# ======================================================================
import os

import numpy as np

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.signal import butter, filtfilt

from tqdm.auto import tqdm

TARGET_HZ = 32.0

DT        = 1.0 / TARGET_HZ

SOURCES = [
    {"in_dir": "raw_acc",   "out_dir": "acc_32Hz"},
    {"in_dir": "proc_gyro", "out_dir": "gyro_32Hz"},
]

FIXED_WINDOW_SEC = None

ANTIALIAS   = True

AA_ORDER    = 4

AA_CUTOFF   = 0.9 * (TARGET_HZ / 2.0)

GAP_FACTOR  = 200

MIN_GAP_SEC = 5.0

MAX_OUT_SEC = 300.0

FLOAT_FMT   = "%.6f"

N_WORKERS   = max(1, min(24, (os.cpu_count() or 4)))

SKIP_EXISTING = True

def resample_file(in_path, out_path):
    """Read one raw .dat window, write it back out as a 32 Hz .csv.

    Returns (status, original_rate_hz, n_in, n_out).
    """
    try:
        raw = pd.read_csv(in_path, sep=r"\s+", header=None,
                          engine="c", dtype=np.float64).to_numpy()
    except Exception:
        return ("unreadable", np.nan, 0, 0)

    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 4:
        return ("too_short", np.nan, len(raw), 0)

    t, xyz = raw[:, 0], raw[:, 1:4]

    # clean: drop NaN/inf rows, sort by time, drop duplicate timestamps
    good = np.isfinite(t) & np.isfinite(xyz).all(axis=1) & (t > 0)
    t, xyz = t[good], xyz[good]
    order = np.argsort(t, kind="stable")
    t, xyz = t[order], xyz[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, xyz = t[keep], xyz[keep]
    if len(t) < 2:
        return ("too_short", np.nan, len(t), 0)

    # keep the longest continuous run, so one bad timestamp (a zero-filled
    # padding row, a clock reset) cannot stretch the window to decades
    status = "ok"
    steps = np.diff(t)
    gap_limit = max(GAP_FACTOR * np.median(steps), MIN_GAP_SEC)
    breaks = np.flatnonzero(steps > gap_limit)
    if breaks.size:
        starts = np.concatenate(([0], breaks + 1))
        ends   = np.concatenate((breaks + 1, [len(t)]))
        longest = int(np.argmax(ends - starts))
        lo, hi = starts[longest], ends[longest]
        if hi - lo < len(t):
            status = "ok_trimmed"
        t, xyz = t[lo:hi], xyz[lo:hi]
        if len(t) < 2:
            return ("too_short", np.nan, len(t), 0)

    duration = t[-1] - t[0]
    fs_orig  = (len(t) - 1) / duration          # actual rate of this file

    # uniform 32 Hz time grid, starting at the file's first timestamp
    if FIXED_WINDOW_SEC is not None:
        n_out = int(round(FIXED_WINDOW_SEC * TARGET_HZ))
    else:
        n_out = int(np.floor(duration / DT)) + 1
    if n_out > int(MAX_OUT_SEC * TARGET_HZ):
        return ("bad_timestamps", fs_orig, len(t), 0)
    t_new = t[0] + np.arange(n_out) * DT

    src_t, src_xyz = t, xyz

    # downsampling -> low-pass first so nothing above 16 Hz folds back in
    if ANTIALIAS and fs_orig > TARGET_HZ * 1.01:
        n_u  = int(np.floor(duration * fs_orig)) + 1
        t_u  = t[0] + np.arange(n_u) / fs_orig  # uniform grid at the ORIGINAL rate
        u    = np.column_stack([np.interp(t_u, t, xyz[:, k]) for k in range(3)])
        wn   = min(AA_CUTOFF / (fs_orig / 2.0), 0.99)
        b, a = butter(AA_ORDER, wn)
        if len(t_u) > 3 * max(len(a), len(b)):  # filtfilt needs enough padding
            u = filtfilt(b, a, u, axis=0)
        src_t, src_xyz = t_u, u

    # linear interpolation onto the 32 Hz grid (this is the up-sampling step
    # for every file recorded below 32 Hz, and the re-gridding step otherwise)
    out = np.column_stack([t_new] +
                          [np.interp(t_new, src_t, src_xyz[:, k]) for k in range(3)])

    pd.DataFrame(out, columns=["timestamp", "x", "y", "z"]).to_csv(
        out_path, index=False, float_format=FLOAT_FMT)
    return (status, fs_orig, len(t), n_out)

def resample_user(args):
    """Resample every window of one user. Runs in a worker process."""
    in_dir, out_dir, uuid = args
    src = os.path.join(in_dir, uuid)
    dst = os.path.join(out_dir, uuid)
    os.makedirs(dst, exist_ok=True)

    counts, rates = {}, []
    for name in sorted(os.listdir(src)):
        in_path = os.path.join(src, name)
        if not os.path.isfile(in_path):
            continue
        # "1444079161.m_raw_acc.dat" -> "1444079161.csv"
        out_path = os.path.join(dst, name.split(".")[0] + ".csv")
        if SKIP_EXISTING and os.path.exists(out_path):
            counts["already_done"] = counts.get("already_done", 0) + 1
            continue
        status, fs, _, _ = resample_file(in_path, out_path)
        counts[status] = counts.get(status, 0) + 1
        if status.startswith("ok"):
            rates.append(fs)
    return uuid, counts, rates

# ======================================================================
# from data_processing.ipynb, cell 3
# ======================================================================
import json

import random

import multiprocessing as mp

G         = 9.80665

BAND      = (0.5, 2.0)

CONSISTENCY_MIN = 0.80

# ======================================================================
# from data_processing.ipynb, cell 5
# ======================================================================
import json

import multiprocessing as mp

import numpy as np

import pandas as pd

import pyarrow as pa

import pyarrow.parquet as pq

AR_ORDER         = 16

AR_TRAIN_N       = 256

def ar_fit_yw(x, p):
    """AR(p) coefficients by Yule-Walker / Levinson-Durbin. Returns (coeffs, mean).

    Yule-Walker is used rather than least squares because it always yields a
    stable model, which is what keeps the forecast bounded.
    """
    x = np.asarray(x, float)
    mu = x.mean()
    xc = x - mu
    n = len(xc)
    p = min(p, n - 1)
    if p < 1:
        return np.zeros(0), mu
    r = np.correlate(xc, xc, mode="full")[n - 1:n + p] / n     # autocorrelation
    if r[0] <= 1e-30:                                          # constant signal
        return np.zeros(0), mu
    a = np.zeros(p + 1)
    E = r[0]
    for k in range(1, p + 1):
        acc = r[k] - (np.dot(a[1:k], r[k - 1:0:-1]) if k > 1 else 0.0)
        kap = acc / E
        new = a.copy()
        new[k] = kap
        if k > 1:
            new[1:k] = a[1:k] - kap * a[k - 1:0:-1]
        a = new
        E *= (1 - kap ** 2)
        if E <= 1e-30:
            break
    return a[1:], mu

def ar_forecast(x, p, n_ahead):
    """Iterate an AR(p) fitted on x forward n_ahead steps."""
    if n_ahead <= 0:
        return np.zeros(0)
    a, mu = ar_fit_yw(x, p)
    if len(a) == 0:
        return np.full(n_ahead, mu)                            # flat at the mean
    hist = list(np.asarray(x, float)[-len(a):] - mu)
    out = np.empty(n_ahead)
    for i in range(n_ahead):
        nxt = float(np.dot(a, hist[::-1]))
        out[i] = nxt
        hist.append(nxt)
        hist.pop(0)
    return out + mu

def gyro_on_acc_grid(tg, vg3, ta):
    """Gyro (3 axes) at accelerometer timestamps.

    Inside the gyro's range: linear interpolation. Outside: the gyro series is
    first extended on its own uniform grid by AR forecasting (backwards at the
    start, forwards at the end), then interpolated as normal.
    """
    dt = float(np.median(np.diff(tg))) if len(tg) > 1 else 1.0 / TARGET_HZ
    before = ta < tg[0]
    after = ta > tg[-1]
    n_b = int(np.ceil((tg[0] - ta[0]) / dt)) if before.any() else 0
    n_a = int(np.ceil((ta[-1] - tg[-1]) / dt)) if after.any() else 0

    t_ext = tg
    if n_b:
        t_ext = np.concatenate([tg[0] - dt * np.arange(n_b, 0, -1), t_ext])
    if n_a:
        t_ext = np.concatenate([t_ext, tg[-1] + dt * np.arange(1, n_a + 1)])

    out = {}
    for ax, v in vg3.items():
        head = ar_forecast(v[:AR_TRAIN_N][::-1], AR_ORDER, n_b)[::-1] if n_b else np.zeros(0)
        tail = ar_forecast(v[-AR_TRAIN_N:], AR_ORDER, n_a) if n_a else np.zeros(0)
        v_ext = np.concatenate([head, v, tail])
        out[f"gyro_{ax}"] = np.interp(ta, t_ext, v_ext)
    return out, (before | after)

