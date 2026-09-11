"""Raw recording -> fixed 4-second segments, using the notebook's own steps.

Input is one user's recording in ExtraSensory's raw layout: a folder of
accelerometer minute-files (<epoch>.m_raw_acc.dat) and a folder of gyroscope
minute-files (<epoch>.m_proc_gyro.dat). The file names are the wall-clock epoch
of each minute, which is what the time-of-day feature needs.

Steps, in the order data_processing.ipynb applied them to the training data:
  1. resample every file to 32 Hz            (cell 1, resample_file, unchanged)
  2. unit check on the accelerometer         (cell 3's rule and constants)
  3. merge gyroscope onto the acc clock      (cell 5, gyro_on_acc_grid, AR extrapolation)
  4. cut each minute into non-overlapping 128-sample segments (cell 7's layout)

Resampling goes through a temporary CSV, exactly as the notebook did, so the
6-decimal rounding of the stored training data is reproduced too.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import nb_steps as N

SEG_LEN = 128


def _resample(args):
    path, tmpdir = args
    out = os.path.join(tmpdir, f"{os.getpid()}_{Path(path).name}.csv")
    status, fs, n_in, n_out = N.resample_file(str(path), out)
    if not status.startswith("ok"):
        return Path(path).name.split(".")[0], status, None, None
    d = pd.read_csv(out)
    os.remove(out)
    return (Path(path).name.split(".")[0], status,
            d["timestamp"].to_numpy(), d[["x", "y", "z"]].to_numpy())


def _resample_all(folder, pattern, workers):
    files = sorted(Path(folder).glob(pattern))
    with tempfile.TemporaryDirectory() as tmp, \
            ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        res = list(pool.map(_resample, [(f, tmp) for f in files], chunksize=16))
    ok = {e: (t, v) for e, s, t, v in res if t is not None}
    bad = {s: sum(1 for _, s2, t, _ in res if t is None and s2 == s) for _, s, t, _ in res if t is None}
    return ok, len(files), bad


def unit_decision(acc: dict) -> tuple[str, float]:
    """Cell 3's rule: the unit is decided only when >= CONSISTENCY_MIN of the
    minutes agree. 'g' keeps the data, 'm/s^2' divides by G, anything else is
    'inconsistent' - the case that got one training user dropped."""
    med = np.array([np.median(np.linalg.norm(v, axis=1)) for _, v in acc.values()])
    lo, hi = N.BAND
    in_g = float(np.mean((med > lo) & (med < hi)))
    in_ms2 = float(np.mean((med > lo * N.G) & (med < hi * N.G)))
    if in_ms2 >= N.CONSISTENCY_MIN:
        return "m/s^2", in_ms2
    if in_g >= N.CONSISTENCY_MIN:
        return "g", in_g
    return "inconsistent", max(in_g, in_ms2)


def ingest(acc_dir, gyro_dir, workers: int = 16) -> dict:
    acc, n_acc, bad_acc = _resample_all(acc_dir, "*.dat", workers)
    gyr, n_gyr, bad_gyr = _resample_all(gyro_dir, "*.dat", workers)
    unit, agree = unit_decision(acc) if acc else ("none", 0.0)
    if unit == "m/s^2":                    # as cell 3: divide, then 6-decimal CSV rounding
        acc = {e: (t, np.round(v / N.G, 6)) for e, (t, v) in acc.items()}

    Xs, ws, ss = [], [], []
    no_gyro = too_short = 0
    for e in sorted(acc, key=int):
        ta, va = acc[e]
        if e not in gyr or len(gyr[e][0]) < 2:
            no_gyro += 1
            continue
        tg, vg = gyr[e]
        cols, _ = N.gyro_on_acc_grid(tg, {"x": vg[:, 0], "y": vg[:, 1], "z": vg[:, 2]}, ta)
        M = np.column_stack([va, cols["gyro_x"], cols["gyro_y"], cols["gyro_z"]])
        offs = np.arange(0, len(ta) - SEG_LEN + 1, SEG_LEN)
        if len(offs) == 0:
            too_short += 1
            continue
        Xs.append(np.stack([M[o:o + SEG_LEN] for o in offs]).astype(np.float32))
        ws.append(np.full(len(offs), int(e), np.int64))
        ss.append((offs // SEG_LEN).astype(np.int32))
    X = np.concatenate(Xs) if Xs else np.zeros((0, SEG_LEN, 6), np.float32)
    return {"X": X, "window": np.concatenate(ws) if ws else np.zeros(0, np.int64),
            "seg_index": np.concatenate(ss) if ss else np.zeros(0, np.int32),
            "report": {"acc_files": n_acc, "gyro_files": n_gyr,
                       "acc_unusable": bad_acc, "gyro_unusable": bad_gyr,
                       "unit": unit, "unit_agreement": round(agree, 3),
                       "windows_used": len(Xs), "windows_without_gyro": no_gyro,
                       "windows_too_short": too_short, "segments": int(len(X))}}
