"""Feature extractor, copied VERBATIM from random_forest_model.ipynb (cell 2).

Imported by the robustness experiment, which must recompute features from
degraded signal. Kept byte-identical in logic to the notebook, and verified
against the cached features in ../rf_features before use.
"""
import numpy as np

SEG_LEN = 128
FS = 32.0
CHUNK = 40_000

# ---------------------------------------------------------------------------
#  Feature extraction: (N, 128, 6) -> (N, 213)
# ---------------------------------------------------------------------------
#  The 6 raw channels are first expanded to 10 signals. The four derived ones
#  are the point of the exercise:
#     |acc|, |gyro|        rotation-invariant - unchanged by how the phone sits
#     acc_vertical         acceleration along the estimated gravity direction
#     acc_horizontal       the component perpendicular to it
#  Gravity is estimated as the segment mean, which at 4 s is dominated by the
#  static component. Splitting motion into vertical and horizontal converts a
#  device-relative description into a world-relative one: walking oscillates
#  vertically, cycling mostly horizontally, and that holds across users in a way
#  raw acc_z does not.
# ---------------------------------------------------------------------------

SIGNAL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                "acc_mag", "gyro_mag", "acc_vert", "acc_horiz"]
FREQS = np.fft.rfftfreq(SEG_LEN, d=1.0 / FS)          # 65 bins, 0 .. 16 Hz
BANDS = [(0.0, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, 16.0)]
AC_LO, AC_HI = 8, 64                                   # lags: 4 Hz down to 0.5 Hz


def _signals(X):
    """(N,128,6) -> (N,128,10): raw channels plus the four derived signals."""
    acc, gyr = X[:, :, :3], X[:, :, 3:]
    am = np.linalg.norm(acc, axis=2)
    gm = np.linalg.norm(gyr, axis=2)
    g = acc.mean(axis=1, keepdims=True)                       # gravity estimate
    gn = g / np.maximum(np.linalg.norm(g, axis=2, keepdims=True), 1e-8)
    vert = (acc * gn).sum(axis=2)                             # along gravity
    horiz = np.linalg.norm(acc - vert[:, :, None] * gn, axis=2)
    return np.concatenate([X, am[:, :, None], gm[:, :, None],
                           vert[:, :, None], horiz[:, :, None]], axis=2), gn[:, 0, :]


def _feature_names():
    n = []
    for stat in ["mean", "std", "min", "max", "median", "p25", "p75", "rms", "mad",
                 "skew", "kurt", "zcr", "jerk_mean", "jerk_std",
                 "dom_freq", "dom_power", "spec_energy", "spec_entropy", "spec_centroid"]:
        n += [f"{s}_{stat}" for s in SIGNAL_NAMES]
    for s in ["acc_mag", "gyro_mag"]:
        n += [f"{s}_band{lo}-{hi}Hz" for lo, hi in BANDS]
        n += [f"{s}_ac_peak", f"{s}_ac_lag"]
    n += ["corr_acc_xy", "corr_acc_xz", "corr_acc_yz",
          "corr_gyr_xy", "corr_gyr_xz", "corr_gyr_yz", "corr_accmag_gyrmag"]
    n += ["grav_mag", "grav_dir_x", "grav_dir_y", "grav_dir_z"]
    return n


FEATURE_NAMES = _feature_names()


def extract_features(X):
    """X: (N, 128, 6) float32 -> (N, len(FEATURE_NAMES)) float32."""
    S, gdir = _signals(np.asarray(X, dtype=np.float32))        # (N,128,10)
    mu = S.mean(1)
    sd = S.std(1)
    dev = S - mu[:, None, :]

    # --- A: distribution, B: shape, C: dynamics -----------------------------
    diff = np.abs(np.diff(S, axis=1))
    sd_safe = np.maximum(sd, 1e-8)
    parts = [mu, sd, S.min(1), S.max(1), np.median(S, 1),
             np.percentile(S, 25, axis=1), np.percentile(S, 75, axis=1),
             np.sqrt((S ** 2).mean(1)), np.abs(dev).mean(1),
             (dev ** 3).mean(1) / sd_safe ** 3,                  # skewness
             (dev ** 4).mean(1) / sd_safe ** 4 - 3.0,            # excess kurtosis
             (np.diff(np.signbit(dev), axis=1) != 0).mean(1),    # zero crossings
             diff.mean(1), diff.std(1)]

    # --- D: spectral --------------------------------------------------------
    F = np.abs(np.fft.rfft(dev, axis=1))                        # (N,65,10)
    P = F ** 2
    P1 = P[:, 1:, :]                                            # drop DC
    k = P1.argmax(1)                                            # (N,10)
    tot = np.maximum(P1.sum(1), 1e-12)
    Pn = P1 / tot[:, None, :]
    parts += [FREQS[1:][k],                                     # dominant freq
              np.take_along_axis(P1, k[:, None, :], 1)[:, 0, :],  # its power
              tot,
              -(Pn * np.log(Pn + 1e-12)).sum(1),                # spectral entropy
              (FREQS[1:, None] * P1).sum(1) / tot]              # spectral centroid

    # --- E: band energy, F: periodicity (magnitudes only) -------------------
    for idx in (6, 7):                                          # acc_mag, gyro_mag
        p = P[:, :, idx]
        parts += [np.stack([p[:, (FREQS >= lo) & (FREQS < hi)].sum(1)
                            for lo, hi in BANDS], axis=1)]
        ac = np.fft.irfft(p, n=SEG_LEN, axis=1)                 # autocorrelation
        ac = ac / np.maximum(ac[:, :1], 1e-12)
        seg = ac[:, AC_LO:AC_HI]
        lag = seg.argmax(1)
        parts += [np.stack([seg.max(1), (lag + AC_LO).astype(np.float32)], axis=1)]

    # --- G: cross-axis correlation -----------------------------------------
    def corr(a, b):
        za, zb = S[:, :, a] - mu[:, None, a], S[:, :, b] - mu[:, None, b]
        return (za * zb).mean(1) / np.maximum(sd[:, a] * sd[:, b], 1e-8)
    parts += [np.stack([corr(0, 1), corr(0, 2), corr(1, 2),
                        corr(3, 4), corr(3, 5), corr(4, 5), corr(6, 7)], axis=1)]

    # --- H: orientation -----------------------------------------------------
    parts += [np.linalg.norm(mu[:, :3], axis=1, keepdims=True), gdir]

    out = np.concatenate([p if p.ndim == 2 else p[:, None] for p in parts], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_chunked(X, chunk=CHUNK):
    """Extract in batches - the (N,128,10) intermediate is 5 KB per segment."""
    return np.concatenate([extract_features(X[s:s + chunk])
                           for s in range(0, len(X), chunk)], axis=0)
