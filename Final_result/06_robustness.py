"""Figure 5 - robustness: degrade the raw signal, rerun the WHOLE pipeline.

Scope: fold 0 (12 held-out users, 241,576 test segments) and its fixed question
set. For every degradation level the raw (128, 6) segments are corrupted, the
213 features are re-extracted with the verbatim notebook extractor, the improved
Random Forest re-predicts (time-of-day and tuned thresholds included), timelines
are rebuilt and the same questions re-answered. Question parsing is text-only and
does not see the signal, so the cached Qwen2.5-3B intents are reused.

Degradations (each on its own, the others off):
  noise     additive Gaussian with ABSOLUTE sigma, in g for the accelerometer and
            rad/s for the gyroscope. (A first attempt scaled sigma by each channel's
            global std; that std is dominated by phone orientation, ~0.3-0.6 g,
            while the motion inside a segment is ~0.001-0.002 g, so even the
            mildest level swamped the signal. Absolute units are the honest scale:
            a phone accelerometer's own noise is around 0.001 g.)
  dropout   k% of samples removed at random in every segment, gaps re-filled by
            linear interpolation (what a system does with missing data)
  rate      signal resampled to r Hz and back to 32 Hz by linear interpolation,
            i.e. a slower sensor; no anti-aliasing, as a cheap sensor would not
"""
import json
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import fr_common as C
import features as F

S = C.S
TIA = C.ROOT / "Try_increase_accuracy"
sys.path.insert(0, str(TIA))
from common import RF_PARAMS, hour_features       # noqa: E402  read-only import

FOLD = 0
LEVELS = {"noise": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
          "dropout": [0, 10, 25, 50, 75, 90],
          "rate": [32, 25, 20, 15, 10, 5]}

# ---- the final model for this fold, retrained deterministically -------------
z = np.load(C.ROOT / f"rf_features/fold_{FOLD}.npz")
wtr = np.load(TIA / f"cache/wtr_fold_{FOLD}.npy")
t0 = time.time()
rf = RandomForestClassifier(**RF_PARAMS).fit(np.hstack([z["Xtr"], hour_features(wtr)]), z["ytr"])
print(f"retrained improved RF for fold {FOLD} in {time.time() - t0:.0f}s", flush=True)
W = np.asarray(json.load(open(TIA / "results/time__thresholds.json"))["weights_per_fold"][FOLD])
wte, yte = z["wte"], z["yte"]
H = hour_features(wte)

# must reproduce the saved probabilities of the chosen configuration exactly
ref = np.load(TIA / f"cache/proba/time_fold_{FOLD}.npz")["Pte"]
P0 = np.zeros((len(yte), 7), np.float32); P0[:, rf.classes_] = rf.predict_proba(np.hstack([z["Xte"], H]))
assert np.allclose(P0, ref, atol=1e-6), "retrained model does not reproduce the final model"
print("  reproduces the final model's test probabilities exactly", flush=True)

X0 = np.load(C.ROOT / f"balanced_folds/fold_{FOLD}/X_test.npy")
CH_SD = X0.reshape(-1, 6).std(0)

ev = pickle.load(open(C.CACHE / "eval_set.pkl", "rb"))
QS_all, TL = ev["questions"], ev["timelines"]
intents_all = json.load(open(C.CACHE / "parse_qwen3b.json"))["intents"]
ut_all = C.user_tables()
users = set(C.fold_test_users(FOLD))
sel = [i for i, q in enumerate(QS_all) if q["user"] in users]
QS, INT = [QS_all[i] for i in sel], [intents_all[i] for i in sel]
print(f"  fixed question set: {len(QS)} questions over {len(users)} users", flush=True)


def degrade(X, kind, lvl, rng):
    if kind == "noise":
        return X + rng.normal(0, 1, X.shape).astype(np.float32) * np.float32(lvl)
    t = np.arange(X.shape[1], dtype=np.float64)
    out = np.empty_like(X)
    if kind == "dropout":
        if lvl == 0:
            return X.copy()
        n_keep = max(2, int(round(X.shape[1] * (1 - lvl / 100))))
        for i in range(len(X)):
            keep = np.sort(np.r_[0, X.shape[1] - 1,
                                 rng.choice(np.arange(1, X.shape[1] - 1), n_keep - 2, replace=False)])
            for c in range(6):
                out[i, :, c] = np.interp(t, keep, X[i, keep, c])
        return out
    if lvl >= 32:                                     # rate
        return X.copy()
    tr = np.arange(0, X.shape[1], 32.0 / lvl)          # sample times at r Hz
    for c in range(6):
        low = np.array([np.interp(tr, t, X[i, :, c]) for i in range(len(X))])
        out[:, :, c] = np.array([np.interp(t, tr, row) for row in low])
    return out


KINDS = [k for k in sys.argv[1:] if k in LEVELS] or list(LEVELS)
out_path = C.TAB / "robustness.json"
results = json.load(open(out_path)) if out_path.exists() else {}
for kind in KINDS:
    levels = LEVELS[kind]
    results[kind] = []
    for lvl in levels:
        t0 = time.time()
        rng = np.random.default_rng(12345)
        Xd = degrade(X0, kind, lvl, rng)
        Fd = np.hstack([F.extract_chunked(Xd), H])
        P = np.zeros((len(yte), 7), np.float32); P[:, rf.classes_] = rf.predict_proba(Fd)
        yp = (P * W).argmax(1)
        cls = C.S  # noqa
        from sklearn.metrics import f1_score
        acc = float((yp == yte).mean())
        mf1 = float(f1_score(yte, yp, average="macro", labels=range(7), zero_division=0))
        tls = {}
        for uid in users:
            ut = ut_all[uid]
            tls[uid] = C.pred_timeline(ut, uid, yp[ut["rows"]])
        rows = [C.score_one(q, S.resolve(it, tls[q["user"]]), None) for q, it in zip(QS, INT)]
        agg = C.aggregate(rows)
        results[kind].append({"level": lvl, "clf_accuracy": acc, "clf_macro_f1": mf1,
                              "qa_macro": agg["macro"], "qa_per_type": agg["per_type"]})
        print(f"  {kind:8s} {lvl:>6}  classifier acc {acc:.4f} F1 {mf1:.4f} | "
              f"QA macro {agg['macro']:.4f}   ({time.time() - t0:.0f}s)", flush=True)
        del Xd, Fd
json.dump(results, open(out_path, "w"), indent=1)
print("saved tables/robustness.json")
