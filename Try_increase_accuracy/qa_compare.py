"""Does the better classifier give better ANSWERS? Same questions, two timelines.

Questions are generated from the ground-truth timelines of fold 0's test users
(identical seeds for both runs), then answered once from timelines built on the
submitted RF's predictions and once from the improved model's. The parent
project's modules are imported read-only; nothing outside this folder is written.
"""
import json
import sys

import numpy as np
import pandas as pd

from common import HERE, ROOT

sys.path.insert(0, str(ROOT))
import activity_timeline as T          # noqa: E402  (read-only import)
import qa_benchmark as QB              # noqa: E402

FOLD = 0
users = json.load(open(ROOT / f"balanced_folds/fold_{FOLD}/report.json"))["users"]["test"]
u = np.load(ROOT / f"balanced_folds/fold_{FOLD}/u_test.npy")
w = np.load(ROOT / f"balanced_folds/fold_{FOLD}/w_test.npy")
orig = np.load(ROOT / "rf_results/predictions.npz", allow_pickle=True)
impr = np.load(HERE / "final_model/predictions.npz")
yt = orig[f"y_true_{FOLD}"]
assert np.array_equal(yt, impr[f"y_true_{FOLD}"]), "test labels differ between the two runs"

segidx = []
for uid in users:
    ss = pd.read_parquet(ROOT / f"segmented_4s/{uid}.parquet",
                         columns=["seg_start"])["seg_start"].to_numpy()[::128]
    segidx.append(ss[ss % 128 == 0] // 128)
segidx = np.concatenate(segidx)
cfg = T.TimelineConfig()

scores = {}
for label, yp in (("submitted RF", orig[f"y_pred_{FOLD}"]), ("improved", impr[f"y_pred_{FOLD}"])):
    QS, ANS = [], []
    for j, uid in enumerate(users):
        m = u == j
        df = T.build_segment_table(w[m], segidx[m], yp[m], yt[m])
        tl_true = T.build_timeline(df.assign(y_pred=df["y_true"]), cfg, user=uid, with_evidence=False)
        tl_pred = T.build_timeline(df, cfg, user=uid, with_evidence=False)
        qs = QB.generate_questions(tl_true, n_per_type=6, seed=j)
        QS += qs; ANS += [QB.answer_from_timeline(q, tl_pred) for q in qs]
    scores[label] = QB.score(QS, ANS)

a, b = scores["submitted RF"], scores["improved"]
print(f"fold {FOLD} test users, {a['n']} questions, answered from each model's timeline\n")
print(f"  {'question type':16s} {'submitted RF':>13s} {'improved':>10s} {'change':>8s}")
for t in a["per_type"]:
    x, y = a["per_type"][t]["accuracy"], b["per_type"][t]["accuracy"]
    print(f"  {t:16s} {100 * x:12.1f}% {100 * y:9.1f}% {100 * (y - x):+7.1f}")
print(f"  {'MACRO':16s} {100 * a['macro_accuracy']:12.1f}% {100 * b['macro_accuracy']:9.1f}% "
      f"{100 * (b['macro_accuracy'] - a['macro_accuracy']):+7.1f}")
json.dump({k: {"macro": v["macro_accuracy"], "per_type": v["per_type"]} for k, v in scores.items()},
          open(HERE / "results/qa_compare_fold0.json", "w"), indent=1)
