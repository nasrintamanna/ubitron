"""Step 1 - ground truth, questions, and both backbones' timelines for all 56 users.

Writes cache/eval_set.pkl and tables/questions.json. Every user contributes
questions of all seven types; answers come from the ground-truth timeline.
"""
import json
import pickle
import random
from collections import Counter

import numpy as np

import fr_common as C

ut_all = C.user_tables(rebuild=True)
X = {f: np.load(C.ROOT / f"balanced_folds/fold_{f}/X_test.npy", mmap_mode="r") for f in range(5)}
QS, TL = [], {}
for n, (uid, ut) in enumerate(sorted(ut_all.items())):
    tl_true, df = C.truth_timeline(ut, uid)
    rng = random.Random(1000 + n)
    QS += C.generate(tl_true, uid, rng)
    TL[uid] = {"truth": tl_true, "df": df,
               "improved": C.pred_timeline(ut, uid, ut["pred"]["improved"], X[ut["fold"]][ut["rows"]]),
               "submitted": C.pred_timeline(ut, uid, ut["pred"]["submitted"])}
    print(f"  {n + 1:2d}/56 {uid[:8]}  truth {len(tl_true['intervals']):4d} episodes | "
          f"improved {len(TL[uid]['improved']['intervals']):4d} | "
          f"submitted {len(TL[uid]['submitted']['intervals']):4d}", flush=True)

pickle.dump({"questions": QS, "timelines": TL}, open(C.CACHE / "eval_set.pkl", "wb"))
json.dump(QS, open(C.TAB / "questions.json", "w"), indent=1)
c = Counter(q["type"] for q in QS)
print(f"\n{len(QS)} questions over {len(TL)} users:")
for t in C.TYPES:
    yn = Counter(q["gt"]["answer"] for q in QS if q["type"] == t and q["gt"]["answer"] in ("Yes", "No"))
    print(f"  {t:15s} {c[t]:5d}" + (f"   (Yes {yn['Yes']}, No {yn['No']})" if yn else ""))
