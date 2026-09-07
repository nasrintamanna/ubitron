# ExtraSensory → 32 Hz Activity Recognition Dataset

Processing pipeline that turns the raw ExtraSensory accelerometer and gyroscope
recordings into fixed-length, labelled, class-balanced training sets for 7-class
human activity recognition.

All steps live in [`data_processing.ipynb`](data_processing.ipynb), cells 1–10.
Every cell is idempotent and safe to re-run.

---

## Result

| | |
|---|---|
| **Final artefact** | `balanced_folds/` — 5 cross-validation folds, ready to train |
| **Segments** | 2,551,686 fixed 4-second windows (128 samples @ 32 Hz, 6 channels) |
| **Users** | 56 |
| **Classes** | 7, indexed 0–6 |
| **Class balance** | 1.5 : 1 (from 120 : 1 in the source) |

### Activity classes

| index | activity | source column | windows |
|---|---|---|---|
| 0 | Lying down | `label:LYING_DOWN` | 98,639 |
| 1 | Sitting | `label:SITTING` | 126,758 |
| 2 | Walking | `label:FIX_walking` | 21,300 |
| 3 | Running | `label:FIX_running` | 1,078 |
| 4 | Bicycling | `label:BICYCLING` | 4,741 |
| 5 | Standing in place | `original_label:STANDING_IN_PLACE` | 7,821 |
| 6 | Standing and moving | `original_label:STANDING_AND_MOVING` | 28,003 |

Labels 1–7 in the parquet folders, 0–6 in `balanced_folds/*.npy`.
See `balanced_folds/label_map.json`.

---

## Pipeline

```
raw_acc/ + proc_gyro/          60 + 57 users, variable sampling rate
        │
        ▼  cells 1-2   resample to a uniform 32 Hz grid
acc_32Hz/ + gyro_32Hz/         60 + 57 users, 736,968 CSV files
        │
        ▼  cell 3      normalise accelerometer units to g
acc_32Hz/                      59 users (1 dropped)
        │
        ▼  cell 5      merge acc + gyro on the accelerometer clock
merged_acc_gyro/               56 users, 236,457,636 rows
        │
        ▼  cell 6      attach the 7 activity labels
labeled_acc_gyro/              56 users, 288,340 windows
        │
        ▼  cell 7      cut into fixed 4 s segments
segmented_4s/                  2,551,686 segments
        │
        ▼  cells 8-10  fold splits, undersample, augment, reindex
balanced_folds/                5 folds × (train / val / test)
```

### 1 · Resample to 32 Hz — cells 1–2

Source rates varied far more than the nominal figures: accelerometer
13.8–233.9 Hz (median 34.7), gyroscope 14.8–202.3 Hz (median 40.0).

- **Below 32 Hz** (9.4% of acc files, 2.4% of gyro) → linear interpolation up.
- **Above 32 Hz** → zero-phase 4th-order Butterworth low-pass at 14.4 Hz before
  resampling, so content above the new 16 Hz Nyquist cannot alias back in.
- Window length is preserved rather than forced, so no value is extrapolated.

**Timestamp guards.** Rows with `t <= 0` are dropped, the window is split at any
gap larger than both 200× the median sample interval and 5 s (keeping the
longest run), and a window whose span still exceeds 300 s is rejected. Without
these, one all-zero padding row in a file using epoch timestamps implied a
1.44-billion-second span and a 343 GiB allocation.

### 2 · Normalise accelerometer units — cell 3

ExtraSensory did not use one unit convention. **26 users logged m/s², 34 logged
g** — the split is exactly Android vs iPhone. Mixing them makes every
scale-sensitive feature ~9.8× larger for one group, which a model uses to
identify the *user* rather than the activity.

25 users were divided by 9.80665, in place, each file written to a temp and
moved with `os.replace` so an interruption cannot leave a partial CSV.

One user (`BEF6C611`) was **dropped**: only ~50% of its files agree on any single
scale, with per-file magnitudes running continuously from 0.21 to 19.0. No single
factor corrects it. Removed from both sensors; the raw source is untouched.

### 3 · Merge accelerometer + gyroscope — cell 5

Gyroscope is resampled onto the **accelerometer's timestamps**, which are left
unchanged.

The gyroscope covers only ~90.6% of the accelerometer's time span (49% for the
worst user), so ~10.9% of rows fall outside its recorded range and must be
extrapolated rather than interpolated.

**AR extrapolation.** An AR(16) is fitted to the nearest 256 gyro samples and
iterated forward on the gyroscope's own grid. Coefficients come from Yule-Walker
solved by Levinson-Durbin, which always yields a **stable** model — so the
forecast provably decays toward the signal's mean instead of diverging.

Measured against the linear extrapolation it replaced:

| \|gyro\| (rad/s) | measured | AR | linear |
|---|---|---|---|
| p99 | 2.831 | 0.859 | 19.513 |
| p99.9 | 6.007 | 3.372 | 67.810 |
| max | 21.95 | 32.57 | **203.17** |

Rows above 10 rad/s fell from 2.490% to 0.003% — below the measured rate of
0.008%. `merged_acc_gyro/` carries a `gyro_extrapolated` boolean marking these
rows.

418 accelerometer windows have no gyroscope counterpart and are skipped.

### 4 · Attach labels — cell 6

Labels are per-minute, keyed by the epoch `timestamp` that is also each window's
filename, so the join is direct.

**Zero windows carry more than one of the 7 labels** — verified across all
356,461 — so a single integer is unambiguous. The code raises if a multi-label
window ever appears rather than silently choosing one.

68,121 windows (19%) carry none of the 7 and are dropped.

### 5 · Fixed 4-second segments — cell 7

128 samples = 4.0 s at 32 Hz, stride 64 (50% overlap). Only 15 of 288,340
windows were too short. Tails that don't fill a segment are dropped, never
padded — padding recreates the flat-line artefact that distorts variance and
energy features.

`seg_start % 128 == 0` recovers the non-overlapping subset for evaluation, so no
regeneration is needed to change stride.

### 6 · Train / validation / test splits — cell 8

Built on ExtraSensory's own `cv_5_folds/`: subject-wise, platform-stratified,
every user in exactly one test fold.

- Fold lists intersected with the 56 available users. The 4 missing users are
  **all Android**, shifting platform balance from 26/34 to 22/34.
- 8 validation users carved from each training pool, **requiring ≥2 Running and
  ≥2 Bicycling users** — a random draw often contains zero Running, which makes
  early stopping on macro-F1 meaningless.
- Modest rare-class contributors are preferred for validation, keeping heavy
  ones in training.

Roughly 65 / 15 / 20 by users. Deterministic on `SEED = 1000`.

### 7 · Balance and augment — cells 9–10

**Training data only.** Validation and test keep the natural class distribution
and non-overlapping segments.

*Undersampling* — majority classes keep only non-overlapping segments, then an
equal per-user quota with water-filling redistribution caps heavy contributors.
Within a user, segments are picked spread across the session, not randomly.

*Augmentation* — all label-preserving:

1. **Rotation about the estimated gravity axis.** A free 3D rotation would move
   gravity in the sensor frame and can turn Sitting into something resembling
   Lying down while keeping the Sitting label. Rotating about gravity varies
   only heading. Verified: gravity magnitude unchanged to 4 decimals.
2. Small free rotation, ≤15°, for orientation tolerance.
3. Time warping — cadence variation, directly relevant to Running and Bicycling.
4. Scaling of the **dynamic component only** — scaling total acceleration would
   make gravity read something other than 1 g.
5. Jitter proportional to each channel's own standard deviation.

Target 50,000 per class, capped at 6× the real count. Running is the only class
that doesn't reach the target.

| index | activity | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| 0 | Lying down | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 |
| 1 | Sitting | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 |
| 2 | Walking | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 |
| 3 | Running | 33,576 | 44,256 | 33,018 | 29,514 | 42,636 |
| 4 | Bicycling | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 |
| 5 | Standing in place | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 |
| 6 | Standing and moving | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 |
| | **total** | 333,576 | 344,256 | 333,018 | 329,514 | 342,636 |

---

## Directory reference

| folder | size | contents |
|---|---|---|
| `raw_acc/`, `proc_gyro/` | 56 GB | untouched source |
| `acc_32Hz/`, `gyro_32Hz/` | 21 GB | 32 Hz per-minute CSVs, `timestamp,x,y,z` |
| `merged_acc_gyro/` | 9.6 GB | acc+gyro fused, with `gyro_extrapolated` |
| `labeled_acc_gyro/` | 7.5 GB | variable-length labelled windows |
| `segmented_4s/` | 7.2 GB | fixed 128-sample segments |
| `updated_cv_5_folds/` | 84 KB | train/val/test UUID lists + `splits.json` |
| `balanced_folds/` | 12 GB | **train from here** |

### `balanced_folds/fold_<i>/`

| file | shape | meaning |
|---|---|---|
| `X_train.npy` | (N, 128, 6) float32 | segments × timesteps × channels |
| `y_train.npy` | (N,) int8 | class 0–6 |
| `u_train.npy` | (N,) int16 | index into `report.json["users"]["train"]` |
| `aug_train.npy` | (N,) bool | `True` = synthetic |
| `X_val` / `X_test` + `y_`, `u_`, `w_` | | `w_` = parent window id |
| `norm_mean.npy`, `norm_std.npy` | (6,) | fitted on **real training segments only** |
| `class_weights.json`, `report.json` | | |

Channel order: `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`.
There is no `w_train` — window identity is unused during training and augmented
segments have no single parent.

---

## How to use it

`fold_<i>/` is **one complete experiment**, not a fifth of the data. Train the
same architecture five times from scratch:

```python
for i in range(5):
    model = build_model()                      # fresh weights each run
    model.fit(fold_i.train, val=fold_i.val)    # early stop on macro-F1
    preds[i] = model.predict(fold_i.X_test)
```

Each model predicts only its own test fold — the users it never saw. The five
test sets are disjoint and cover all 56 users exactly once, so **concatenate the
five prediction sets into one confusion matrix**. Do not average five fold
scores: fold 4's test set is 133k segments and fold 2's is 371k.

**Do not ensemble the five models.** Four of them trained on any given fold's
test users — averaging their predictions leaks.

For a deployable model afterwards, train once more on all 56 users. The
cross-validation estimate already tells you what to expect from it.

### Before the first run

1. **Apply normalisation** — `(X - norm_mean) / norm_std`, per fold. Not
   pre-applied, so the arrays stay traceable to `segmented_4s`.
2. **Class weights** — from `class_weights.json`. Running is the only class
   lifted (1.11–1.60 across folds).
3. **Load with `mmap_mode="r"`** — `X_train` is ~1 GB per fold.
4. **Metric: macro-F1 or balanced accuracy.** Validation and test are at the
   natural 120:1 distribution; a model predicting only Sitting and Lying down
   scores 79% accuracy while being useless.
5. **Fix torch/NumPy.** torch 2.2.0 is compiled against NumPy 1.x and
   `torch.from_numpy` fails on NumPy 2.4.6 — every path from `.npy` into PyTorch
   is blocked. Upgrade torch to 2.3+. TensorFlow 2.21 is unaffected.

---

## Known limitations

**Running concentrates in individuals.** Only 25 of 56 users ever ran, and one
supplies 36.9% of fold 0's training Running segments. Undersampling quotas
cannot flatten a class with no surplus to trim. Report per-user metric
distributions alongside the pooled number.

**Rare classes are thin per fold.** Bicycling appears in 23 of 56 users. Test
folds hold 522–4,014 Running segments — an 8× spread — so single-fold rare-class
numbers are noise. With ~5 test users per fold, the honest confidence interval on
Running recall is roughly ±43 points; pooled across folds, ±19.

**10.9% of gyroscope values are AR forecasts**, and the `gyro_extrapolated` flag
was not carried past `merged_acc_gyro/`. Augmenting a rare-class segment whose
gyro is partly synthetic multiplies a model artefact. Recovering the flag means
rebuilding from cell 6 onward.

**Augmentation preserves rare-class user skew.** Copies are drawn round-robin,
which replicates the existing per-user distribution rather than flattening it.

**Three users have accelerometer but no gyroscope** (`61359772`, `CCAF77F0`,
`F50235E0`). They remain in `acc_32Hz/` and are usable for an
accelerometer-only baseline with all 12 test users per fold.

---

## Environment

```
numpy 2.4.6 · pandas 3.0.5 · scipy 1.17.1 · pyarrow 25.0.1 · Python 3.11.5
```

Cells parallelise across 24 workers. Full rebuild is roughly 1 hour;
`balanced_folds` alone is ~25 minutes.
