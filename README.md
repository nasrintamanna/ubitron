# UbiQ : Activity Question Answering from Wearable Signals

A system that answers plain-language questions about a person's day - *"How long
did the user walk?"*, *"Did she lie down for a prolonged period?"* - from the
accelerometer and gyroscope of a phone, and backs every answer with the stretch of
signal it rests on. Built on the ExtraSensory dataset for CS60055 (Ubiquitous
Computing), Hackathon Challenge 1: *Ask the Sensors*.

**Results at a glance** (5-fold subject-wise cross-validation, every user tested once):

| component | accuracy | macro-F1 |
|---|---|---|
| CNN classifier | 0.366 | 0.259 |
| Random Forest classifier | 0.473 | 0.334 |
| **Upgraded Random Forest** (+ time of day + tuned thresholds) | **0.660** | **0.438** |

| question answering (1,720 questions, all 56 users) | overall QA accuracy |
|---|---|
| previous classifier + Qwen2.5-3B | 45.2% |
| **upgraded classifier + Qwen2.5-3B** | **50.6%** |

---

## 1. Dataset pipeline

All data processing lives in [`data_processing.ipynb`](data_processing.ipynb)
(guide per cell in section 6).

```
raw_acc/ + proc_gyro/        60 + 57 users, sampling rate varies per minute (13.8-233.9 Hz)
   │  resample to 32 Hz
acc_32Hz/ + gyro_32Hz/       one CSV per minute
   │  put the accelerometer in g
   │  merge gyroscope onto the accelerometer clock
merged_acc_gyro/             56 users, 236,457,636 rows
   │  attach the 7 activity labels
labeled_acc_gyro/            288,340 labelled minutes
   │  cut into 4-second segments
segmented_4s/                2,551,686 segments of 128 samples x 6 channels
   │  subject-wise folds, balance and augment the training data
balanced_folds/              5 folds x (train / val / test)
```

- **Resampling to 32 Hz.** Minutes recorded below 32 Hz are linearly interpolated
  up. Minutes above it are low-pass filtered first (zero-phase Butterworth,
  14.4 Hz) so nothing aliases. Timestamp guards drop padding rows and split at
  impossible gaps.
- **Units.** 26 users logged acceleration in m/s² and 34 in g - exactly the
  Android/iPhone split. 25 users were converted to g. One user (`BEF6C611`) was
  dropped: its minutes disagree with each other on the unit, so no single factor
  fixes it.
- **Merging.** The gyroscope is interpolated onto the accelerometer's timestamps.
  Where it has no samples (10.9% of rows), values are extrapolated with a stable
  AR(16) model, which decays to the signal's mean instead of running away.
- **Labels.** Five activities come from `feature_labels`, two from
  `original_labels`. No minute carries more than one of the seven, so each gets a
  single label. Minutes with none of the seven are dropped.
- **Segments.** 128 samples = 4 s, stride 64. Evaluation uses the non-overlapping
  subset (1,333,415 segments).
- **Splits.** ExtraSensory's own subject-wise 5 folds, restricted to our 56 users,
  plus 8 validation users per fold that must include at least 2 Running and 2
  Bicycling users.
- **Balancing (training data only).** The majority classes are undersampled with
  equal per-user quotas. The minority classes are augmented with label-preserving
  transforms: rotation about the gravity axis, time warping, scaling of the
  dynamic component and jitter. The class ratio falls from 120:1 to 1.5:1.

| index | activity | source column | labelled minutes |
|---|---|---|---|
| 0 | Lying down | `label:LYING_DOWN` | 98,639 |
| 1 | Sitting | `label:SITTING` | 126,758 |
| 2 | Walking | `label:FIX_walking` | 21,300 |
| 3 | Running | `label:FIX_running` | 1,078 |
| 4 | Bicycling | `label:BICYCLING` | 4,741 |
| 5 | Standing in place | `original_label:STANDING_IN_PLACE` | 7,821 |
| 6 | Standing and moving | `original_label:STANDING_AND_MOVING` | 28,003 |

---

## 2. CNN classifier

[`cnn_model.ipynb`](cnn_model.ipynb) (PyTorch), one model per fold.

- **Input:** a 4 s segment as 6 channels x 128 samples, normalised with the
  fold's training statistics.
- **Network:** two parallel convolution branches - kernel 5 for fast transients,
  kernel 21 for a full gait cycle - joined, then two conv + max-pool blocks, global
  average pooling, and a dense head (dropout 0.4) with 7 outputs.
- **Training:** the balanced, augmented `balanced_folds` training set; class-weighted
  cross-entropy; Adam (lr 1e-3, weight decay 1e-4); early stopping on validation
  macro-F1 (patience 4).

**Result:** accuracy **0.366**, macro-F1 **0.259** pooled (per fold 0.368 ± 0.046
and 0.263 ± 0.030). Validation macro-F1 peaked after only 1-6 epochs: the network
fits the training users quickly but does not transfer to new ones. It was set aside
for the Random Forest.

---

## 3. Random Forest classifier

[`random_forest_model.ipynb`](random_forest_model.ipynb), one forest per fold.

- **Features:** each segment becomes **213 numbers**. The six raw channels are
  expanded to ten signals - adding acceleration and gyroscope magnitude (unaffected
  by how the phone is held) and the vertical and horizontal parts of acceleration.
  From these come distribution statistics, shape, jerk, spectral features
  (dominant frequency, entropy, centroid), band energies, periodicity,
  cross-axis correlations and the direction of gravity.
- **Training data:** `segmented_4s` at its natural distribution, capped at 120,000
  segments per class with equal per-user quotas, and **no augmentation** - trees
  would split on near-duplicate synthetic segments.
- **Model:** 200 trees, `min_samples_leaf=4`, `max_features="sqrt"`,
  `class_weight="balanced"`. Validation and test come from `balanced_folds`.

**Result over all 7 activities** (pooled over 5 folds, 1,333,415 test segments):

| metric | pooled | per fold (mean ± std) |
|---|---|---|
| accuracy | **0.4733** | 0.4731 ± 0.0207 |
| macro-F1 | **0.3337** | 0.3286 ± 0.0305 |
| balanced accuracy | 0.3566 | 0.3651 ± 0.0283 |
| Cohen's kappa | 0.2096 | 0.2018 ± 0.0339 |

| activity | precision | recall | F1 |
|---|---|---|---|
| Lying down | 0.634 | 0.255 | 0.364 |
| Sitting | 0.520 | 0.713 | 0.601 |
| Walking | 0.372 | 0.580 | 0.454 |
| Running | 0.129 | 0.090 | 0.106 |
| Bicycling | 0.522 | 0.608 | 0.562 |
| Standing in place | 0.075 | 0.065 | 0.070 |
| Standing and moving | 0.176 | 0.185 | 0.180 |

The main weakness: **Lying down is mistaken for Sitting** - that single pair is
47.6% of all the model's errors. A still phone looks the same either way.

---

## 4. Upgraded Random Forest classifier

Developed in [`Try_increase_accuracy/`](Try_increase_accuracy/). The forest itself is
unchanged - same features, settings, data and folds. Two things changed, each chosen
on **validation** macro-F1 and only then reported on test.

**Step 1 - time of day as two extra features.** From the true labels, **74.3% of
lying-down minutes fall between 22:00 and 07:00, against 15.1% of sitting minutes**.
Each segment's minute id is a UTC epoch; it is converted to San Diego local time and
encoded as the sine and cosine of the hour (so 23:00 and 01:00 sit close together).
This takes the features from 213 to 215. The training segments' timestamps had not
been kept, so they were recovered by replaying the notebook's deterministic segment
selection, and checked against the cached labels.

**Step 2 - decision thresholds tuned for the real class balance.** The forest's
probabilities treat all classes as equally common, but test data is 44% Sitting and
0.4% Running. Each class's probability is multiplied by a weight before choosing the
largest, with the weights tuned on validation to maximise macro-F1. The five folds
agreed closely: Lying down x1.8-4.0, Walking x0.35, Bicycling x0.17.

**Also tried, rejected:** stronger regularisation (`min_samples_leaf` 20-100),
gradient boosting and a two-stage still/moving classifier all changed nothing.
Per-user normalisation *hurt* (macro-F1 0.262), because each user's average motion
reflects their lifestyle, not their sensor.

**Result over all 7 activities** (pooled over 5 folds):

| metric | Random Forest | **Upgraded** | change | per fold (mean ± std) |
|---|---|---|---|---|
| accuracy | 0.4733 | **0.6597** | +0.1864 | 0.6610 ± 0.0126 |
| macro-F1 | 0.3337 | **0.4382** | +0.1045 | 0.4449 ± 0.0385 |
| balanced accuracy | 0.3566 | **0.3979** | +0.0413 | 0.4193 ± 0.0394 |
| Cohen's kappa | 0.2096 | **0.4744** | +0.2648 | 0.4740 ± 0.0209 |

| activity | precision | recall | F1 | F1 before |
|---|---|---|---|---|
| Lying down | 0.796 | 0.724 | **0.758** | 0.364 |
| Sitting | 0.667 | 0.796 | **0.726** | 0.601 |
| Walking | 0.634 | 0.399 | **0.490** | 0.454 |
| Running | 0.310 | 0.128 | **0.181** | 0.106 |
| Bicycling | 0.913 | 0.446 | **0.600** | 0.562 |
| Standing in place | 0.078 | 0.082 | **0.080** | 0.070 |
| Standing and moving | 0.261 | 0.210 | **0.233** | 0.180 |

**Every class's F1 improved.** Lying-down recall nearly tripled. Walking and
Bicycling recall fell: the tuned weights make the model cautious about them, trading
recall for precision, which raises their F1.

> **Caveat:** most of the gain comes from time of day, which needs **wall-clock**
> timestamps. On a recording that only gives seconds from its start, the pipeline
> falls back to a model without it (RF + tuned thresholds alone: macro-F1 0.388).

---

## 5. From classifier to answers: the SLM pipeline

```
recording ─▶ classifier ─▶ activity timeline ──────────────┐
             (4 s segments)  intervals, totals, counts,     │
                             evidence features              ▼
question ─▶ Qwen2.5-3B: parse the question ─▶ Python: compute the answer ─▶ required
            into an intent                     from the timeline             output format
```

- **Timeline.** Per-segment predictions are smoothed (majority vote over 5
  segments), merged into activity intervals and summarised as totals, counts and
  transitions. An episode must last at least 60 s, so classifier blips cannot
  create activities. Every interval carries signal evidence: acceleration and
  gyroscope magnitude, step frequency and gravity direction.
- **The SLM parses, Python computes.** Qwen turns the question into a structured
  intent - identification, verification, duration, count, comparison, grounding,
  open-world, or unsupported. All numbers are computed in Python from the timeline,
  so a duration or count can never be invented. The full timeline (~171,000 tokens)
  would not fit Qwen's 32,768-token context anyway.
- **Explanations** cite only measured values. Questions the sensors cannot answer
  ("heart rate?") return N/A in every field rather than a guess.

**Evaluation, briefly.** On 1,720 questions over all 56 users, overall QA accuracy
(macro over the 7 question types) is **50.6%** with the upgraded classifier, against
45.2% with the previous one. Qwen parses 98.5% of questions correctly, and perfect
parsing would reach only 51.4% - so the remaining errors come from the classifier,
not the SLM. Duration and count answers are the weakest (about 10% each), because
the predicted timeline is more fragmented than the truth. All five figures the brief
requires are in [`Final_result/`](Final_result/README.md).

---

## 6. Guide to `data_processing.ipynb`

| cell | what it does | output |
|---|---|---|
| **1** | Defines the resampler: raw accelerometer and gyroscope minutes to 32 Hz, with anti-aliasing and timestamp guards | (functions only) |
| **2** | Runs the resampling for both sensors and checks every output is exactly 32 Hz | `acc_32Hz/`, `gyro_32Hz/` |
| **3** | Converts the accelerometer to g for users who logged in m/s², and flags users whose units are inconsistent | `acc_32Hz/` (in place) |
| **4** | Counts the resampled files per user and checks them against the raw folders | (report) |
| **5** | Merges the gyroscope onto the accelerometer's timestamps, with AR extrapolation where the gyroscope is missing | `merged_acc_gyro/` |
| **6** | Attaches the 7 activity labels from `feature_labels` and `original_labels`, dropping unlabelled minutes | `labeled_acc_gyro/` |
| **7** | Cuts every labelled minute into fixed 4 s segments (128 samples, 50% overlap) | `segmented_4s/` |
| **8** | Builds the train / validation / test user lists for the 5 folds | `updated_cv_5_folds/` |
| **9** | Builds each fold's arrays: undersamples and augments the training data, keeps val/test natural | `balanced_folds/` |
| **10** | Converts labels from 1-7 to 0-6 in `balanced_folds` | `balanced_folds/` |
| **11** | Scratch cell for inspecting an array | - |

> **One step is not in the notebook.** Cell 3 *flags* `BEF6C611` but does not
> remove it. It was removed afterwards with the command below. When rebuilding from
> scratch, run it **after cell 3 and before cell 5**, or the results will not match:
> ```bash
> rm -rf acc_32Hz/BEF6C611-50DA-4971-A040-87FB979F3FC1 gyro_32Hz/BEF6C611-50DA-4971-A040-87FB979F3FC1
> ```

---

## 7. Commands

Run everything from the project folder. Notebooks run from the terminal with
`jupyter execute`, which writes the results to files (use `--inplace` as well to keep
the printed tables inside the notebook).

### 7.1 Build the Random Forest and see its evaluation

```bash
jupyter execute random_forest_model.ipynb    # trains the 5 fold models -> rf_results/
jupyter execute rf_evaluation_plots.ipynb    # confusion matrix + per-class chart
```

The first run also extracts features into `rf_features/` (about 20 min); later runs
reuse them and take about 8 min.

- **Confusion matrix:** `xdg-open rf_results/confusion_matrix.png`
- **Per-class F1 / accuracy / balanced accuracy chart:** `xdg-open rf_results/per_class_metrics.png`
- **Numbers:** `rf_results/pooled_metrics.json` (pooled, per class) and
  `rf_results/folds.json` (per fold)

Or open the images in VS Code. To read the printed tables, run
`jupyter execute --inplace random_forest_model.ipynb` and open the notebook.

### 7.2 Build the upgraded Random Forest and see its evaluation

Needs `rf_features/` from 7.1.

```bash
cd Try_increase_accuracy
python3 reconstruct_wtr.py                 # recover the training timestamps
python3 run_experiments.py baseline time   # train both configurations, 5 folds
python3 tune_thresholds.py baseline time   # tune the decision thresholds on validation
python3 final_model.py                     # package the upgraded model and its metrics
cd ..
```

About 20 min in total. `final_model.py` prints the per-fold and pooled metrics and the
per-class comparison with the Random Forest.

- **Confusion matrix:** `xdg-open Try_increase_accuracy/final_model/confusion_matrix.png`
- **Confusion matrix with precision / recall / F1 beside it:** `xdg-open Final_result/figures/fig2_confusion_matrix.png`
- **Numbers:** `Try_increase_accuracy/final_model/metrics.json`
- **Every experiment compared:** `Try_increase_accuracy/results/summary.md` and
  `results/comparison.png` (to regenerate: run all experiments listed by
  `python3 run_experiments.py --list`, then `python3 summarize.py`)

For the single **deployable** upgraded model used on new recordings - trained once
on all 56 users - run (about 3 min each):

```bash
python3 upgraded_pipeline/train_final_model.py            # with time of day
python3 upgraded_pipeline/train_final_model.py --no-time  # fallback without clock time
```

### 7.3 Ask the SLM (Qwen2.5-3B)

```bash
# one question about a user
python3 upgraded_pipeline/ask_upgraded.py --user 00EABED2 "How long did the user walk?"

# an interactive session - type questions, blank line to quit
python3 upgraded_pipeline/ask_upgraded.py --user 00EABED2 -i

# a new raw recording and a file of questions (needs the models from 7.2)
python3 upgraded_pipeline/ask_upgraded.py --acc raw_acc/<uuid> --gyro proc_gyro/<uuid> \
        -q questions.txt -o answers.txt
```

`--user` takes any of the 56 user ids, or a unique prefix of one. Add `--no-slm` to
answer with keyword parsing only (no GPU), and `--stats` for latency and memory. The
first run downloads Qwen2.5-3B-Instruct (about 6 GB). The previous system is still
available as `python3 ask.py -u 00EABED2 "..."`. More in
[`upgraded_pipeline/README.md`](upgraded_pipeline/README.md).

---

## 8. Repository map

| folder / file | contents |
|---|---|
| `data_processing.ipynb` | the dataset pipeline (section 6) |
| `cnn_model.ipynb`, `cnn_results/` | CNN classifier and its results |
| `random_forest_model.ipynb`, `rf_results/` | Random Forest and its results |
| `rf_evaluation_plots.ipynb` | Random Forest confusion matrix and per-class chart |
| `Try_increase_accuracy/` | the experiments behind the upgraded classifier |
| `upgraded_pipeline/` | the complete upgraded system: ingest, classify, answer |
| `Final_result/` | the five required figures and all QA metrics |
| `activity_timeline.py` | predictions to activity timeline |
| `slm_query_engine.py` | Qwen question parsing, answer computation, output format |
| `ask.py`, `build_timelines.py`, `timelines/` | the previous system, kept unchanged |
| `qa_benchmark.py` | question generator and scorer used during development |

Environment: Python 3.11, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, PyTorch
2.5.1 (CUDA 12.1), transformers 5.16.1, bitsandbytes 0.50.2 (only for the quantised
models in `Final_result/`). Measured on an NVIDIA RTX A4500 (20 GB).
