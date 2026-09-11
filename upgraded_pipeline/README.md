# upgraded_pipeline — Ask the Sensors, with the upgraded classifier

A complete, separate pipeline that answers natural-language questions about a
sensor recording using the **upgraded recognition backbone** (Random Forest +
time-of-day features + validation-tuned class thresholds) and the same SLM query
engine (Qwen2.5-3B-Instruct). The original system — `../ask.py`,
`../build_timelines.py`, `../timelines/` — is unchanged and still works.

Measured in `../Final_result/`: overall QA accuracy **50.6%** with this backbone,
against 45.2% with the original one (same SLM, same 1,720 questions, all 56 users).

## First time after cloning

The two trained models are not in the repository — they exceed GitHub's 100 MB
file limit and are listed in `.gitignore`. Build them once (about 3 minutes each;
needs `segmented_4s/` from `data_processing.ipynb`):

```bash
python3 upgraded_pipeline/train_final_model.py            # rf_time_leaf4.joblib
python3 upgraded_pipeline/train_final_model.py --no-time  # rf_notime_leaf4.joblib
```

Only the new-recording path needs them; the known-user path works without them.

## Two ways to use it

**A known user (one of the 56).** Answers come from that user's *out-of-fold*
predictions — made by a model that never saw them — so they are honest:

```bash
python3 upgraded_pipeline/ask_upgraded.py --user 00EABED2 "How long did the user walk?"
python3 upgraded_pipeline/ask_upgraded.py --user 9DC38D04 -i          # interactive
```

**A new recording** — ExtraSensory's raw layout: a folder of accelerometer
minute-files (`<epoch>.m_raw_acc.dat`) and a folder of gyroscope minute-files
(`<epoch>.m_proc_gyro.dat`). This is the path for grading-time data:

```bash
python3 upgraded_pipeline/ask_upgraded.py \
    --acc  path/to/raw_acc/<uuid>  --gyro path/to/proc_gyro/<uuid> \
    -q questions.txt -o answers.txt
```

Useful options: `--no-slm` (keyword parsing, no GPU), `--no-explain`, `--stats`,
`--save-timeline file.json` (reuse with `--timeline file.json`),
`--clock auto|on|off` (time-of-day feature; `auto` decides from the file names).

Several questions can be passed at once — each one ending with `?` is answered
separately.

## What happens to a new recording

```
raw minute-files ─▶ ingest.py ─────────────────────────────▶ predict.py ────────────────────▶ SLM
                    1. resample to 32 Hz        (nb cell 1)   5. 213 features + time of day    parse question (Qwen)
                    2. accelerometer unit check (nb cell 3)   6. final RF → × tuned weights    resolve in Python
                    3. merge gyro onto acc clock (nb cell 5)  7. timeline: smooth, intervals,  write explanation
                    4. 128-sample segments       (nb cell 7)     evidence features
```

Steps 1–3 run the **notebook's own code**: `nb_steps.py` is extracted mechanically
from `data_processing.ipynb` with Python's `ast` module (definitions only, never
hand-edited). `features.py` is the RF notebook's extractor, verbatim.

## Verified

| check | result |
|---|---|
| ingestion vs the stored pipeline, a user logged in g (`00EABED2`) | all 10,604 segments found, max difference **0.00** |
| ingestion vs the stored pipeline, a user logged in m/s² (`86A4F379`) | unit detected (100% agreement), all 13,479 segments, max difference **0.00** |
| known-user timelines vs those evaluated in `Final_result/` | **identical** interval lists (checked on 2 users) |
| new-recording path on an unseen user (`BEF6C611`, never in any training) | runs end to end: 12,858 segments from 3,450 minutes, classified in 9.0 s; unit warning raised |

So a new recording is processed exactly the way the training data was.

## The models

| file | features | size | use |
|---|---|---|---|
| `models/rf_time_leaf4.joblib` | 213 + time of day | 371 MB | default, when minute ids are wall-clock epochs |
| `models/rf_notime_leaf4.joblib` | 213 | 398 MB | fallback when the recording has no clock time |

Both trained on **all 56 users** (597,849 segments) with the RF notebook's exact
training-set rule and hyper-parameters. Their class weights are the geometric mean
of the five weight vectors tuned on validation in `Try_increase_accuracy/` — the
folds agreed closely, so one averaged setting is stable. Metadata in
`models/*.json`. Retrain with `python3 train_final_model.py [--no-time]`.

## Know before using

- **Answers about the 56 known users use out-of-fold predictions, not the final
  model.** The final model was trained on them; asking it about them would be
  in-sample and look better than it really is.
- **Time of day needs wall-clock minute ids.** ExtraSensory file names are UTC
  epochs, so it works there. If a recording's names are not epochs, the pipeline
  switches to the fallback model automatically — and loses most of the accuracy gain.
- **Inconsistent accelerometer units** (neither g nor m/s² in ≥80% of minutes)
  produce a warning rather than a refusal — the case that got one training user
  dropped. Treat those answers with caution.
- **Minutes without a gyroscope file are skipped**; the report line says how many.
- **"N/A" for a time with no data is correct.** ExtraSensory is duty-cycled and has
  long gaps; asking about a moment in a gap returns N/A rather than a guess.

## Files

| file | purpose |
|---|---|
| `ask_upgraded.py` | the command-line tool |
| `ingest.py` | raw recording → segments |
| `nb_steps.py` | notebook functions, extracted verbatim (generated) |
| `features.py`, `timefeat.py` | feature extraction, time of day |
| `predict.py` | classification and timeline building, both paths |
| `train_final_model.py` | trains and saves the final models |
| `build_timelines.py` | timelines for the known users → `timelines/` |
| `models/` | the two trained models + metadata |
| `timelines/` | upgraded timelines for all 56 users |
