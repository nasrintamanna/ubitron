# QA accuracy by question type

Overall QA accuracy = fraction correct per question type, **macro-averaged over the 7 types** so the abundant sedentary questions do not dominate.

| question type | n | rule | final system | before upgrade | perfect parsing |
|---|---|---|---|---|---|
| Identification | 280 | exact match | 58.2% | 51.4% | 58.2% |
| Verification | 336 | exact match (yes/no) | 70.8% | 70.5% | 72.6% |
| Duration | 222 | within ±10% | 9.9% | 11.3% | 9.9% |
| Count | 222 | within ±1 | 10.4% | 5.0% | 10.4% |
| Comparison | 221 | exact match | 84.6% | 75.6% | 88.2% |
| Grounding | 215 | answer + IoU≥0.5 + modality/channels | 31.2% | 24.2% | 31.2% |
| Open-world | 224 | categorical match | 89.3% | 78.1% | 89.3% |
| **Overall (macro)** | 1720 | | **50.6%** | 45.2% | 51.4% |
| Overall (micro) | 1720 | | 52.3% | 47.2% | 53.1% |

## Categorical answers

| metric | identification |
|---|---|
| accuracy | 58.2% |
| macro-F1 | 0.350 |
| balanced accuracy | 0.338 |

Comparison accuracy 84.6%, open-world categorical accuracy 89.3%.

## Binary verification (positive class = Yes)

| accuracy | precision | recall | F1 | specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 70.8% | 0.807 | 0.548 | 0.652 | 0.869 | 92 | 22 | 76 | 146 |

## Numeric answers

| type | accuracy within tolerance | tolerance | MAE | MAPE | median abs. error |
|---|---|---|---|---|---|
| duration | 9.9% | ±10% | 18,250 s | 71.5% | 9,326 s |
| count | 10.4% | ±1 | 40.9 | — | 22 |

## Temporal answers and evidence grounding

| measure | value |
|---|---|
| grounding accuracy (answer + IoU≥0.5 + modality/channels) | 31.2% |
| grounding questions, answer alone correct | 84.2% |
| median IoU of cited onset interval | 0.000 |

**The price of demanding evidence** — answers whose reference has an interval to cite:

| type | n | answer correct | grounded & correct | gap | grounding precision |
|---|---|---|---|---|---|
| Identification | 280 | 58.2% | 42.5% | 15.7 pts | 82.6% |
| Verification | 168 | 54.8% | 29.8% | 25.0 pts | 83.7% |
| Duration | 222 | 9.9% | 4.5% | 5.4 pts | 57.7% |
| Grounding | 167 | 91.6% | 23.4% | 68.3 pts | 47.1% |
| Open-world | 127 | 87.4% | 48.8% | 38.6 pts | 92.5% |
| **all** | 964 | 56.1% | 29.0% | 27.1 pts | 68.8% |

## Question parsing (Qwen2.5-3B)

Parsed intent equals the question's true type for **98.5%** of questions.
- Identification: 100.0%
- Verification: 92.6%
- Duration: 100.0%
- Count: 100.0%
- Comparison: 100.0%
- Grounding: 100.0%
- Open-world: 100.0%
