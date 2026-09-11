# Accuracy versus overhead

Same 1,720 questions, same improved backbone; only the question-parsing SLM changes. Measured one query at a time on an NVIDIA RTX A4500 (20 GB), full answer path (parse + resolve + explanation).

| configuration | params | size (MB) | latency (ms, median) | peak VRAM (MB) | energy (J/query) | intent parsed correctly | overall QA accuracy |
|---|---|---|---|---|---|---|---|
| Rules only (no SLM) | 0M | 0 | 0 | 0 | 0.0 | 95.1% | **48.7%** |
| Qwen2.5-0.5B fp16 | 494M | 942 | 187 | 979 | 29.9 | 71.2% | **39.9%** |
| Qwen2.5-1.5B fp16 | 1,544M | 2,944 | 479 | 3,006 | 93.0 | 80.5% | **39.9%** |
| Qwen2.5-3B 4-bit | 3,086M | 1,917 | 840 | 2,202 | 155.7 | 73.3% | **39.0%** |
| Qwen2.5-3B 8-bit | 3,086M | 3,240 | 3,181 | 3,365 | 404.5 | 98.3% | **50.5%** |
| Qwen2.5-3B fp16 | 3,086M | 5,886 | 848 | 5,961 | 167.3 | 98.5% | **50.6%** |
