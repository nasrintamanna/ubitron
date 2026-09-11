"""Step 2 - parse every question with one SLM configuration, and measure its cost.

  python3 02_parse.py qwen3b              # full fp16 3B (the deployed system)
  python3 02_parse.py qwen3b_8bit qwen3b_4bit qwen1.5b qwen0.5b rules

Parsing is batched for throughput, using the deployed engine's own prompts and
post-processing, so the intents are exactly what ask.py would produce. Cost is
measured separately, ONE query at a time through the full answer_query path
(parse + resolve + explanation), because the brief asks for single-query cost.
"""
import json
import pickle
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

import fr_common as C

S = C.S
CONFIGS = {
    "rules":       (None, None),
    "qwen0.5b":    ("Qwen/Qwen2.5-0.5B-Instruct", None),
    "qwen1.5b":    ("Qwen/Qwen2.5-1.5B-Instruct", None),
    "qwen3b":      ("Qwen/Qwen2.5-3B-Instruct", None),
    "qwen3b_8bit": ("Qwen/Qwen2.5-3B-Instruct", "8bit"),
    "qwen3b_4bit": ("Qwen/Qwen2.5-3B-Instruct", "4bit"),
}
BATCH, N_LATENCY = 32, 40


def gpu_power_sampler(stop, out):
    while not stop.is_set():
        try:
            w = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2).stdout.strip()
            out.append(float(w.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.1)


def batch_parse(eng, queries):
    tok, model = eng.tok, eng.model
    tok.padding_side = "left"
    raws = []
    for s in range(0, len(queries), BATCH):
        chunk = queries[s:s + BATCH]
        texts = []
        for q in chunk:
            msgs = [{"role": "system", "content": S.PARSE_SYSTEM}]
            for u, a in S.PARSE_EXAMPLES:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": q})
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            raws.append(tok.decode(row, skip_special_tokens=True).strip())
        print(f"    parsed {min(s + BATCH, len(queries)):5d}/{len(queries)}", flush=True)
    return raws


def run(name):
    model_id, quant = CONFIGS[name]
    ev = pickle.load(open(C.CACHE / "eval_set.pkl", "rb"))
    QS, TL = ev["questions"], ev["timelines"]
    queries = [q["query"] for q in QS]
    print(f"\n=== {name} ===", flush=True)
    stats = {"config": name, "model_id": model_id, "quantization": quant}

    if model_id is None:
        eng = None
        intents = [S.normalise_intent("", q) for q in queries]
        stats.update(params=0, footprint_mb=0.0, disk_mb=0.0)
    else:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        eng = S.QwenEngine(model_id=model_id, quantization=quant)
        stats["load_s"] = time.perf_counter() - t0
        stats["params"] = int(sum(p.numel() for p in eng.model.parameters()))
        stats["footprint_mb"] = eng.model.get_memory_footprint() / 2**20
        from huggingface_hub import snapshot_download
        import os
        snap = snapshot_download(model_id, allow_patterns=["*.safetensors"])
        stats["disk_mb"] = (sum(os.path.getsize(os.path.join(snap, f)) for f in os.listdir(snap)
                                if f.endswith(".safetensors")) / 2**20
                            if quant is None else stats["footprint_mb"])
        t0 = time.perf_counter()
        raws = batch_parse(eng, queries)
        stats["batch_parse_s"] = time.perf_counter() - t0
        intents = [S.normalise_intent(r, q) for r, q in zip(raws, queries)]

    # ---- single-query cost, end to end (parse + resolve + explanation) ----
    rng = np.random.default_rng(0)
    sample = rng.choice(len(QS), N_LATENCY, replace=False)
    if eng is not None:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    lat, power, stop = [], [], threading.Event()
    th = threading.Thread(target=gpu_power_sampler, args=(stop, power)) if eng else None
    if th:
        th.start()
    for i in sample:
        q = QS[i]
        t0 = time.perf_counter()
        S.answer_query(q["query"], TL[q["user"]]["improved"], eng, explain=True)
        if eng is not None:
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    if th:
        stop.set(); th.join()
    stats["latency_median_s"] = statistics.median(lat)
    stats["latency_p90_s"] = float(np.percentile(lat, 90))
    stats["peak_vram_mb"] = (torch.cuda.max_memory_allocated() / 2**20) if eng else 0.0
    stats["gpu_power_w"] = float(np.mean(power)) if power else 0.0
    stats["energy_j_per_query"] = stats["gpu_power_w"] * stats["latency_median_s"]

    json.dump({"stats": stats, "intents": intents}, open(C.CACHE / f"parse_{name}.json", "w"))
    agree = None
    print(f"  params {stats['params'] / 1e6:,.0f}M | footprint {stats['footprint_mb']:,.0f} MB | "
          f"latency {stats['latency_median_s'] * 1000:,.0f} ms | peak VRAM "
          f"{stats['peak_vram_mb']:,.0f} MB | ~{stats['energy_j_per_query']:.1f} J/query", flush=True)
    del eng
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for n in sys.argv[1:] or ["qwen3b"]:
        run(n)
