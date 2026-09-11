#!/usr/bin/env python3
"""Ask the Sensors - UPGRADED pipeline (RF + time of day + tuned thresholds, then Qwen).

The original ../ask.py is unchanged; this is a separate pipeline.

  # one of the 56 known users (answers from the upgraded out-of-fold timeline)
  python3 upgraded_pipeline/ask_upgraded.py --user 00EABED2 "How long did she walk?"

  # a NEW raw recording: folders of ExtraSensory minute-files
  python3 upgraded_pipeline/ask_upgraded.py --acc path/to/raw_acc/<uuid> \\
          --gyro path/to/proc_gyro/<uuid> -q questions.txt -o answers.txt

  # interactive, keyword parsing only (no GPU), print resource use
  python3 upgraded_pipeline/ask_upgraded.py --user 9DC38D04 -i --no-slm --stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict                                            # noqa: E402

S = None


def main():
    global S
    p = argparse.ArgumentParser(description="Upgraded Ask-the-Sensors pipeline",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("question", nargs="*")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--user", "-u", help="one of the 56 known users (UUID prefix)")
    src.add_argument("--acc", help="folder of raw accelerometer minute-files (new recording)")
    src.add_argument("--timeline", "-t", help="an already-built timeline JSON")
    p.add_argument("--gyro", help="folder of gyroscope minute-files (with --acc)")
    p.add_argument("--clock", choices=["auto", "on", "off"], default="auto",
                   help="time-of-day feature: auto uses it when minute ids are wall-clock epochs")
    p.add_argument("--questions", "-q"); p.add_argument("--out", "-o")
    p.add_argument("--interactive", "-i", action="store_true")
    p.add_argument("--no-slm", action="store_true"); p.add_argument("--no-explain", action="store_true")
    p.add_argument("--save-timeline", help="write the built timeline here for reuse")
    p.add_argument("--stats", action="store_true")
    a = p.parse_args()
    if a.acc and not a.gyro:
        p.error("--acc needs --gyro")

    t0 = time.perf_counter()
    if a.timeline:
        tl = json.load(open(a.timeline))
    elif a.user:
        hits = sorted((HERE / "timelines").glob(f"{a.user}*.json"))
        tl = json.load(open(hits[0])) if len(hits) == 1 else predict.known_user_timeline(a.user)
    else:
        tl = predict.recording_timeline(a.acc, a.gyro, label=Path(a.acc).name, clock=a.clock)
    t_tl = time.perf_counter() - t0
    if a.save_timeline:
        predict.T.save_timeline(tl, a.save_timeline)

    queries = [l.strip() for l in open(a.questions) if l.strip() and not l.startswith("#")] \
        if a.questions else []
    # Words accumulate into a question until one ends with "?", so both
    #   ask_upgraded.py -u X How long did she walk?
    #   ask_upgraded.py -u X "Question one?" "Question two?"
    # do the right thing. (Joining everything would merge separate questions.)
    cur = []
    for w in a.question:
        cur.append(w)
        if w.rstrip().endswith("?"):
            queries.append(" ".join(cur)); cur = []
    if cur:
        queries.append(" ".join(cur))
    if not queries and not a.interactive:
        p.error("give a question, --questions FILE, or --interactive")

    import slm_query_engine as S_                          # read-only import from the project
    S = S_
    eng = None
    if not a.no_slm:
        t1 = time.perf_counter(); eng = S.QwenEngine(); t_load = time.perf_counter() - t1
    sink = open(a.out, "w") if a.out else sys.stdout
    s = tl.get("source", {})
    print(f"# recording: {tl.get('user', '')}  ({len(tl['intervals'])} intervals, "
          f"{tl.get('recording_span_s', 0):,.0f} s, {tl.get('time_base', '')})", file=sink)
    print(f"# classifier: {s.get('classifier', '?')}  [{s.get('kind', '?')}"
          + (f", time of day {'on' if s.get('time_of_day') else 'OFF'}" if 'time_of_day' in s else "")
          + f"]  timeline ready in {t_tl:.1f}s", file=sink)
    if s.get("kind") == "new recording":
        r = s["ingest"]
        print(f"# ingest: {r['segments']:,} segments from {r['windows_used']:,} minutes, unit {r['unit']}"
              f" ({100 * r['unit_agreement']:.0f}% agreement), {r['windows_without_gyro']} minutes "
              f"without gyroscope skipped", file=sink)
        if r["unit"] == "inconsistent":
            print("# WARNING: accelerometer units are inconsistent across this recording; "
                  "answers may be unreliable", file=sink)
    if eng:
        print(f"# model: {eng.model_id}, loaded in {t_load:.1f}s", file=sink)

    def run(q):
        r = S.answer_query(q, tl, eng, explain=not (a.no_explain or a.no_slm))
        print(f"\nQuery: {q}\n{r['text']}\n# intent={r['intent']['intent']} "
              f"activities={r['intent']['activities']} latency={r['seconds']:.2f}s", file=sink)
        sink.flush()

    for q in queries:
        run(q)
    if a.interactive:
        print("\n# interactive - blank line or Ctrl-D to quit", file=sys.stderr)
        while True:
            try:
                q = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run(q)
    if a.stats and eng:
        import torch
        st = eng.stats; n = max(st["parse_calls"], 1)
        print(f"\n# {n} queries | {st['seconds'] / n:.2f}s each on GPU | peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", file=sink)
    if a.out:
        sink.close(); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
