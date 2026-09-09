#!/usr/bin/env python3
"""Ask the Sensors - command-line interface to the activity question-answering system.

Examples
--------
  # one question
  python3 ask.py --user 00EABED2 "How long did she walk in total?"

  # several questions from a file, answers written out in the required format
  python3 ask.py --user 00EABED2 --questions qs.txt --out answers.txt

  # interactive session (model stays loaded between questions)
  python3 ask.py --user 00EABED2 -i

  # no language model at all: structured fields only, ~5 ms per query
  python3 ask.py --user 00EABED2 --no-slm "How many times did she walk?"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

TIMELINE_DIR = "timelines"


def find_timeline(spec: str) -> str:
    """Accept a path, a full UUID, or a unique prefix of one."""
    if os.path.isfile(spec):
        return spec
    hits = sorted(glob.glob(os.path.join(TIMELINE_DIR, f"{spec}*.json")))
    if not hits:
        avail = [os.path.basename(p)[:8] for p in sorted(glob.glob(f"{TIMELINE_DIR}/*.json"))]
        sys.exit(f"no timeline matching {spec!r} in {TIMELINE_DIR}/\n"
                 f"available: {', '.join(avail) if avail else '(none - build one first)'}")
    if len(hits) > 1:
        sys.exit(f"{spec!r} matches {len(hits)} timelines; be more specific")
    return hits[0]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Answer natural-language questions about a sensor recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("question", nargs="*", help="the question to ask")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", "-u", help="timeline UUID or unique prefix")
    g.add_argument("--timeline", "-t", help="path to a timeline JSON file")
    p.add_argument("--questions", "-q", help="file with one question per line")
    p.add_argument("--out", "-o", help="write answers here instead of stdout")
    p.add_argument("--interactive", "-i", action="store_true", help="ask repeatedly")
    p.add_argument("--no-slm", action="store_true",
                   help="skip the language model (no free-text parsing, no Explanation)")
    p.add_argument("--no-explain", action="store_true",
                   help="parse with the model but skip the Explanation field")
    p.add_argument("--quantize", choices=["4bit", "8bit"],
                   help="load the model quantized (for the efficiency comparison)")
    p.add_argument("--stats", action="store_true", help="print resource usage at the end")
    a = p.parse_args()

    path = a.timeline or find_timeline(a.user)
    tl = json.load(open(path))

    queries = []
    if a.questions:
        queries += [l.strip() for l in open(a.questions) if l.strip()
                    and not l.startswith("#")]
    if a.question:
        queries.append(" ".join(a.question))
    if not queries and not a.interactive:
        sys.exit("give a question, --questions FILE, or --interactive")

    import slm_query_engine as S
    eng = None
    t_load = 0.0
    if not a.no_slm:
        t0 = time.perf_counter()
        eng = S.QwenEngine(quantization=a.quantize)
        t_load = time.perf_counter() - t0

    sink = open(a.out, "w") if a.out else sys.stdout
    n_user = len(tl.get("intervals", []))
    print(f"# recording: {os.path.basename(path)}  "
          f"({n_user} intervals, {tl.get('recording_span_s', 0):,.0f} s, "
          f"{tl.get('time_base', '')})", file=sink)
    if eng:
        print(f"# model: {eng.model_id}"
              f"{' (' + a.quantize + ')' if a.quantize else ''}, loaded in {t_load:.1f}s",
              file=sink)

    def run(q: str) -> None:
        r = S.answer_query(q, tl, eng, explain=not (a.no_explain or a.no_slm))
        print(f"\nQuery: {q}", file=sink)
        print(r["text"], file=sink)
        print(f"# intent={r['intent']['intent']} "
              f"activities={r['intent']['activities']} "
              f"latency={r['seconds']:.2f}s", file=sink)
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
        s = eng.stats
        n = max(s["parse_calls"], 1)
        print(f"\n# {n} queries | {s['seconds']:.1f}s on GPU "
              f"({s['seconds'] / n:.2f}s each) | "
              f"{s['tokens_in']:,} tokens in, {s['tokens_out']:,} out", file=sink)
        if torch.cuda.is_available():
            print(f"# peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", file=sink)
    if a.out:
        sink.close()
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
