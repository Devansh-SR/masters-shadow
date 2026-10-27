"""Evaluation for the results table. Never writes the index and never logs gaps.

    python eval.py                          # retrieval: top-1, top-3, abstentions, query time
    python eval.py data/clips/demo.mp4      # also times each pipeline stage on the clip

data/eval_questions.json: [{"question": "...", "expected": "<record id>" or null}]
"""
import json
import pathlib
import statistics
import sys
import time

import config as cfg
import query

# Evaluation must not pollute the gap backlog. search() calls the module-level
# log_gap, so replacing it here silences it.
query.log_gap = lambda *a, **k: None


def retrieval(quiet=False):
    """Run every eval question. Prints the table and also returns the numbers
    (tune.py and report.py reuse them)."""
    say = (lambda *a, **k: None) if quiet else print
    qs = json.loads(pathlib.Path("data/eval_questions.json").read_text(encoding="utf-8"))
    records, vecs = query.load()
    ids_in_index = {r["id"] for r in records}
    missing = [q["expected"] for q in qs if q["expected"] and q["expected"] not in ids_in_index]
    if missing:
        print(f"WARNING: expected ids not in the index: {missing}")

    ans = [q for q in qs if q["expected"]]
    unans = [q for q in qs if not q["expected"]]
    top1 = top3 = wrong_abstain = right_abstain = 0
    times, rows = [], []
    say(f"THRESHOLD = {cfg.THRESHOLD}\n\n score  result     question")
    for q in qs:
        t = time.perf_counter()
        res = query.search(q["question"], records, vecs, k=3)
        times.append(time.perf_counter() - t)
        best = res["hits"][0]["score"] if res["hits"] else float("nan")
        ids = [h["id"] for h in res["hits"]]
        if q["expected"]:
            wrong_abstain += res["abstain"]
            top1 += (not res["abstain"]) and ids[:1] == [q["expected"]]
            top3 += (not res["abstain"]) and q["expected"] in ids
            tag = "abstain!" if res["abstain"] else ("top1" if ids[:1] == [q["expected"]]
                                                     else "top3" if q["expected"] in ids else "miss")
        else:
            right_abstain += res["abstain"]
            tag = "abstain" if res["abstain"] else "ANSWERED!"
        # The per-question best score is what you use to choose THRESHOLD.
        say(f" {best:.3f}  {tag:<9}  {'[A] ' if q['expected'] else '[U] '}{q['question']}")
        rows.append({"question": q["question"], "expected": q["expected"], "best": best,
                     "top1_id": ids[0] if ids else None, "in_top3": bool(q["expected"]) and q["expected"] in ids})

    median = statistics.median(times) if times else float("nan")
    say(f"\nanswerable: {len(ans)}  top-1: {top1}  top-3: {top3}  wrongly abstained: {wrong_abstain}")
    say(f"unanswerable: {len(unans)}  correctly abstained: {right_abstain}")
    say(f"median query time: {median:.3f}s over {len(times)} queries")
    return {"n_ans": len(ans), "top1": top1, "top3": top3, "wrong_abstain": wrong_abstain,
            "n_unans": len(unans), "right_abstain": right_abstain, "median_s": median, "rows": rows}


def timing(clip):
    from index import embed_docs, record_text
    from structure import structure_all
    from transcribe import transcribe
    from windows import make_windows

    clip = pathlib.Path(clip)
    t0 = time.perf_counter()
    segs = transcribe(clip, cfg.WHISPER, cfg.DEVICE, cfg.COMPUTE); t1 = time.perf_counter()
    wins = make_windows(segs, cfg.MIN_LEN, cfg.MAX_LEN, cfg.PAUSE, cfg.OVERLAP); t2 = time.perf_counter()
    recs = structure_all(wins, clip.name); t3 = time.perf_counter()
    good = [r for r in recs if r["verified"]]
    if good:
        embed_docs([record_text(r) for r in good])   # embedded but NOT saved
    t4 = time.perf_counter()
    print(f"\ndevice={cfg.DEVICE} compute={cfg.COMPUTE}")
    print(f"transcribe {t1 - t0:.1f}s  window {t2 - t1:.2f}s  structure {t3 - t2:.1f}s  "
          f"embed {t4 - t3:.1f}s  total {t4 - t0:.1f}s")
    print(f"segments: {len(segs)}  windows: {len(wins)}  records: {len(good)} verified, "
          f"{len(recs) - len(good)} unverified")
    return {"transcribe": t1 - t0, "window": t2 - t1, "structure": t3 - t2, "embed": t4 - t3,
            "total": t4 - t0, "segments": len(segs), "windows": len(wins),
            "verified": len(good), "unverified": len(recs) - len(good)}


if __name__ == "__main__":
    retrieval()
    if len(sys.argv) > 1:
        timing(sys.argv[1])
