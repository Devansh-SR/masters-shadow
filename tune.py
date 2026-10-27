"""Suggest THRESHOLD and GAP_SIM from your own data, and optionally write them to config.py.

    python tune.py            # print the scores and the suggestions
    python tune.py --write    # also write both values into config.py

THRESHOLD: separates the best scores of answerable questions answered correctly
(top-1) from the best scores of unanswerable questions (data/eval_questions.json).
GAP_SIM: separates paraphrases of the same question from different questions
(data/gap_pairs.json).

Both are tuned on the same questions that eval.py later reports, so the README must
say so: the results are indicative, not a held-out test.
"""
import itertools
import json
import pathlib
import re
import sys

import eval as ev
import query


def best_cut(pos, neg):
    """Value that best separates pos (should be >= cut) from neg (should be < cut).
    Candidate cuts are the midpoints between neighbouring observed scores, never a score
    itself, so no question sits exactly on the cut. Pick the fewest errors; among ties,
    the widest margin to the nearest score. A clean split gives the midpoint of the gap."""
    if not pos or not neg:
        return None, None
    vals = sorted(set(pos + neg))
    cuts = [vals[0] - 0.01] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 0.01]

    def errors(c):
        return sum(p < c for p in pos) + sum(n >= c for n in neg)

    def margin(c):
        return min(abs(c - v) for v in vals)

    best = min(cuts, key=lambda c: (errors(c), -margin(c), -c))
    return round(best, 3), errors(best)


def tune_threshold():
    res = ev.retrieval(quiet=True)
    rows = res["rows"]
    pos = [r["best"] for r in rows if r["expected"] and r["top1_id"] == r["expected"]]
    neg = [r["best"] for r in rows if not r["expected"]]
    wrong_top1 = [r for r in rows if r["expected"] and r["top1_id"] != r["expected"]]
    print("Best score per question (sorted):")
    for r in sorted(rows, key=lambda r: -r["best"]):
        kind = "U" if not r["expected"] else ("A ok" if r["top1_id"] == r["expected"] else "A wrong-top1")
        print(f"  {r['best']:.3f}  {kind:<13} {r['question']}")
    cut, errors = best_cut(pos, neg)
    if cut is None:
        print("\nTHRESHOLD: need at least one correctly answered and one unanswerable question.")
    else:
        print(f"\nAnswerable answered correctly: {len(pos)} (lowest {min(pos):.3f}); "
              f"unanswerable: {len(neg)} (highest {max(neg):.3f})")
        print(f"Suggested THRESHOLD = {cut}" + (" (clean separation)" if errors == 0
              else f" (groups overlap: {errors} question(s) land on the wrong side; report this)"))
    if wrong_top1:
        print(f"Note: {len(wrong_top1)} answerable question(s) have the wrong record first; "
              "a threshold cannot fix that.")
    return cut


def tune_gap_sim():
    pairs = json.loads(pathlib.Path("data/gap_pairs.json").read_text(encoding="utf-8"))
    vecs = [(query.embed_query(a), query.embed_query(b)) for a, b in pairs]
    same = [float(a @ b) for a, b in vecs]
    diff = [float(x @ y) for (i, p), (j, q) in itertools.combinations(enumerate(vecs), 2)
            for x in p for y in q]
    cut, errors = best_cut(same, diff)
    print(f"\nParaphrase pairs: lowest similarity {min(same):.3f}; different questions: highest {max(diff):.3f}")
    print(f"Suggested GAP_SIM = {cut}" + (" (clean separation)" if errors == 0
          else f" (overlap: {errors} comparison(s) on the wrong side)"))
    return cut


def write_config(name, value):
    p = pathlib.Path("config.py")
    text = p.read_text(encoding="utf-8")
    line = f"{name} = {value}  # tuned by tune.py on this repo's eval data (same questions as the results)"
    new, n = re.subn(rf"^{name} = .*$", line, text, flags=re.M)
    if n != 1:
        sys.exit(f"Could not find a single '{name} = ...' line in config.py; edit it by hand.")
    p.write_text(new, encoding="utf-8")
    print(f"config.py: {line}")


if __name__ == "__main__":
    thr = tune_threshold()
    gap = tune_gap_sim()
    if "--write" in sys.argv:
        print()
        if thr is not None:
            write_config("THRESHOLD", thr)
        if gap is not None:
            write_config("GAP_SIM", gap)
        print("Now run: python eval.py   (it re-reads config.py)")
    else:
        print("\nRe-run with --write to save these into config.py.")
