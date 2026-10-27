"""Index integrity and retrieval diagnosis. Read-only: changes nothing.

    python check_index.py

1. Records per source clip (stale clips show up here).
2. Alignment: re-embeds every record one at a time and compares it with the stored
   vector. Every row should score about 1.00; a low score means record and vector
   are out of step, and every search result is wrong.
3. For each answerable eval question: where the expected record ranks, its score,
   and the top 3 records actually returned.
"""
import json
import pathlib
from collections import Counter

import numpy as np

import query
from index import embed_docs, record_text

query.log_gap = lambda *a, **k: None   # diagnosis must not write the gap log

recs, vecs = query.load()
print("== 1. records per source ==")
for src, n in Counter(r["source"] for r in recs).most_common():
    print(f"  {n:4d}  {src}")

print("\n== 2. alignment (each record re-embedded alone vs stored vector) ==")
fresh = np.vstack([embed_docs([record_text(r)]) for r in recs])
self_sim = (fresh * vecs).sum(axis=1)
bad = [(i, s) for i, s in enumerate(self_sim) if s < 0.98]
print(f"  min {self_sim.min():.3f}   median {np.median(self_sim):.3f}   rows below 0.98: {len(bad)} of {len(recs)}")
for i, s in bad[:10]:
    best = int(np.argmax(vecs @ fresh[i]))
    print(f"   row {i} {recs[i]['id']}: {s:.3f}; its text best matches stored row {best} ({recs[best]['id']})")

print("\n== 3. answerable questions ==")
qs = [q for q in json.loads(pathlib.Path("data/eval_questions.json").read_text(encoding="utf-8")) if q["expected"]]
ids = [r["id"] for r in recs]
for q in qs:
    qv = query.embed_query(q["question"])
    sims = vecs @ qv
    order = list(np.argsort(-sims))
    if q["expected"] in ids:
        e = ids.index(q["expected"])
        where = f"expected {q['expected']} rank {order.index(e) + 1}/{len(recs)} score {sims[e]:.3f}"
    else:
        where = f"expected {q['expected']} NOT IN INDEX"
    print(f"\n  Q: {q['question']}\n     {where}")
    for i in order[:3]:
        print(f"     {sims[i]:.3f}  {recs[i]['id']:<16} {recs[i]['step']}")
