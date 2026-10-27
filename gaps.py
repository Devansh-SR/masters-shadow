"""Group logged gap questions so the expert sees "three juniors asked about X".

Greedy clustering: each question joins the first group whose first question
(the group's "center") has cosine >= GAP_SIM with it; otherwise it starts a
new group. Vectors are already normalized, so the dot product is the cosine.
"""
import json
import pathlib

import numpy as np

from config import GAP_SIM

GAP_LOG = pathlib.Path("store/gap_log.jsonl")


def clusters():
    if not GAP_LOG.exists():
        return []
    rows = [json.loads(l) for l in GAP_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = []
    for r in rows:
        v = np.array(r["vector"], dtype="float32")
        for g in groups:
            if float(g["center"] @ v) >= GAP_SIM:
                g["questions"].append(r["question"])
                break
        else:                                   # no group was close enough
            groups.append({"center": v, "questions": [r["question"]]})
    groups.sort(key=lambda g: -len(g["questions"]))  # largest first
    return [{"count": len(g["questions"]), "example": g["questions"][0],
             "questions": g["questions"]} for g in groups]


if __name__ == "__main__":
    for c in clusters():
        print(f"{c['count']}x  {c['example']}")
