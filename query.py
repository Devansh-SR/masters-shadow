"""Question -> ranked records, with abstention and gap logging.

Search is a dot product of normalized vectors, which equals cosine similarity.
If the best score is below THRESHOLD, the system abstains and logs the question
as a knowledge gap instead of guessing.
"""
import json
import pathlib
import sys
import time

import numpy as np
import ollama

from config import EMBED, THRESHOLD, TOP_K

STORE = pathlib.Path("store")


def load():
    """Return (records, vectors). Raises FileNotFoundError if nothing is indexed yet."""
    recs = json.loads((STORE / "records.json").read_text(encoding="utf-8"))
    vecs = np.load(STORE / "vectors.npy")
    if len(recs) != len(vecs):
        raise RuntimeError("records.json and vectors.npy are out of step; delete both and re-run.")
    return recs, vecs


def embed_query(question):
    q = np.array(ollama.embed(model=EMBED, input=[f"search_query: {question}"])["embeddings"],
                 dtype="float32")[0]
    return q / np.linalg.norm(q)


def log_gap(question, score, vec):
    """Append an unanswered question to store/gap_log.jsonl for the expert to see."""
    STORE.mkdir(exist_ok=True)
    with open(STORE / "gap_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "question": question, "score": score,
                            "vector": vec.tolist()}) + "\n")


def search(question, records, vecs, k=TOP_K):
    """Return {"abstain": bool, "hits": [record + score, best first]}."""
    q = embed_query(question)
    sims = vecs @ q if len(vecs) else np.zeros(0, dtype="float32")
    hits = [{"score": float(sims[i]), **records[i]} for i in np.argsort(-sims)[:k]]
    if not hits or hits[0]["score"] < THRESHOLD:
        log_gap(question, hits[0]["score"] if hits else None, q)
        return {"abstain": True, "hits": hits}
    return {"abstain": False, "hits": hits}


if __name__ == "__main__":
    # python query.py "why is the pressure needle oscillating?"
    res = search(" ".join(sys.argv[1:]), *load())
    print("ABSTAIN (logged as gap)" if res["abstain"] else "ANSWER")
    for h in res["hits"]:
        print(f"{h['score']:.3f}  {h['id']}  {h['start']:.0f}s  {h['step']}")
