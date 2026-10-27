"""Records -> local index: store/records.json + store/vectors.npy.

Row i of vectors.npy is the normalized embedding of record i in records.json.
Adding records replaces only records from the same source clip, so sample
records and other clips survive a re-run.
"""
import argparse
import json
import pathlib

import numpy as np
import ollama

import consent
from config import EMBED

STORE = pathlib.Path("store")
DIM = 768  # nomic-embed-text output size


def record_text(r):
    """The text that gets embedded for a record. The verified quote is included because
    reason and fault are often blanked by the grounding check, leaving the speaker's own
    words only in the quote."""
    return (f"{r['component']}. {r['step']}. {r['rationale']} {r['failure_indicator']} "
            f"{r['evidence']}").strip()


def embed_docs(texts):
    """Embed documents with the 'search_document: ' prefix and L2-normalize each row."""
    e = ollama.embed(model=EMBED, input=[f"search_document: {t}" for t in texts])
    v = np.array(e["embeddings"], dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _read():
    rp, vp = STORE / "records.json", STORE / "vectors.npy"
    recs = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else []
    vecs = np.load(vp) if vp.exists() else np.zeros((0, DIM), dtype="float32")
    if len(recs) != len(vecs):
        raise RuntimeError("records.json and vectors.npy are out of step; delete both and re-run.")
    return recs, vecs


def _write(recs, vecs):
    STORE.mkdir(exist_ok=True)
    (STORE / "records.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    np.save(STORE / "vectors.npy", vecs.astype("float32"))


def add_records(new, source=None):
    """Append `new`, first removing old records from the same source(s).
    Pass `source` so a re-run that yields zero records still clears the old ones."""
    old, vecs = _read()
    replaced = {r["source"] for r in new}
    if source:
        replaced.add(source)
    keep = [i for i, r in enumerate(old) if r["source"] not in replaced]
    old, vecs = [old[i] for i in keep], vecs[keep]
    if new:
        nv = embed_docs([record_text(r) for r in new])
        old, vecs = old + list(new), np.vstack([vecs, nv])
    _write(old, vecs)
    return len(old)


def delete_record(rid, who):
    """Expert review: remove one record AND its vector together, and log the deletion."""
    recs, vecs = _read()
    keep = [i for i, r in enumerate(recs) if r["id"] != rid]
    if len(keep) == len(recs):
        return False
    _write([recs[i] for i in keep], vecs[keep])
    consent.log_event("record_deleted", who, rid)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manage the sample records (fallback demo only).")
    ap.add_argument("--load-samples", action="store_true", help="add data/sample_records.json")
    ap.add_argument("--remove-samples", action="store_true", help="remove all sample records")
    ap.add_argument("--remove-source", metavar="CLIP", help="remove all records from one clip, e.g. hyd5.mp4")
    ap.add_argument("--list-sources", action="store_true", help="count records per source clip")
    a = ap.parse_args()
    if a.load_samples:
        samples = json.loads(pathlib.Path("data/sample_records.json").read_text(encoding="utf-8"))
        print(add_records(samples), "records in index")
    elif a.remove_samples:
        print(add_records([], source="sample"), "records in index")
    elif a.remove_source:
        print(add_records([], source=a.remove_source), "records in index")
    elif a.list_sources:
        from collections import Counter
        recs, _ = _read()
        for src, n in Counter(r["source"] for r in recs).most_common():
            print(f"{n:4d}  {src}")
    else:
        ap.print_help()
