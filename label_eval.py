"""Write data/eval_questions.json for YOUR clip, with you as the labeller.

    python label_eval.py hyd5.mp4

It shows each verified record from that clip. For each one, type a question a
junior would ask that this record answers (or press Enter to skip it). It stops
at 10 answerable questions, then adds 10 unanswerable ones and saves the file.

Ground truth must come from a person, so this script never writes questions
itself. Write each question in your own words, the way someone with the problem
would ask it: copying the record's wording makes retrieval look better than it is.
"""
import json
import pathlib
import re
import sys

N_ANSWERABLE = 10
OUT = pathlib.Path("data/eval_questions.json")

# Questions a hydraulics or general maintenance clip should NOT answer.
# They avoid hydraulic words on purpose, so a correct system abstains on them.
# Replace any that your clip does happen to cover.
UNANSWERABLE = [
    "How do I reset the PLC after a power cut?",
    "What is the warranty period on the motor?",
    "How do I log in to the maintenance software?",
    "How do I replace the touchscreen on the control panel?",
    "What voltage does the welding machine need?",
    "How do I calibrate the barcode scanner at the packing station?",
    "Who renews my forklift licence?",
    "How do I update the Wi-Fi driver on the shop laptop?",
    "What is the shift handover procedure?",
    "How do I book annual leave?",
]


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def copies_record(question, record):
    """True if the question reuses any 4 consecutive words of the record."""
    q = words(question)
    src = " " + " ".join(words(" ".join([record["step"], record["evidence"], record["rationale"]]))) + " "
    return any(" " + " ".join(q[i:i + 4]) + " " in src for i in range(len(q) - 3))


def main(clip):
    recs = json.loads(pathlib.Path("store/records.json").read_text(encoding="utf-8"))
    recs = [r for r in recs if r["source"] == clip and not r.get("sample")]
    if not recs:
        sys.exit(f"No records from '{clip}' in store/records.json. Run run_pipeline.py on it first.")
    print(f"{len(recs)} verified records from {clip}. Need {N_ANSWERABLE} questions; "
          f"you may write more than one per record (blank line moves on).\n")

    answerable = []
    for r in recs:
        if len(answerable) >= N_ANSWERABLE:
            break
        print(f"--- {r['id']}  at {r['start']:.0f}s\n  step:      {r['step']}\n  component: {r['component']}\n"
              f"  why:       {r['rationale'] or '-'}\n  fault:     {r['failure_indicator'] or '-'}\n"
              f"  quote:     \"{r['evidence']}\"")
        while len(answerable) < N_ANSWERABLE:
            q = input(f"  question {len(answerable) + 1}/{N_ANSWERABLE} (Enter = next record): ").strip()
            if not q:
                break
            if copies_record(q, r):
                print("  ! That reuses the record's own wording. Rephrase it as a junior would ask, or keep it with 'y'.")
                if input("  keep anyway? [y/N] ").strip().lower() != "y":
                    continue
            answerable.append({"question": q, "expected": r["id"]})
        print()

    if len(answerable) < N_ANSWERABLE:
        print(f"Only {len(answerable)} answerable questions written. That is allowed; report the real count.")
    if OUT.exists():
        OUT.with_suffix(".json.bak").write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
    OUT.write_text(json.dumps(answerable + [{"question": q, "expected": None} for q in UNANSWERABLE],
                              indent=1), encoding="utf-8")
    print(f"Saved {OUT}: {len(answerable)} answerable + {len(UNANSWERABLE)} unanswerable "
          f"(previous file kept as {OUT.name}.bak). Next: python tune.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python label_eval.py <clip file name, e.g. hyd5.mp4>")
    main(pathlib.Path(sys.argv[1]).name)
