"""Speech segments -> windows, each holding roughly one action or explanation.

A window is cut at a pause once it is at least min_len long, and always cut at
max_len. Each window also carries the tail of the previous window as read-only
context, so the LLM can resolve "that valve" without extracting from it.
"""
import json
import sys


def make_windows(segments, min_len=20.0, max_len=60.0, pause=1.0, overlap=5.0):
    groups, cur = [], []
    for i, seg in enumerate(segments):
        cur.append(seg)
        length = cur[-1]["end"] - cur[0]["start"]
        last = i + 1 == len(segments)
        # silence between this segment and the next one
        gap = 999.0 if last else segments[i + 1]["start"] - seg["end"]
        if last or length >= max_len or (length >= min_len and gap >= pause):
            groups.append(cur)
            cur = []

    windows = []
    for n, g in enumerate(groups):
        start, end = g[0]["start"], g[-1]["end"]
        context = ""
        if n > 0:
            # previous window's segments that end within `overlap` seconds of this start
            tail = [s["text"] for s in groups[n - 1] if s["end"] >= start - overlap]
            context = " ".join(tail)
        windows.append({"id": f"w{n:02d}", "start": start, "end": end,
                        "text": " ".join(s["text"] for s in g), "context": context})
    return windows


if __name__ == "__main__":
    # Inspect windows:  python windows.py store/transcript_demo.json
    import config as cfg
    segs = json.loads(open(sys.argv[1], encoding="utf-8").read())
    for w in make_windows(segs, cfg.MIN_LEN, cfg.MAX_LEN, cfg.PAUSE, cfg.OVERLAP):
        print(f"{w['id']} {w['start']:.1f}-{w['end']:.1f}s  {w['text']}\n")
