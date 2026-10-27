"""Measure on THIS machine and write the automatic rows of the README results table.

    python report.py data/clips/hyd5.mp4 --source "Abom79, Hydraulic System Inspection & Troubleshooting Session 2 (archive.org), first 5 min"

Fills: test clip, machine, speech device, clip length, records rejected by the quote
check, retrieval (top-1, top-3, abstentions), seconds per stage, median query time,
and whether the threshold was tuned on the same questions.
Leaves for you (they need a person): hand-labelled steps, the read-through of records,
peak memory. Those rows keep [NOT MEASURED] until you edit them.

It re-runs the models once for timing (a few minutes) and never writes the index or gap log.
"""
import argparse
import ctypes
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

import config as cfg
import eval as ev

README = pathlib.Path("README.md")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def ram_gb():
    if os.name == "nt":
        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEM(); m.dwLength = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.ullTotalPhys / 2**30)
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)


def machine():
    cpu = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) \
        if os.name == "nt" else platform.processor()
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    ollama = run(["ollama", "--version"]).replace("ollama version is", "").strip()
    release = platform.release()
    # platform.release() says "10" on Windows 11; build 22000 and later is Windows 11.
    if platform.system() == "Windows" and release == "10" and int(platform.version().split(".")[-1]) >= 22000:
        release = "11"
    osname = f"{platform.system()} {release} (build {platform.version()})"
    return (f"{cpu.strip() or '[CPU]'}; {ram_gb()} GB RAM; {gpu or 'no NVIDIA GPU found'}; {osname}; "
            f"Python {platform.python_version()}; Ollama {ollama or '[version]'}")


def clip_seconds(path):
    import av  # installed with faster-whisper
    with av.open(str(path)) as c:
        return c.duration / 1e6 if c.duration else float("nan")


def set_row(text, prefix, label, value, how):
    pat = re.compile(rf"^\| {re.escape(prefix)}[^|]*\|.*$", re.M)
    row = f"| {label} | {value} | {how} |"
    if not pat.search(text):
        sys.exit(f"README row starting with '{prefix}' not found; was the table edited?")
    return pat.sub(lambda _: row, text, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--source", required=True, help="where the clip came from, in plain words")
    a = ap.parse_args()
    clip = pathlib.Path(a.clip)

    print("== retrieval =="); r = ev.retrieval()
    print("\n== timing (re-runs the models) =="); t = ev.timing(clip)
    unver = json.loads(pathlib.Path("store/unverified.json").read_text(encoding="utf-8")) \
        if pathlib.Path("store/unverified.json").exists() else []
    rejected = sum(1 for u in unver if u.get("source") == clip.name)
    indexed = sum(1 for x in json.loads(pathlib.Path("store/records.json").read_text(encoding="utf-8"))
                  if x["source"] == clip.name)
    tuned = "yes" if "tuned by tune.py" in pathlib.Path("config.py").read_text(encoding="utf-8") else \
        "no (set by hand)"

    text = README.read_text(encoding="utf-8")
    text = set_row(text, "Test clip", "Test clip", f"`{clip.name}`: {a.source}", "Not in this repository")
    text = set_row(text, "Machine", "Machine", machine(), "report.py (OS, nvidia-smi, ollama)")
    text = set_row(text, "Speech device", "Speech device", f"{cfg.DEVICE} / {cfg.COMPUTE}", "`config.py`")
    text = set_row(text, "Clip length", "Clip length", f"{clip_seconds(clip):.0f} s", "PyAV")
    text = set_row(text, "Records rejected", "Records rejected by the quote/grounding checks",
                   f"{rejected} of {rejected + indexed} (indexed run); {t['unverified']} of "
                   f"{t['verified'] + t['unverified']} (timing run)",
                   "`store/unverified.json`, `eval.py <clip>`")
    text = set_row(text, "Answerable", f"Answerable (of {r['n_ans']}): top-1, top-3, wrongly abstained",
                   f"{r['top1']}, {r['top3']}, {r['wrong_abstain']}", "`eval.py`")
    text = set_row(text, "Unanswerable", f"Unanswerable (of {r['n_unans']}): correctly abstained",
                   f"{r['right_abstain']}", "`eval.py`")
    text = set_row(text, "Seconds", "Seconds: transcribe, window, structure, embed (total)",
                   f"{t['transcribe']:.1f}, {t['window']:.2f}, {t['structure']:.1f}, {t['embed']:.1f} "
                   f"({t['total']:.1f})", "`eval.py <clip>`")
    text = set_row(text, "Median query time", "Median query time",
                   f"{r['median_s']:.3f} s over {r['n_ans'] + r['n_unans']} queries", "`eval.py`")
    text = set_row(text, "THRESHOLD tuned", "THRESHOLD tuned on the same questions?",
                   f"{tuned}; THRESHOLD = {cfg.THRESHOLD}, GAP_SIM = {cfg.GAP_SIM}", "`tune.py`")
    README.write_text(text, encoding="utf-8")
    print("\nREADME.md results table updated. Still yours to fill by hand: expected steps, "
          "the 10-record read-through, peak memory.")


if __name__ == "__main__":
    main()
