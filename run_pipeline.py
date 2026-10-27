"""One command: consent -> transcribe -> window -> structure -> index.

    python run_pipeline.py data/clips/demo.mp4 --who "Expert Name" --consent

Without --consent it refuses and logs the refusal.
"""
import argparse
import json
import pathlib

import config as cfg
from consent import log_event, require_consent
from index import add_records
from structure import structure_all
from transcribe import transcribe
from windows import make_windows

UNVERIFIED = pathlib.Path("store/unverified.json")


def _save_unverified(bad, source):
    """Keep unverified records for inspection, replacing only this clip's entries."""
    old = json.loads(UNVERIFIED.read_text(encoding="utf-8")) if UNVERIFIED.exists() else []
    keep = [r for r in old if r.get("source") != source]
    UNVERIFIED.parent.mkdir(exist_ok=True)
    UNVERIFIED.write_text(json.dumps(keep + bad, indent=1), encoding="utf-8")


def run(clip_path, who, granted):
    """Return (n_verified, n_unverified)."""
    clip = pathlib.Path(clip_path)
    who = (who or "").strip()
    # FIRST line of real work: nothing is opened before consent is checked and logged.
    require_consent(granted, who, clip.name)
    try:
        if not clip.exists():
            raise FileNotFoundError(f"Clip not found: {clip}")
        print("transcribing...")
        segs = transcribe(clip, cfg.WHISPER, cfg.DEVICE, cfg.COMPUTE)
        wins = make_windows(segs, cfg.MIN_LEN, cfg.MAX_LEN, cfg.PAUSE, cfg.OVERLAP)
        print(f"{len(segs)} segments -> {len(wins)} windows; structuring...")
        recs = structure_all(wins, clip.name)
        good = [r for r in recs if r["verified"]]
        bad = [r for r in recs if not r["verified"]]
        _save_unverified(bad, clip.name)
        add_records(good, source=clip.name)       # only verified records are searchable
    finally:
        # Logged even if a stage crashes, so the ledger always shows capture ended.
        log_event("capture_stopped", who, clip.name)
    log_event("records_published", who, f"{len(good)} records from {clip.name}")
    return len(good), len(bad)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--who", required=True, help="name of the expert giving consent")
    ap.add_argument("--consent", action="store_true", help="the expert consents to capture")
    a = ap.parse_args()
    good, bad = run(a.clip, a.who, a.consent)
    print(f"{good} records indexed, {bad} unverified (see store/unverified.json)")
