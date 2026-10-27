"""Consent gate and hash-chained, append-only event log.

Each line of store/consent_log.jsonl is one event:
    {"ts", "event", "who", "detail", "prev", "hash"}
where hash = sha256(prev + canonical JSON of {ts, event, who, detail}).
Editing any line changes its hash, so that line and every line after it
fail verification.

Limits (say these to judges): the chain shows the log was not edited after the
fact. It does NOT prove who clicked consent (the name is typed in), and someone
with file access could rewrite the whole chain from scratch. Anchoring the
latest hash outside the device is Stage II.
"""
import hashlib
import json
import pathlib
import time

LOG = pathlib.Path("store/consent_log.jsonl")
GENESIS = "0" * 64                       # "previous hash" of the very first line
FIELDS = ("ts", "event", "who", "detail")  # the fields covered by the hash


class ConsentRequired(PermissionError):
    """Raised when capture is attempted without the expert's consent."""


def _digest(prev, body):
    # sort_keys makes the JSON canonical: same body always gives the same bytes.
    canonical = json.dumps(body, sort_keys=True)
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


def _lines():
    if not LOG.exists():
        return []
    return [l for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]


def _last_hash():
    lines = _lines()
    return json.loads(lines[-1])["hash"] if lines else GENESIS


def log_event(event, who, detail=""):
    """Append one event, chained to the previous line's hash."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    body = {"ts": time.time(), "event": event, "who": who, "detail": str(detail)}
    prev = _last_hash()
    entry = {**body, "prev": prev, "hash": _digest(prev, body)}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def verify():
    """Return (True, number_of_entries) or (False, first_bad_line_number)."""
    prev, count = GENESIS, 0
    if not LOG.exists():
        return True, 0
    for n, line in enumerate(LOG.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            body = {k: e[k] for k in FIELDS}
            ok = e["prev"] == prev and e["hash"] == _digest(prev, body)
        except (ValueError, KeyError):
            ok = False                     # unparseable or missing field = tampered
        if not ok:
            return False, n
        prev, count = e["hash"], count + 1
    return True, count


def require_consent(granted, who, clip):
    """Call before any capture work. Verifies the chain, then logs the decision."""
    ok, where = verify()
    if not ok:
        raise RuntimeError(f"Consent log failed verification at line {where}; refusing to continue.")
    who = (who or "").strip()
    if not (granted and who):
        log_event("capture_refused", who or "unknown", clip)
        raise ConsentRequired("Capture refused: the expert has not given consent (tick consent and enter a name).")
    log_event("consent_granted", who, clip)


if __name__ == "__main__":
    print(verify())
