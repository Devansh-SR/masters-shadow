"""Windows -> structured records via a local LLM (Ollama), with a quote check.

Three things keep a 3B model honest:
  1. output is forced into a JSON schema,
  2. temperature is 0,
  3. every record must carry a quote ("evidence") from the window text, and code
     (not the model) checks it against the transcript.
Small models rarely copy a quote perfectly, so the check works in two tiers:
  - exact: the normalized quote appears in the window as is;
  - snapped: the closest run of transcript words matches the quote at >= SNAP_MIN
    (difflib ratio). The stored evidence is then REPLACED by those transcript words,
    so every quote the app shows is verbatim from the transcript.
Anything weaker is marked verified=False and never indexed.
A real quote is not enough on its own: the model can attach a true quote to an invented
step. So a second check grounds the other fields in the transcript:
  - component: at least one of its content words must appear in the window (or its
    read-only context, which exists to resolve "that valve"); otherwise rejected;
  - rationale and failure_indicator: if under half their content words appear in the
    window, the field is blanked instead of keeping an invented reason.
Set $env:MS_DEBUG = "1" to print the model's raw output and each check.
"""
import difflib
import os
import json
import pathlib
import re

import ollama

from config import LLM

FIELDS = ["step", "component", "rationale", "failure_indicator", "evidence"]
MAX_EVIDENCE_WORDS = 25
SNAP_MIN = 0.80          # minimum similarity for a snapped quote; raise it to be stricter
DEBUG = os.environ.get("MS_DEBUG") == "1"

RECORD_SCHEMA = {
    "type": "object",
    "properties": {"records": {"type": "array", "items": {
        "type": "object",
        "properties": {k: {"type": "string"} for k in FIELDS},
        "required": FIELDS}}},
    "required": ["records"],
}

SYSTEM = (
    "You extract know-how records from a transcript of an experienced technician "
    "talking while they work or explain equipment. A record is any of: an action they take, "
    "a check or inspection they make, or an explanation of how a part works or fails. "
    "Extract only what the technician actually says in TRANSCRIPT; usually 1 to 4 records. "
    "Use CONTEXT only to understand what words like 'it' or 'that valve' refer to. "
    "If the window is only greetings or chat with no know-how, return an empty list. Never add "
    "parts, values or reasons that are not in TRANSCRIPT. For each record give: step (a short "
    "imperative built only from words and parts the speaker mentions in TRANSCRIPT), "
    "component (the part, as the speaker names it), rationale (the reason the speaker gives, "
    "in their words; empty string if none), failure_indicator (what the speaker says a fault "
    "looks, sounds or feels like; empty string if none), evidence (one continuous phrase copied "
    "word for word from TRANSCRIPT, 8 to 25 words, that supports the record; do not fix grammar)."
)


def norm(t):
    """Lowercase, drop punctuation, collapse spaces, so the quote check ignores formatting."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def _tokens(text):
    """Whitespace tokens of the original text with their normalized form and char span."""
    out = []
    for m in re.finditer(r"\S+", text):
        for word in norm(m.group()).split():      # "it's" -> "it", "s", each keeps the span
            out.append((word, m.start(), m.end()))
    return out


def check_evidence(evidence, window_text):
    """Return (verified, evidence_to_store, how).
    how is 'exact', 'snapped 0.91', or the reason it failed."""
    e = norm(evidence)
    if not e:
        return False, evidence, "empty quote"
    if len(e.split()) > MAX_EVIDENCE_WORDS + 5:
        return False, evidence, "quote too long"
    if f" {e} " in f" {norm(window_text)} ":     # whole words only: "ump" must not match "pump"
        return True, evidence, "exact"
    # Snap: slide over the window's words, compare runs of similar length to the quote.
    q = e.split()
    toks = _tokens(window_text)
    best, span = 0.0, None
    for size in range(max(1, len(q) - 3), len(q) + 4):
        for i in range(0, max(1, len(toks) - size + 1)):
            run = toks[i:i + size]
            if not run:
                continue
            r = difflib.SequenceMatcher(None, q, [t[0] for t in run], autojunk=False).ratio()
            if r > best:
                best, span = r, (run[0][1], run[-1][2])
    if span and best >= SNAP_MIN:
        snapped = window_text[span[0]:span[1]].strip(" ,;:")
        return True, snapped, f"snapped {best:.2f}"
    return False, evidence, f"no match (best {best:.2f})"


STOP = set("""a an and are as at be been but by can could do does for from had has have how i if in
into is it its just like may more most my no not of off on or our out so some such than that the their
them then there these they this those to too up us very was we were what when where which while who why
will with would you your check inspect inspection make sure look ensure verify test step part""".split())


def _content(text):
    return [w for w in norm(text).split() if len(w) >= 3 and w not in STOP]


def grounded(field, source):
    """Share of the field's content words found in source (5-letter stems, so leak/leaks match)."""
    words = _content(field)
    if not words:
        return 1.0
    stems = {w[:5] for w in norm(source).split()}
    return sum(w[:5] in stems for w in words) / len(words)


def is_verified(evidence, window_text):
    """True if the quote passes the exact or snapped check."""
    return check_evidence(evidence, window_text)[0]


def structure_window(w, summary, clip):
    user = (f"RUNNING SUMMARY:\n{summary}\n\n"
            f"CONTEXT (read-only, do not extract from it):\n{w['context'] or '(none)'}\n\n"
            f"TRANSCRIPT (extract only from this):\n{w['text']}")
    items = []
    for attempt in range(2):                      # one try plus one retry
        try:
            r = ollama.chat(model=LLM, format=RECORD_SCHEMA, options={"temperature": 0},
                            messages=[{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": user}])
            raw = r["message"]["content"]
            if DEBUG:
                print(f"--- {w['id']} raw model output:\n{raw}")
            items = json.loads(raw)["records"]
            if not isinstance(items, list):
                raise ValueError("'records' is not a list")
            break
        except Exception as e:                    # print the error, then retry once
            print(f"window {w['id']} attempt {attempt + 1} failed: {e}")
            items = []

    stem = pathlib.Path(clip).stem
    out = []
    for i, it in enumerate(items):
        rec = {"id": f"{stem}-{w['id']}-{i}",     # clip stem makes ids unique across clips
               "window": w["id"], "start": w["start"], "end": w["end"]}
        for k in FIELDS:                          # copy only schema fields, as strings
            rec[k] = str(it.get(k, "")).strip() if isinstance(it, dict) else ""
        ok, ev, how = check_evidence(rec["evidence"], w["text"])
        # grounded() returns 1.0 for a field with no content words, so an empty or "it"
        # component would pass; require at least one content word that is in the transcript.
        if ok and (not _content(rec["component"])
                   or grounded(rec["component"], w["text"] + " " + w["context"]) == 0):
            ok, how = False, how + "; component missing or not in transcript"
        for k in ("rationale", "failure_indicator"):
            if rec[k] and grounded(rec[k], w["text"]) < 0.5:
                if DEBUG:
                    print(f"  [{k} blanked: not in transcript] {rec[k]!r}")
                rec[k] = ""
        if DEBUG:
            print(f"  [{how}] {rec['evidence']!r}")
        rec["evidence"], rec["verified"], rec["match"] = ev, ok, how
        rec["source"] = clip
        rec["sample"] = False
        out.append(rec)
    return out


def structure_all(windows, clip_name):
    """Structure every window. The running summary is built in code from VERIFIED
    components only (no extra model call), so it cannot invent anything."""
    records, comps = [], set()
    for n, w in enumerate(windows, 1):
        summary = "Components discussed so far: " + (", ".join(sorted(comps)) or "none")
        recs = structure_window(w, summary, clip_name)
        comps |= {r["component"] for r in recs if r["verified"] and r["component"]}
        ok = sum(r["verified"] for r in recs)
        print(f"  window {n}/{len(windows)} ({w['id']}): {ok} verified, {len(recs) - ok} unverified")
        records += recs
    return records
