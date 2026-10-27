"""Tamper tests for the hash-chained consent ledger.   Run: python -m pytest"""
import json

import pytest

import consent


@pytest.fixture
def log(tmp_path, monkeypatch):
    # Point the ledger at a temporary file so tests never touch store/.
    monkeypatch.setattr(consent, "LOG", tmp_path / "log.jsonl")
    return consent.LOG


def test_clean_chain_verifies(log):
    consent.log_event("consent_granted", "expert", "clip1")
    consent.log_event("capture_stopped", "expert", "clip1")
    assert consent.verify() == (True, 2)


def test_edit_breaks_verification(log):
    consent.log_event("consent_granted", "expert", "clip1")
    consent.log_event("capture_stopped", "expert", "clip1")
    lines = log.read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[0])
    e["who"] = "someone else"                      # tamper with the first entry
    lines[0] = json.dumps(e)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, bad = consent.verify()
    assert not ok and bad == 1


def test_deleted_line_breaks_verification(log):
    for ev in ("consent_granted", "capture_stopped", "records_published"):
        consent.log_event(ev, "expert", "clip1")
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")  # drop the middle line
    ok, bad = consent.verify()
    assert not ok and bad == 2


def test_refusal_is_logged_and_raises(log):
    with pytest.raises(consent.ConsentRequired):
        consent.require_consent(False, "expert", "clip1")
    with pytest.raises(consent.ConsentRequired):
        consent.require_consent(True, "   ", "clip1")      # consent without a name is refused
    events = [json.loads(l)["event"] for l in log.read_text(encoding="utf-8").splitlines()]
    assert events == ["capture_refused", "capture_refused"]
    assert consent.verify() == (True, 2)


def test_tampered_log_blocks_new_capture(log):
    consent.log_event("consent_granted", "expert", "clip1")
    log.write_text(log.read_text(encoding="utf-8").replace("expert", "x"), encoding="utf-8")
    with pytest.raises(RuntimeError):
        consent.require_consent(True, "expert", "clip2")
