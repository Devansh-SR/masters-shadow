"""Streamlit demo: Ask, Capture, Gaps, Expert review, Ledger.   streamlit run app.py"""
import pathlib

import streamlit as st

import consent
import gaps
from config import THRESHOLD
from consent import ConsentRequired
from index import delete_record
from query import load, search
from run_pipeline import run

CLIPS = pathlib.Path("data/clips")

st.set_page_config(page_title="The Master's Shadow", layout="wide")
st.title("The Master's Shadow")
st.caption("Stage I prototype: runs offline on this laptop. Nothing runs on Hailo hardware yet.")
tab_ask, tab_cap, tab_gaps, tab_rev, tab_led = st.tabs(
    ["Ask", "Capture", "Gaps", "Expert review", "Ledger"])


def _clear_answer():
    """Drop the cached answer when the index changes, so a stale answer is never shown."""
    st.session_state.pop("res", None)
    st.session_state.pop("res_q", None)


# ---------------- Ask ----------------
with tab_ask:
    # A form only submits when the button is pressed, so Streamlit reruns
    # (any click elsewhere) never search again or log duplicate gaps.
    with st.form("ask"):
        q = st.text_input("Ask a question", placeholder="Why is the pressure needle oscillating?")
        asked = st.form_submit_button("Ask")
    if asked and q.strip():
        try:
            records, vecs = load()
            st.session_state["res"] = search(q.strip(), records, vecs)
        except FileNotFoundError:
            st.session_state["res"] = None
        st.session_state["res_q"] = q.strip()

    if "res_q" in st.session_state:
        res = st.session_state.get("res")
        st.write(f"**Question:** {st.session_state['res_q']}")
        if res is None:
            st.warning("No records yet. Run a capture first.")
        elif res["abstain"]:
            st.warning("No expert has covered this yet. It has been logged for the expert.")
            if res["hits"]:
                h = res["hits"][0]
                # Shown only as a label, never as the answer.
                st.caption(f"Closest record is BELOW the confidence threshold ({THRESHOLD:.2f}) "
                           f"and is not an answer: \"{h['step']}\" (score {h['score']:.2f})")
        else:
            for n, h in enumerate(res["hits"], 1):
                below = h["score"] < THRESHOLD
                st.subheader(f"{n}. {h['step']}" + ("  (below threshold)" if below else ""))
                if h.get("sample"):
                    st.info("SAMPLE record: hand-written for the fallback demo, not extracted by the system.")
                st.write(f"**Component:** {h['component']}")
                st.write(f"**Why:** {h['rationale'] or 'not stated'}")
                st.write(f"**A fault looks like:** {h['failure_indicator'] or 'not stated'}")
                st.caption(f"Expert's words: \"{h['evidence']}\"  |  {h['start']:.0f}s to "
                           f"{h['end']:.0f}s  |  score {h['score']:.2f}  |  {h['source']}")
                path = CLIPS / h["source"]
                if path.exists():
                    # The video opens at the start of the window this record came from.
                    st.video(str(path), start_time=int(h["start"]))

# ---------------- Capture ----------------
with tab_cap:
    clips = sorted(CLIPS.glob("*.mp4"))
    if not clips:
        st.info("Put an .mp4 with spoken narration in data/clips/ and reload.")
    clip = st.selectbox("Clip", clips, format_func=lambda p: p.name)
    who = st.text_input("Expert name")
    agreed = st.checkbox("I consent to this session being captured and indexed")
    if st.button("Start capture") and clip is not None:
        try:
            with st.spinner("Transcribing, structuring and indexing (a few minutes)..."):
                good, bad = run(clip, who, agreed)
            _clear_answer()
            st.success(f"{good} records indexed, {bad} unverified (kept out of search).")
        except ConsentRequired as e:
            st.error(str(e))                   # the refusal is already in the ledger
        except Exception as e:
            st.exception(e)

# ---------------- Gaps ----------------
with tab_gaps:
    groups = gaps.clusters()
    if not groups:
        st.info("No gaps logged yet.")
    for c in groups:
        st.write(f"**{c['count']}x** {c['example']}")
        with st.expander("All wordings"):
            for qq in c["questions"]:
                st.write(f"- {qq}")

# ---------------- Expert review ----------------
with tab_rev:
    try:
        records, _ = load()
    except FileNotFoundError:
        records = []
    if not records:
        st.info("No records indexed yet.")
    reviewer = st.text_input("Reviewer name", key="reviewer")
    for r in records:
        c1, c2 = st.columns([6, 1])
        tag = " (SAMPLE)" if r.get("sample") else ""
        c1.write(f"**{r['component']}**: {r['step']}{tag}  \n"
                 f"_\"{r['evidence']}\"_  ({r['start']:.0f}s, {r['source']})")
        if c2.button("Delete", key=f"del-{r['id']}"):
            if not reviewer.strip():
                st.warning("Enter the reviewer's name first.")
            else:
                delete_record(r["id"], reviewer.strip())   # removes record + vector, logs it
                _clear_answer()
                st.rerun()

# ---------------- Ledger ----------------
with tab_led:
    ok, info = consent.verify()
    if ok:
        st.success(f"Ledger verified: {info} entries, hash chain intact.")
    else:
        st.error(f"Ledger FAILED verification at line {info}.")
    st.caption("The chain shows the log was not edited after the fact. It does not prove who clicked consent.")
    if consent.LOG.exists():
        st.code("\n".join(consent.LOG.read_text(encoding="utf-8").splitlines()[-10:]))
