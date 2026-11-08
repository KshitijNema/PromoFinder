"""PromoFinder dashboard. Run with: streamlit run app.py"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluator import build_evaluator
from src.pipeline import DEFAULT_ALIASES, DEFAULT_INPUT, DEFAULT_OUTPUT, run_pipeline

st.set_page_config(page_title="PromoFinder", page_icon=":clapper:", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .pf-sub {color: #5f6673; margin-top: -0.6rem; margin-bottom: 1rem;}
    .pf-quote {border-left: 3px solid #2a78d6; padding: 0.15rem 0 0.15rem 0.7rem; margin: 0.2rem 0 0.7rem 0;}
    .pf-label {font-size: 0.78rem; font-weight: 600; letter-spacing: .03em; color: #5f6673; text-transform: uppercase;}
    @media (prefers-color-scheme: dark) { .pf-sub, .pf-label {color: #a4abb8;} }
    </style>""",
    unsafe_allow_html=True,
)

STATUS_TEXT = {"pass": "✅ Pass", "warn": "⚠️ Review", "fail": "❌ Fail"}
SCORE_LABELS = [("hook", "Hook", 25), ("cliffhanger", "Cliffhanger", 30), ("emotion", "Emotion", 20), ("standalone", "Standalone", 15), ("curiosity", "Curiosity", 10)]
FILES = ["run_manifest.json", "04_ranked_promos.json", "05_quality_report.json", "01_parse_issues.json"]


def signature(folder: Path) -> tuple:
    return tuple((name, (folder / name).stat().st_mtime_ns if (folder / name).exists() else 0) for name in FILES)


@st.cache_data(show_spinner=False)
def load_run(folder: str, _key: tuple) -> dict | None:
    path = Path(folder)
    if not all((path / name).exists() for name in FILES):
        return None
    read = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    review_path = path / "06_review_results.json"
    return {"manifest": read(FILES[0]), "ranked": read(FILES[1]), "quality": read(FILES[2]), "issues": read(FILES[3]),
            "review": json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None}


def label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def segment_text(promo: dict) -> str:
    return " + ".join(f"Ep {s['episode']} {s['start_raw']}–{s['end_raw']}" for s in promo["segments"])


def execute(input_path, output_dir, aliases, evaluator_name) -> None:
    bar = st.sidebar.progress(0.0, "Starting")
    try:
        manifest = run_pipeline(input_path, output_dir, build_evaluator(evaluator_name), aliases, progress=lambda m, f: bar.progress(f, m))
    except Exception as exc:  # shown to the user instead of a stack trace
        bar.empty()
        st.sidebar.error(f"Run failed: {exc}")
        return
    bar.empty()
    st.session_state["output_dir"] = str(output_dir)
    st.sidebar.success(f"Done: {manifest['episodes_parsed']} episodes, {manifest['candidates']:,} candidates, {manifest['seconds_elapsed']}s")
    load_run.clear()


st.sidebar.title("Run analysis")
source = st.sidebar.radio("Screenplays", ["Project folder", "Upload a ZIP"], help="A new show can be analysed by uploading a ZIP of .txt scripts.")
uploaded = st.sidebar.file_uploader("ZIP of .txt scripts", type="zip") if source == "Upload a ZIP" else None
options = {"Rules baseline (free, no AI model)": "heuristic"}
if os.environ.get("ANTHROPIC_API_KEY"):
    options["Rules + Claude re-score (paid)"] = "anthropic"
if os.environ.get("PROMOFINDER_LLM_BASE_URL") and os.environ.get("PROMOFINDER_MODEL"):
    options["Rules + other LLM re-score"] = "openai_compatible"
choice = st.sidebar.selectbox("Evaluator", list(options), help="LLM options appear when their environment variables are set.")
if st.sidebar.button("Run pipeline", type="primary", width="stretch"):
    if source == "Upload a ZIP":
        if uploaded is None:
            st.sidebar.warning("Choose a ZIP first.")
        else:
            target = DEFAULT_OUTPUT / "uploads" / Path(uploaded.name).stem
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=target) as handle:
                handle.write(uploaded.getvalue())
            execute(handle.name, target, None, options[choice])
            Path(handle.name).unlink(missing_ok=True)
    else:
        execute(DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_ALIASES if DEFAULT_ALIASES.exists() else None, options[choice])

folder = Path(st.session_state.get("output_dir", str(DEFAULT_OUTPUT)))
run = load_run(str(folder), signature(folder))

st.title("PromoFinder")
st.markdown('<p class="pf-sub">Finds and ranks 20–60 second promos in timestamped microdrama screenplays.</p>', unsafe_allow_html=True)

if run is None:
    st.info("No results yet. Put the scripts in data/raw, or upload a ZIP, then choose **Run pipeline** in the sidebar.")
    st.stop()

manifest, ranked, quality, issues = run["manifest"], run["ranked"], run["quality"], run["issues"]
st.caption(f"Method: {manifest['method']}. Scores order candidates for an editor and do not predict views. Results contain confidential script text, so keep this page private.")

overview, leaderboard, checks, exports = st.tabs(["Overview", "Promo leaderboard", "Quality checks", "Export"])

with overview:
    columns = st.columns(4)
    columns[0].metric("Episodes processed", manifest["episodes_parsed"])
    columns[1].metric("Candidates generated", f"{manifest['candidates']:,}")
    columns[2].metric("Ranked promos", manifest["ranked"])
    columns[3].metric("Blocked by checks", manifest["blocked_by_quality_checks"])
    if manifest["files_failed"] or manifest["missing_episode_numbers"]:
        st.warning(f"Files that could not be read: {manifest['files_failed']}. Missing episode numbers: {manifest['missing_episode_numbers'] or 'none'}.")
    st.write(f"Parsed **{manifest['events_parsed']:,}** timed events in **{manifest['scenes_parsed']}** scenes, covering "
             f"**{manifest['total_runtime_s'] / 60:.1f} minutes** of script. Last run {manifest['run_utc']} UTC in {manifest['seconds_elapsed']}s.")
    kinds = manifest["candidates_by_kind"]
    st.write("Candidates by type: " + ", ".join(f"{label(k)} **{v:,}**" for k, v in kinds.items()) + ".")
    left, right = st.columns(2)
    per_episode = pd.DataFrame([{"Episode": e, "Promos": 1} for p in ranked for e in p["episodes"]]).groupby("Episode", as_index=False).sum()
    with left:
        st.subheader("Episodes in the ranked list")
        chart = alt.Chart(per_episode).mark_bar(color="#2a78d6", cornerRadiusEnd=4, size=9).encode(
            x=alt.X("Episode:Q", scale=alt.Scale(domain=[0, max(manifest["episodes_parsed"], int(per_episode["Episode"].max())) + 1]), axis=alt.Axis(grid=False, tickMinStep=5)),
            y=alt.Y("Promos:Q", axis=alt.Axis(values=list(range(0, int(per_episode["Promos"].max()) + 1)), format="d", gridOpacity=0.3)), tooltip=["Episode", "Promos"]).properties(height=240)
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Story situations in the ranked list")
        situations = pd.DataFrame([{"Situation": label(k), "Promos": v} for k, v in quality["final_list"]["situations"].items()]).sort_values("Promos", ascending=False)
        chart = alt.Chart(situations).mark_bar(color="#2a78d6", cornerRadiusEnd=4, size=14).encode(
            y=alt.Y("Situation:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)), x=alt.X("Promos:Q", axis=alt.Axis(tickMinStep=1, format="d", gridOpacity=0.3)), tooltip=["Situation", "Promos"]).properties(height=240)
        st.altair_chart(chart, width="stretch")
    st.subheader("Post these first")
    for promo in ranked[:3]:
        st.markdown(f"**#{promo['rank']} · {segment_text(promo)} · score {promo['scores']['overall']:.0f}.** {promo['post_order_reason']}")

with leaderboard:
    all_episodes = sorted({e for p in ranked for e in p["episodes"]})
    a, b, c = st.columns([2, 1, 2])
    chosen = a.multiselect("Episode", all_episodes, placeholder="All episodes")
    minimum = b.slider("Minimum score", 0, 100, 0)
    kinds_available = sorted({p["kind"] for p in ranked})
    kind_filter = c.multiselect("Promo type", kinds_available, format_func=label, placeholder="All types")
    visible = [p for p in ranked if (not chosen or set(p["episodes"]) & set(chosen)) and p["scores"]["overall"] >= minimum and (not kind_filter or p["kind"] in kind_filter)]
    st.session_state["visible"] = visible
    st.write(f"Showing **{len(visible)}** of {len(ranked)} promos.")
    if visible:
        table = pd.DataFrame([{"Rank": p["rank"], "Timestamps": segment_text(p), "Length (s)": round(p["duration_s"]), "Score": p["scores"]["overall"], "Hook": p["scores"]["hook"],
                               "Cliffhanger": p["scores"]["cliffhanger"], "Emotion": p["scores"]["emotion"], "Standalone": p["scores"]["standalone"], "Curiosity": p["scores"]["curiosity"], "Situation": label(p["situation"])} for p in visible])
        st.dataframe(table, hide_index=True, width="stretch")
    for promo in visible:
        with st.container(border=True):
            head, score = st.columns([5, 1])
            head.markdown(f"#### #{promo['rank']} · {label(promo['situation'])}")
            head.caption(f"{promo['candidate_id']} · {label(promo['kind'])} · {promo['duration_s']:.1f}s")
            score.metric("Score", f"{promo['scores']['overall']:.1f}")
            st.dataframe(pd.DataFrame([{"Order": s["order"], "Episode": s["episode"], "Start": s["start_raw"], "End": s["end_raw"], "Seconds": s["duration_s"], "Script line (start)": s["start_line"], "Script line (end)": s["end_line"]} for s in promo["segments"]]), hide_index=True, width="stretch")
            left, right = st.columns(2)
            left.markdown(f'<div class="pf-label">Hook</div><div class="pf-quote">{promo["hook_text"]}</div>', unsafe_allow_html=True)
            right.markdown(f'<div class="pf-label">Cliffhanger</div><div class="pf-quote">{promo["cliffhanger_text"]}</div>', unsafe_allow_html=True)
            st.markdown(f"**Why it works:** {promo['why_it_works']}")
            st.caption(f"Posting note: {promo['post_order_reason']}")
            with st.expander("Scores, explanations, script and checks"):
                for column, (key, name, weight) in zip(st.columns(5), SCORE_LABELS):
                    column.markdown(f"**{name}** ({weight}%)  \n{promo['scores'][key]:.0f} / 100")
                    column.progress(promo["scores"][key] / 100)
                st.markdown(f"**Hook:** {promo['hook_explanation']}")
                st.markdown(f"**Cliffhanger:** {promo['cliffhanger_explanation']}")
                st.markdown(f"**Standalone context:** {promo['standalone_explanation']}")
                st.markdown(f"**Why a new viewer would stop:** {promo['why_new_viewer']}")
                if promo["adjustments"]:
                    st.markdown("**Ranking adjustments:** " + "; ".join(promo["adjustments"]))
                if promo["kind"] != "continuous":
                    st.markdown(f"**Why segments are joined:** {promo['justification']}")
                st.markdown("**Weaknesses:**")
                st.markdown("\n".join(f"- {w}" for w in promo["weaknesses"]) or "- None detected by the rules")
                st.markdown("**Dialogue and action:**")
                st.dataframe(pd.DataFrame([{"Seg": x["segment"], "Time": x["start"], "Type": x["type"], "Who": x["character"].title(), "Text": x["text"]} for x in promo["excerpt"] if x["type"] != "camera"]), hide_index=True, width="stretch")
                st.markdown("**Quality checks:**")
                st.dataframe(pd.DataFrame([{"Check": label(k), "Result": STATUS_TEXT[v["status"]], "Detail": v["detail"]} for k, v in promo["quality_checks"].items()]), hide_index=True, width="stretch")

with checks:
    st.subheader("All candidates")
    rows = [{"Check": label(name), "Pass": counts.get("pass", 0), "Review": counts.get("warn", 0), "Fail": counts.get("fail", 0)} for name, counts in quality["summary"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    final = quality["final_list"]
    st.subheader("Final ranked list")
    a, b, c = st.columns(3)
    a.metric("Distinct episodes used", final["distinct_episodes_used"])
    b.metric("Most promos from one episode", final["most_used_episode_count"])
    c.metric("Largest overlap between promos", f"{final['max_overlap_fraction']:.0%}")
    pool = quality["pool_duplicates"]
    st.write(f"The raw pool has {pool['exact_duplicates']} exact duplicates and {pool['continuous_pairs_over_80pct_overlap']:,} heavily overlapping pairs out of {pool['continuous_pairs_compared']:,} compared. Overlap is expected before ranking, and the ranker removes it.")
    if run["review"]:
        st.subheader("Manual-style review")
        review = run["review"]
        st.write(review["headline"])
        st.dataframe(pd.DataFrame(review["table"]), hide_index=True, width="stretch")
    st.subheader("Script format issues found while parsing")
    st.dataframe(pd.DataFrame([{"Issue": label(k), "Count": v} for k, v in issues["issue_counts"].items()]), hide_index=True, width="stretch")
    if issues["spelling_variant_groups"]:
        st.write("Possible spelling drift in spoken names:")
        st.markdown("\n".join(f"- {', '.join(group)}" for group in issues["spelling_variant_groups"]))

with exports:
    st.write("Exports follow the filters on the leaderboard. They contain script text, so share them only with the show's team.")
    selected = st.session_state.get("visible", ranked)
    flat = pd.DataFrame([{"rank": p["rank"], "candidate_id": p["candidate_id"], "kind": p["kind"], "episodes": ",".join(map(str, p["episodes"])), "segments": segment_text(p), "duration_s": p["duration_s"], **p["scores"], "situation": p["situation"], "hook_text": p["hook_text"], "cliffhanger_text": p["cliffhanger_text"], "why_it_works": p["why_it_works"], "post_order_reason": p["post_order_reason"], "weaknesses": " | ".join(p["weaknesses"]), "method": p["method"]} for p in selected])
    a, b = st.columns(2)
    a.download_button("Download CSV", flat.to_csv(index=False).encode("utf-8-sig"), "promofinder_ranked.csv", "text/csv", width="stretch")
    b.download_button("Download JSON", json.dumps(selected, indent=1, ensure_ascii=False).encode("utf-8"), "promofinder_ranked.json", "application/json", width="stretch")
    st.caption(f"Raw outputs for this run are in {folder}.")
