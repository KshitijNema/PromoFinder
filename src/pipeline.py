"""Run every stage end to end and write the raw outputs."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import __version__
from .candidate_generator import generate_candidates
from .evaluator import BaseEvaluator, HeuristicEvaluator
from .models import Candidate, Evaluation
from .parser import apply_character_aliases, format_time, load_aliases, parse_folder, safe_extract
from .quality_checks import check_ranked_list, run_quality_checks
from .ranking import DEFAULT_TOP_N, rank_candidates
from .scorer import WEIGHTS, analyze_show

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "outputs"
DEFAULT_ALIASES = ROOT / "data" / "aliases.json"
EXPECTED_ROWS = {"RANKED": "04_ranked_promos"}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _excerpt(candidate: Candidate, episodes: dict) -> list[dict]:
    """Script lines inside the promo, each with its original timestamps and source line."""
    output = []
    for number, segment in enumerate(candidate.segments, 1):
        for event in episodes[segment.episode].events:
            if segment.start_s - 0.001 <= event.start_s < segment.end_s - 0.001:
                output.append({"segment": number, "episode": event.episode, "type": event.type, "character": event.character,
                               "tones": event.tones, "text": event.text, "start": event.start_raw, "end": event.end_raw,
                               "offset_s": round(event.start_s - segment.start_s, 2), "source_line": event.line_no})
    return output


def _segments_text(candidate: Candidate) -> str:
    return " + ".join(f"Ep{s.episode} {s.start_raw}-{s.end_raw}" for s in candidate.segments)


def _hash_inputs(files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def run_pipeline(input_path: Path | str = DEFAULT_INPUT, output_dir: Path | str = DEFAULT_OUTPUT,
                 evaluator: BaseEvaluator | None = None, aliases_path: Path | str | None = DEFAULT_ALIASES,
                 top_n: int = DEFAULT_TOP_N, expected_episodes: int | None = None,
                 progress: Callable[[str, float], None] | None = None) -> dict:
    """Parse, generate, evaluate, check and rank. Returns the run manifest."""
    report = progress or (lambda message, fraction: None)
    started = time.perf_counter()
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        input_path = safe_extract(input_path, output_dir / "_extracted")
    evaluator = evaluator or HeuristicEvaluator()

    report("Parsing screenplays", 0.05)
    episodes, failures = parse_folder(input_path)
    if not episodes:
        raise ValueError(f"No readable screenplays under {input_path}. Failures: {failures}")
    apply_character_aliases(episodes, load_aliases(Path(aliases_path) if aliases_path else None))
    by_number = {e.number: e for e in episodes}
    span = range(min(by_number), max(by_number) + 1)
    missing = [n for n in span if n not in by_number]
    _write_json(output_dir / "01_parsed_episodes.json", [e.to_dict() for e in episodes])

    report("Analysing story signals", 0.2)
    ctx = analyze_show(episodes)
    issue_counts = Counter(i["code"] for e in episodes for i in e.issues)
    _write_json(output_dir / "01_parse_issues.json", {
        "files_failed": failures, "missing_episode_numbers": missing, "issue_counts": dict(issue_counts),
        "spelling_variant_groups": sorted({tuple(v) for v in ctx.name_variants.values()}),
        "speaker_labels": dict(Counter(ev.label_raw.split("(")[0].strip() for e in episodes for ev in e.events if ev.is_spoken)),
        "per_episode": {e.number: e.issues for e in episodes}})

    report("Generating candidates", 0.35)
    candidates = generate_candidates(episodes, ctx)
    _write_json(output_dir / "02_candidates.json", [asdict(c) | {"duration_s": c.duration} for c in candidates])

    report(f"Evaluating {len(candidates)} candidates", 0.5)
    evaluations = evaluator.evaluate(candidates, ctx)
    _write_json(output_dir / "03_scores.json", [asdict(e) for e in evaluations])
    by_id = {c.id: c for c in candidates}
    pd.DataFrame([{"candidate_id": e.candidate_id, "kind": by_id[e.candidate_id].kind, "segments": _segments_text(by_id[e.candidate_id]),
                   "duration_s": by_id[e.candidate_id].duration, "overall": e.overall, "hook": e.hook, "cliffhanger": e.cliffhanger,
                   "emotion": e.emotion, "standalone": e.standalone, "curiosity": e.curiosity, "situation": e.situation, "method": e.method}
                  for e in evaluations]).to_csv(output_dir / "03_scores.csv", index=False, encoding="utf-8-sig")

    report("Running quality checks", 0.7)
    quality = run_quality_checks(candidates, evaluations, episodes)
    report("Ranking", 0.85)
    ranked = rank_candidates(candidates, evaluations, set(quality["blocked"]), ctx.last_episode, top_n=top_n)
    expected = expected_episodes if expected_episodes is not None else len(span)
    quality["final_list"] = check_ranked_list(ranked, candidates, expected, len(episodes))
    _write_json(output_dir / "05_quality_report.json", quality)

    evaluation_index = {e.candidate_id: e for e in evaluations}
    full_ranked, flat = [], []
    for entry in ranked:
        candidate, evaluation = by_id[entry["candidate_id"]], evaluation_index[entry["candidate_id"]]
        checks = quality["per_candidate"][candidate.id]
        statuses = Counter(r["status"] for r in checks.values())
        full_ranked.append({
            **entry, "kind": candidate.kind, "duration_s": candidate.duration, "episodes": candidate.episodes,
            "segments": [asdict(s) | {"order": i, "duration_s": s.duration} for i, s in enumerate(candidate.segments, 1)],
            "scores": {"hook": evaluation.hook, "cliffhanger": evaluation.cliffhanger, "emotion": evaluation.emotion,
                       "standalone": evaluation.standalone, "curiosity": evaluation.curiosity, "overall": evaluation.overall},
            "weights": WEIGHTS, "hook_text": evaluation.hook_text, "cliffhanger_text": evaluation.cliffhanger_text,
            "hook_explanation": evaluation.hook_explanation, "cliffhanger_explanation": evaluation.cliffhanger_explanation,
            "standalone_explanation": evaluation.standalone_explanation, "why_new_viewer": evaluation.why_new_viewer,
            "why_it_works": evaluation.why_it_works, "weaknesses": evaluation.weaknesses, "method": evaluation.method,
            "justification": candidate.justification, "flags": candidate.flags, "signals": evaluation.signals,
            "quality_checks": checks, "excerpt": _excerpt(candidate, by_number)})
        flat.append({
            "rank": entry["rank"], "candidate_id": candidate.id, "episodes": ",".join(map(str, candidate.episodes)), "kind": candidate.kind,
            "segments": _segments_text(candidate), "duration_s": candidate.duration, "overall": evaluation.overall, "rank_score": entry["rank_score"],
            "hook": evaluation.hook, "cliffhanger": evaluation.cliffhanger, "emotion": evaluation.emotion, "standalone": evaluation.standalone,
            "curiosity": evaluation.curiosity, "situation": evaluation.situation, "hook_text": evaluation.hook_text, "cliffhanger_text": evaluation.cliffhanger_text,
            "why_it_works": evaluation.why_it_works, "post_order_reason": entry["post_order_reason"], "weaknesses": " | ".join(evaluation.weaknesses),
            "qc_warnings": statuses.get("warn", 0), "qc_failures": statuses.get("fail", 0), "method": evaluation.method})
    _write_json(output_dir / "04_ranked_promos.json", full_ranked)
    pd.DataFrame(flat).to_csv(output_dir / "04_ranked_promos.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "promofinder_version": __version__, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(),
        "input": str(input_path), "input_sha256": _hash_inputs([e.source_file for e in episodes]), "evaluator": evaluator.name,
        "method": evaluations[0].method if evaluations else "", "episodes_parsed": len(episodes), "files_failed": len(failures), "missing_episode_numbers": missing,
        "events_parsed": sum(len(e.events) for e in episodes), "scenes_parsed": sum(len(e.scenes) for e in episodes), "total_runtime_s": round(sum(e.duration for e in episodes), 1),
        "candidates": len(candidates), "candidates_by_kind": dict(Counter(c.kind for c in candidates)), "blocked_by_quality_checks": len(quality["blocked"]),
        "ranked": len(ranked), "weights": WEIGHTS, "seconds_elapsed": round(time.perf_counter() - started, 2),
        "first_promo": format_time(by_id[ranked[0]["candidate_id"]].segments[0].start_s) if ranked else None}
    _write_json(output_dir / "run_manifest.json", manifest)
    report("Done", 1.0)
    return manifest
