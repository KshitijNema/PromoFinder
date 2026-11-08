"""Automated checks. They re-read the source files and do not reuse scoring code.

fail = the promo breaks the brief or cannot be traced, and is blocked from ranking.
warn = usable, but an editor should look at the flagged point.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .models import EVENT_CAMERA, HOOK_WINDOW_S, MAX_PROMO_S, MIN_PROMO_S, Candidate, Episode, Evaluation
from .ranking import overlap_ratio

START_ROW = re.compile(r"Start:\s*\(([^)]*)\)")
END_ROW = re.compile(r"End:\s*\(([^)]*)\)")
SCENE_EDGE = 0.05


@lru_cache(maxsize=512)
def _source_lines(path: str) -> tuple[str, ...]:
    return tuple(Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines())


def _seconds(raw: str) -> float:
    value = 0.0
    for part in raw.split(":"):
        value = value * 60 + float(part)
    return round(value, 3)


def _check(results: dict, name: str, status: str, detail: str) -> None:
    results[name] = {"status": status, "detail": detail}


def _clip(episode: Episode, start: float, end: float) -> list:
    return [e for e in episode.events if start - 0.001 <= e.start_s < end - 0.001]


def check_candidate(candidate: Candidate, evaluation: Evaluation, episodes: dict[int, Episode]) -> dict:
    results: dict = {}
    total = candidate.duration
    in_range = MIN_PROMO_S <= total <= MAX_PROMO_S
    _check(results, "duration_20_60s", "pass" if in_range else "fail", f"{total:.2f}s across {len(candidate.segments)} segment(s)")

    missing = [s.episode for s in candidate.segments if s.episode not in episodes]
    _check(results, "valid_episode_reference", "fail" if missing else "pass", f"Unknown: {missing}" if missing else "All episodes exist")
    if missing:
        return results

    order_problems = []
    for position, segment in enumerate(candidate.segments):
        if segment.end_s <= segment.start_s:
            order_problems.append(f"segment {position + 1} ends before it starts")
        if segment.end_s > episodes[segment.episode].duration + 0.01:
            order_problems.append(f"segment {position + 1} runs past the episode end")
    for first, second in zip(candidate.segments, candidate.segments[1:]):
        if (second.episode, second.start_s) < (first.episode, first.end_s):
            order_problems.append("segments are out of story order")
    _check(results, "timestamp_order", "fail" if order_problems else "pass", "; ".join(order_problems) or "Segments ordered")

    trace_problems = []
    for segment in candidate.segments:
        lines = _source_lines(episodes[segment.episode].source_file)
        start_row, end_row = lines[segment.start_line - 1], lines[segment.end_line - 1]
        start_match, end_match = START_ROW.search(start_row), END_ROW.search(end_row)
        if not start_match or start_match.group(1).strip() != segment.start_raw or _seconds(start_match.group(1)) != segment.start_s:
            trace_problems.append(f"episode {segment.episode} line {segment.start_line} start does not match")
        if not end_match or end_match.group(1).strip() != segment.end_raw or _seconds(end_match.group(1)) != segment.end_s:
            trace_problems.append(f"episode {segment.episode} line {segment.end_line} end does not match")
    _check(results, "traceable_to_source", "fail" if trace_problems else "pass", "; ".join(trace_problems) or "Each boundary matches its source line")

    first = candidate.segments[0]
    opening = [e for e in _clip(episodes[first.episode], first.start_s, first.end_s) if e.start_s - first.start_s < HOOK_WINDOW_S]
    audible = [e for e in opening if e.type != EVENT_CAMERA and e.text]
    if audible:
        earliest = min(e.start_s - first.start_s for e in audible)
        _check(results, "hook_within_3s", "pass", f"First line or action starts {earliest:.2f}s in")
    elif opening:
        _check(results, "hook_within_3s", "fail", "Only camera notes in the first 3 seconds")
    else:
        _check(results, "hook_within_3s", "fail", "Nothing starts in the first 3 seconds")

    all_events = [e for s in candidate.segments for e in _clip(episodes[s.episode], s.start_s, s.end_s)]
    spoken = sum(e.is_spoken and bool(e.text) for e in all_events)
    actions = sum(not e.is_spoken and bool(e.text) for e in all_events)
    empty = sum(not e.text for e in all_events)
    status = "fail" if spoken == 0 else "warn" if actions == 0 or empty else "pass"
    _check(results, "dialogue_and_action_present", status, f"{spoken} lines, {actions} visual or sound events, {empty} empty")

    timing_notes = []
    for segment in candidate.segments:
        episode = episodes[segment.episode]
        for event in episode.events:
            if event.is_spoken and event.start_s < segment.start_s - SCENE_EDGE and event.end_s > segment.start_s + SCENE_EDGE:
                timing_notes.append(f"line runs across the start of the episode {segment.episode} segment")
            if event.is_spoken and event.start_s < segment.end_s - SCENE_EDGE and event.end_s > segment.end_s + SCENE_EDGE:
                timing_notes.append(f"line runs across the end of the episode {segment.episode} segment")
        cuts = {round(e.start_s, 2) for e in episode.events if e.type == EVENT_CAMERA} | {round(e.end_s, 2) for e in episode.events if e.type == EVENT_CAMERA}
        if round(segment.start_s, 2) not in cuts:
            timing_notes.append(f"episode {segment.episode} start falls mid-shot")
        if round(segment.end_s, 2) not in cuts:
            timing_notes.append(f"episode {segment.episode} end falls mid-shot")
    _check(results, "timing_precision", "warn" if timing_notes else "pass",
           "; ".join(sorted(set(timing_notes))) or "All cuts land on shot boundaries with no line split")

    if len(candidate.segments) > 1:
        _check(results, "segment_joins", "warn", "Script cannot confirm visual or audio continuity across this join")
    if evaluation.signals.get("resolved_ending"):
        _check(results, "cliffhanger_signal", "warn", "Final moments read as settled")
    else:
        _check(results, "cliffhanger_signal", "pass", "No settled-ending markers in the final moments")
    return results


def duplicate_stats(candidates: list[Candidate], sample_cap: int = 20) -> dict:
    """How much the raw pool repeats itself. Overlap here is expected; the final list must have little."""
    per_key: Counter = Counter(tuple((s.episode, s.start_s, s.end_s) for s in c.segments) for c in candidates)
    exact = sum(count - 1 for count in per_key.values() if count > 1)
    by_episode: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.kind == "continuous":
            by_episode.setdefault(candidate.segments[0].episode, []).append(candidate)
    heavy = compared = 0
    examples = []
    for items in by_episode.values():
        for i, first in enumerate(items):
            for second in items[i + 1:]:
                compared += 1
                if overlap_ratio(first, second) > 0.8:
                    heavy += 1
                    if len(examples) < sample_cap:
                        examples.append([first.id, second.id])
    return {"exact_duplicates": exact, "continuous_pairs_compared": compared, "continuous_pairs_over_80pct_overlap": heavy, "examples": examples}


def run_quality_checks(candidates: list[Candidate], evaluations: list[Evaluation], episodes: list[Episode]) -> dict:
    index = {e.number: e for e in episodes}
    by_id = {e.candidate_id: e for e in evaluations}
    per_candidate = {c.id: check_candidate(c, by_id[c.id], index) for c in candidates}
    summary: dict[str, Counter] = {}
    for checks in per_candidate.values():
        for name, result in checks.items():
            summary.setdefault(name, Counter())[result["status"]] += 1
    blocked = sorted(cid for cid, checks in per_candidate.items() if any(r["status"] == "fail" for r in checks.values()))
    return {"per_candidate": per_candidate, "summary": {k: dict(v) for k, v in summary.items()}, "blocked": blocked,
            "pool_duplicates": duplicate_stats(candidates)}


def check_ranked_list(ranked: list[dict], candidates: list[Candidate], expected_episodes: int, parsed_episodes: int) -> dict:
    by_id = {c.id: c for c in candidates}
    overlaps = []
    for i, first in enumerate(ranked):
        for second in ranked[i + 1:]:
            ratio = overlap_ratio(by_id[first["candidate_id"]], by_id[second["candidate_id"]])
            if ratio > 0.0:
                overlaps.append({"a": first["candidate_id"], "b": second["candidate_id"], "shared_fraction": round(ratio, 3)})
    used = Counter(n for entry in ranked for n in by_id[entry["candidate_id"]].episodes)
    return {"ranked_count": len(ranked), "episodes_expected": expected_episodes, "episodes_parsed": parsed_episodes,
            "distinct_episodes_used": len(used), "most_used_episode_count": max(used.values(), default=0),
            "overlapping_pairs": overlaps, "max_overlap_fraction": max((o["shared_fraction"] for o in overlaps), default=0.0),
            "situations": dict(Counter(entry["situation"] for entry in ranked))}
