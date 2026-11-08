"""Choose a varied posting list from the scored candidates."""
from __future__ import annotations

from collections import Counter

from .models import Candidate, Evaluation

DEFAULT_TOP_N = 30
MAX_PER_EPISODE = 2
MAX_OVERLAP = 0.30
LATE_ARC_FRACTION = 0.75
EARLY_ARC_FRACTION = 0.20


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Shared seconds as a share of the shorter promo."""
    shared = 0.0
    for first in a.segments:
        for second in b.segments:
            if first.episode == second.episode:
                shared += max(0.0, min(first.end_s, second.end_s) - max(first.start_s, second.start_s))
    return shared / min(a.duration, b.duration)


def rank_candidates(candidates: list[Candidate], evaluations: list[Evaluation], blocked: set[str], last_episode: int,
                    top_n: int = DEFAULT_TOP_N) -> list[dict]:
    """Greedy selection: take the best remaining promo, after penalties for repeating what is already chosen."""
    by_id = {c.id: c for c in candidates}
    pool = [e for e in evaluations if e.candidate_id not in blocked]
    chosen: list[dict] = []
    episode_uses: Counter = Counter()
    situation_uses: Counter = Counter()
    cast_uses: Counter = Counter()
    while pool and len(chosen) < top_n:
        best, best_value, best_notes = None, float("-inf"), []
        for evaluation in pool:
            candidate = by_id[evaluation.candidate_id]
            if any(episode_uses[n] >= MAX_PER_EPISODE for n in candidate.episodes):
                continue
            if any(overlap_ratio(candidate, by_id[p["candidate_id"]]) > MAX_OVERLAP for p in chosen):
                continue
            value = evaluation.overall
            notes: list[str] = []
            repeat = sum(episode_uses[n] for n in candidate.episodes)
            if repeat:
                value -= 8.0 * repeat
                notes.append(f"-{8 * repeat} episode already used")
            neighbours = sum(episode_uses[n + d] for n in candidate.episodes for d in (-1, 1))
            if neighbours:
                value -= 1.5 * neighbours
                notes.append(f"-{1.5 * neighbours} adjacent episode used")
            crowded = max(0, situation_uses[evaluation.situation] - 1)
            if crowded:
                value -= 2.5 * crowded
                notes.append(f"-{2.5 * crowded} situation already used")
            cast = tuple(sorted(evaluation.signals.get("speakers", []))[:3])
            if cast_uses[cast]:
                value -= 1.0 * cast_uses[cast]
                notes.append(f"-{cast_uses[cast]} same speakers already used")
            if max(candidate.episodes) > LATE_ARC_FRACTION * last_episode:
                value -= 4.0
                notes.append("-4 late-season spoiler risk")
            if max(candidate.episodes) <= max(5, EARLY_ARC_FRACTION * last_episode):
                value += 5.0
                notes.append("+5 early story: new viewers can start the show here")
            if len(candidate.episodes) > 1:
                value -= 6.0
                notes.append("-6 cross-episode join risk")
            elif len(candidate.segments) > 1:
                value -= 4.0
                notes.append("-4 same-episode join risk")
            if value > best_value:
                best, best_value, best_notes = evaluation, value, notes
        if best is None:
            break
        candidate = by_id[best.candidate_id]
        pool.remove(best)
        for number in candidate.episodes:
            episode_uses[number] += 1
        situation_uses[best.situation] += 1
        cast_uses[tuple(sorted(best.signals.get("speakers", []))[:3])] += 1
        chosen.append({"rank": len(chosen) + 1, "candidate_id": best.candidate_id, "rank_score": round(best_value, 1),
                       "overall": best.overall, "adjustments": best_notes, "situation": best.situation})
    for entry, reason in zip(chosen, _post_first_reasons(chosen, by_id, {e.candidate_id: e for e in evaluations}, last_episode)):
        entry["post_order_reason"] = reason
    return chosen


def _post_first_reasons(chosen: list[dict], by_id: dict, evaluations: dict, last_episode: int) -> list[str]:
    reasons = []
    for entry in chosen:
        evaluation, candidate = evaluations[entry["candidate_id"]], by_id[entry["candidate_id"]]
        parts = [f"Hook {evaluation.hook:.0f}, cliffhanger {evaluation.cliffhanger:.0f}, standalone {evaluation.standalone:.0f}"]
        first = min(candidate.episodes)
        if first <= max(5, 0.2 * last_episode):
            parts.append("comes early, so a new viewer can start the show near this point")
        elif max(candidate.episodes) > LATE_ARC_FRACTION * last_episode:
            parts.append("hold for later; it risks spoiling a late payoff")
        else:
            parts.append("mid-season escalation that adds variety after the opening promos")
        if evaluation.signals.get("backstory_debt"):
            parts.append("needs an editor check on unfamiliar names")
        reasons.append("; ".join(parts) + ".")
    return reasons
