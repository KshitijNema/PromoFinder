"""Turn each episode into 20-60 second candidate promos with exact, traceable boundaries."""
from __future__ import annotations

from itertools import product

from .models import (EVENT_CAMERA, MAX_PROMO_S, MIN_PROMO_S, Candidate, Episode, Event, Segment)
from .scorer import ShowContext

HOOKS_PER_EPISODE = 12
ENDS_PER_EPISODE = 14
KEEP_CONTINUOUS = 50
KEEP_MONTAGE = 14
KEEP_CROSS = 10
TOLERANCE = 0.05


def _shot_at(episode: Episode, moment: float) -> Event | None:
    return next((e for e in episode.events if e.type == EVENT_CAMERA and e.start_s <= moment < e.end_s), None)


def _straddles_start(episode: Episode, moment: float) -> bool:
    return any(e.is_spoken and e.start_s < moment - TOLERANCE and e.end_s > moment + 0.25 for e in episode.events)


def _straddles_end(episode: Episode, moment: float) -> bool:
    return any(e.is_spoken and e.start_s < moment - TOLERANCE and e.end_s > moment + 0.10 for e in episode.events)


def snap_start(episode: Episode, event: Event) -> Event | None:
    """Prefer the cut that opens the hook's shot, unless that delays the hook too long."""
    shot = _shot_at(episode, event.start_s + 0.001)
    if shot is not None and event.start_s - shot.start_s <= 1.5 and not _straddles_start(episode, shot.start_s):
        return shot
    return None if _straddles_start(episode, event.start_s) else event


def snap_end(episode: Episode, event: Event) -> Event | None:
    """Hold to the end of the shot when it adds a reaction, but never into the next line."""
    shot = _shot_at(episode, max(event.start_s, event.end_s - 0.01))
    next_line = next((e for e in episode.events if e.is_spoken and e.start_s >= event.end_s - TOLERANCE and e is not event), None)
    if shot is not None and 0 <= shot.end_s - event.end_s <= 2.5:
        clear = next_line is None or next_line.start_s >= shot.end_s - TOLERANCE
        if clear and not _straddles_end(episode, shot.end_s):
            return shot
    return None if _straddles_end(episode, event.end_s) else event


def make_segment(episode: Episode, opener: Event, closer: Event) -> Segment:
    return Segment(episode=episode.number, start_s=opener.start_s, end_s=closer.end_s, start_raw=opener.start_raw,
                   end_raw=closer.end_raw, start_line=opener.line_no, end_line=closer.line_no,
                   first_event=opener.index, last_event=closer.index)


def _top_hooks(episode: Episode, ctx: ShowContext, limit: int) -> list[Event]:
    beats = ctx.beats[episode.number]
    ranked = sorted((e for e in episode.events if e.type != EVENT_CAMERA and e.text),
                    key=lambda e: (-round(beats[e.index].hook, 1), e.start_s))  # on near-ties, take the earlier moment
    chosen: list[Event] = []
    for event in ranked:
        if all(abs(event.start_s - other.start_s) >= 2.0 for other in chosen):
            chosen.append(event)
        if len(chosen) >= limit:
            break
    return chosen


def _top_ends(episode: Episode, ctx: ShowContext, limit: int) -> list[Event]:
    beats = ctx.beats[episode.number]
    ranked = sorted((e for e in episode.events if e.text), key=lambda e: beats[e.index].cliff, reverse=True)
    chosen: list[Event] = []
    for event in ranked:
        if all(abs(event.end_s - other.end_s) >= 1.5 for other in chosen):
            chosen.append(event)
        if len(chosen) >= limit:
            break
    last = max(episode.events, key=lambda e: e.end_s)
    if last not in chosen:
        chosen.append(last)
    return chosen


def _has_dialogue(episode: Episode, start: float, end: float, minimum: int = 2) -> bool:
    return sum(1 for e in episode.events if e.is_spoken and start <= e.start_s < end) >= minimum


def continuous_candidates(episode: Episode, ctx: ShowContext) -> list[Candidate]:
    beats = ctx.beats[episode.number]
    pairs: list[tuple[float, Candidate]] = []
    for hook, end in product(_top_hooks(episode, ctx, HOOKS_PER_EPISODE), _top_ends(episode, ctx, ENDS_PER_EPISODE)):
        opener, closer = snap_start(episode, hook), snap_end(episode, end)
        if opener is None or closer is None:
            continue
        duration = closer.end_s - opener.start_s
        if not MIN_PROMO_S <= duration <= MAX_PROMO_S:
            continue
        if not _has_dialogue(episode, opener.start_s, closer.end_s, 3):
            continue
        segment = make_segment(episode, opener, closer)
        candidate = Candidate(id="", kind="continuous", segments=[segment], hook_event=(episode.number, hook.index),
                              end_event=(episode.number, end.index),
                              justification="One unbroken stretch from a high-signal opening to an open ending.")
        pairs.append((beats[hook.index].hook + 1.2 * beats[end.index].cliff, candidate))
    pairs.sort(key=lambda item: item[0], reverse=True)
    unique, seen = [], set()
    for _, candidate in pairs:
        key = (candidate.segments[0].start_s, candidate.segments[0].end_s)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[:KEEP_CONTINUOUS]


def _head_segment(episode: Episode, hook: Event, target: float) -> tuple[Segment, Event] | None:
    opener = snap_start(episode, hook)
    if opener is None:
        return None
    lines = [e for e in episode.events if e.is_spoken and e.end_s > opener.start_s]
    if not lines:
        return None
    closing = min(lines, key=lambda e: abs((e.end_s - opener.start_s) - target))
    closer = closing if not _straddles_end(episode, closing.end_s) else None
    if closer is None or closer.end_s - opener.start_s < 4:
        return None
    return make_segment(episode, opener, closer), hook


def _tail_segment(episode: Episode, end: Event, target: float) -> Segment | None:
    closer = snap_end(episode, end)
    if closer is None:
        return None
    lines = [e for e in episode.events if e.is_spoken and e.start_s < closer.end_s - 3]
    if not lines:
        return None
    opening = min(lines, key=lambda e: abs((closer.end_s - e.start_s) - target))
    if _straddles_start(episode, opening.start_s) or closer.end_s - opening.start_s < 6:
        return None
    return make_segment(episode, opening, closer)


def same_episode_montages(episode: Episode, ctx: ShowContext) -> list[Candidate]:
    """Join a hook and an ending that are too far apart to fit in one continuous clip."""
    beats = ctx.beats[episode.number]
    found: list[tuple[float, Candidate]] = []
    for hook, end in product(_top_hooks(episode, ctx, 6), _top_ends(episode, ctx, 6)):
        if end.end_s - hook.start_s <= MAX_PROMO_S:
            continue
        for head_target, tail_target in product((8.0, 14.0, 22.0), (15.0, 25.0, 35.0)):
            head = _head_segment(episode, hook, head_target)
            tail = _tail_segment(episode, end, tail_target)
            if head is None or tail is None:
                continue
            first, _ = head
            if tail.start_s - first.end_s < 3 or not MIN_PROMO_S <= first.duration + tail.duration <= MAX_PROMO_S:
                continue
            candidate = Candidate(
                id="", kind="same_episode_montage", segments=[first, tail],
                hook_event=(episode.number, hook.index), end_event=(episode.number, end.index),
                justification=(f"Hook and ending sit {end.end_s - hook.start_s:.0f}s apart, too far for one 60s clip, "
                               "so the middle of the same episode is skipped."),
                flags=["editor_check_join"])
            found.append((beats[hook.index].hook + 1.2 * beats[end.index].cliff, candidate))
    found.sort(key=lambda item: item[0], reverse=True)
    return _dedupe(found, KEEP_MONTAGE)


def _linked(earlier: Episode, later: Episode) -> bool:
    if later.number - earlier.number == 1:
        return True
    notes = later.continuity_notes.lower()
    return f"episode {earlier.number}" in notes


def cross_episode_montages(earlier: Episode, later: Episode, ctx: ShowContext) -> list[Candidate]:
    """Join a hook with the ending of a directly connected later episode, if they share a character."""
    if not _linked(earlier, later):
        return []
    found: list[tuple[float, Candidate]] = []
    early_beats, late_beats = ctx.beats[earlier.number], ctx.beats[later.number]
    for hook, end in product(_top_hooks(earlier, ctx, 4), _top_ends(later, ctx, 4)):
        for head_target, tail_target in product((10.0, 18.0, 26.0), (15.0, 25.0, 32.0)):
            head = _head_segment(earlier, hook, head_target)
            tail = _tail_segment(later, end, tail_target)
            if head is None or tail is None:
                continue
            first, _ = head
            if not MIN_PROMO_S <= first.duration + tail.duration <= MAX_PROMO_S:
                continue
            early_cast = {e.character for e in earlier.events
                          if e.is_spoken and first.start_s <= e.start_s < first.end_s}
            late_cast = {e.character for e in later.events if e.is_spoken and tail.start_s <= e.start_s < tail.end_s}
            shared = early_cast & late_cast
            if not shared:
                continue
            candidate = Candidate(
                id="", kind="cross_episode_montage", segments=[first, tail],
                hook_event=(earlier.number, hook.index), end_event=(later.number, end.index),
                justification=(f"Episode {later.number} continues the story of episode {earlier.number} and shares "
                               f"{', '.join(sorted(shared)[:2])}, so the setup can lead straight to the later ending."),
                flags=["editor_check_join", "cross_episode"])
            found.append((early_beats[hook.index].hook + 1.2 * late_beats[end.index].cliff, candidate))
    found.sort(key=lambda item: item[0], reverse=True)
    return _dedupe(found, KEEP_CROSS)


def _dedupe(found: list[tuple[float, Candidate]], limit: int) -> list[Candidate]:
    unique, seen = [], set()
    for _, candidate in found:
        key = tuple((s.episode, s.start_s, s.end_s) for s in candidate.segments)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[:limit]


def generate_candidates(episodes: list[Episode], ctx: ShowContext) -> list[Candidate]:
    candidates: list[Candidate] = []
    ordered = sorted(episodes, key=lambda e: e.number)
    for position, episode in enumerate(ordered):
        candidates.extend(continuous_candidates(episode, ctx))
        candidates.extend(same_episode_montages(episode, ctx))
        for later in ordered[position + 1:position + 3]:
            candidates.extend(cross_episode_montages(episode, later, ctx))
    prefixes = {"continuous": "C", "same_episode_montage": "M", "cross_episode_montage": "X"}
    for number, candidate in enumerate(candidates, 1):
        candidate.id = f"{prefixes[candidate.kind]}{candidate.segments[0].episode:03d}-{number:05d}"
    return candidates
