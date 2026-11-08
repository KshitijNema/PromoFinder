"""Rule-based narrative signals and the five weighted promo scores.

Every number here is a hand-set heuristic. None of it has been fitted to viewer data, so
scores are a way to order candidates for an editor, not a prediction of performance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import EVENT_CAMERA, HOOK_WINDOW_S, Candidate, Episode, Event, Segment
from .parser import build_entity_index, find_name_variants, proper_nouns

WEIGHTS = {"hook": 0.25, "cliffhanger": 0.30, "emotion": 0.20, "standalone": 0.15, "curiosity": 0.10}


def _rx(words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + words + r")\b", re.I)


TONE_HIGH = re.compile(r"scream|shout|shriek|furious|enrag|rage|hyster|venom|desperat|agon|terrif|horrif|sobbing|"
                       r"wail|weeping|shatter|lethal|deadly|menac|cruel|vicious|roar|thunder|booming|panic|anguish|"
                       r"brutal|sadistic|malicious|murder|devastat|broken|choked|crying|tormented|chilling|sinister", re.I)
TONE_MID = re.compile(r"cold|sharp|stern|mock|defiant|fierce|smug|gloat|scorn|bitter|threat|urgent|trembl|plead|"
                      r"accus|icy|harsh|contempt|disgust|derisive|spite|vindict|steely|shock|stunned|fear|"
                      r"tearful|resolute|unyield|dangerous|ominous|taunt|condemn|alarm|grave|commanding|demanding|"
                      r"betray|outrag|incredul|ruthless|heartbroken|anxious|haunt|bark|hiss|seething", re.I)
TONE_LOW = re.compile(r"gentle|soft|tender|warm|playful|calm|quiet|amus|teasing|reassur|loving|content|peaceful|"
                      r"fond|sweet|modest|relieved|grateful|lighthearted|polite|humble|formal|charming|smiling", re.I)

VIOLENT_ACT = _rx(r"strikes?|struck|slaps?|slapped|punch(?:es)?|whips?|lash\w*|stab\w*|slam\w*|shoves?|hurl\w*|flings?|"
                  r"smash\w*|tackl\w+|lung\w+|explo\w+|detonat\w+|fireball|collaps\w+|convuls\w+|drags?|dragged|dragging|"
                  r"seiz\w+|wrench\w*|plung\w+|rears? back|shatter\w*|tear\w* her")
PERIL_PROP = _rx(r"crop|blood\w*|bleed\w*|chok\w+|flames?|inferno|blaze|ablaze|branding|poker|shackl\w+|handcuffs?|manacl\w+|"
                 r"chains?|thorns?|lacerat\w+|gash\w*|welt|grab\w*|grappl\w+")
THREAT = _rx(r"hang\w*|gallows|noose|execut\w+|kill\w*|dead|die|dies|death|corpse|burn\w*|arrest\w*|condemn\w*|destroy\w*|"
             r"disappear|punish\w*|gaol|dungeon|rot|prisoner|sentenc\w+|poison\w*|witch\w*|torture\w*|bleed")
BETRAYAL = _rx(r"marry\w*|married|betray\w*|abandon\w*|cast out|disown\w*|stole|steal\w*|lied|lying|lie|knew|traitor|"
               r"sold|framed?|planted|humiliat\w+|discard\w*|revoked|worthless|changed my mind|choose|chose|imposter|"
               r"impostor|false|never going to")
REVEAL = _rx(r"truth|secret\w*|arsenic|rightful|alive|who are you|who did|identity|recogni\w+|ghost|buried|ring|"
             r"lost son|your son|rumor|expos\w+|proof|invalid|not (?:your|his) lawful|you're the duke|it's you|deceived")
ACCUSE = re.compile(r"\byou (?:knew|let|stole|planted|lied|swore|abandoned|sold|poisoned|tortured|took|married|"
                    r"chose|wanted|have shown|dare|are lying)\b|\bhow dare\b|\bwhat have (?:you|they) done\b", re.I)
COMMAND = _rx(r"guards?|stop|enough|take her|remove her|let her go|take your hands|get out|silence|arrest him|leave")
PLEAD = _rx(r"please|beg\w*|mercy|save (?:him|her|me)|help me|don't")
SUSPENSE_VISUAL = _rx(r"raises?|raising|hoists?|door|handle|latch|lever|footsteps|bootsteps|inches|millimeters|frozen|"
                      r"freez\w+|cracks? open|swings? open|burst\w* open|approach\w*|silhouett\w+|emerge\w*|"
                      r"tighten\w*|closing around|trap\w*|rotat\w+ downward")
EMOTIVE = _rx(r"horror|terror|shock\w*|stunned|disbelief|gasp\w*|tears?|sobs?|weep\w*|fury|rage|dread|panic\w*|"
              r"devastat\w+|agony|anguish|betray\w*")
EXPLICIT_CLIFF = _rx(r"cliffhanger|cuts? (?:sharply |abruptly )?to black|abrupt\w*|freeze-frame|heart-stopping")
RESOLVED = _rx(r"smil\w+|peaceful\w*|warm\w*|tender\w*|embrac\w+|gratitude|relief|relieved|safe\w*|sunlight|together|"
               r"gentle|contented|seren\w+|hope\w*|thank you|nestl\w+|i accept|i will wait|fades? out|fade-out|slowly fad\w+")
PENDING_DECISION = _rx(r"accept my|marry me|will you|take my hand|your answer|choose now|name your price|renounce|"
                       r"what are you going to do|one command|prove it")
ROLES = _rx(r"husband|wife|sister|father|mother|bride|groom|wedding|son|daughter|brother|family|marry\w*|married|"
            r"soldier|maid|duke|duchess|fianc\w+|heir|widow|mistress")
CLEAR_SETTING = _rx(r"altar|bridal|wedding gown|cathedral|chapel|dungeon|cell|chained|shackl\w+|courtroom|judge|gallows|"
                    r"prisoner|fire|flames|handcuffs?|guards?")
MYSTERY = _rx(r"who|why|secret\w*|truth|how long|what do you mean|where did|what is|rumor|identity|really|impossible|"
              r"somewhere|someone")
STAKES = _rx(r"hang\w*|gallows|execut\w+|kill\w*|die|death|dead|marry\w*|wedding|husband|wife|poison\w*|witch\w*|"
             r"duke|heir|title|inherit\w+|dowry|disown\w*|burn\w*|prison\w*|dungeon")
STOP = _rx(r"the|a|an|and|but|or|of|in|on|at|to|for|from|with|by|as|is|are|was|were|be|her|his|their|she|he|they|that|"
           r"this|it|when|while|after|before|into|who|whom|only|just|before|through|upon|about")

SITUATIONS = {
    "betrayal_and_humiliation": _rx(r"altar|wedding|marry\w*|bride|groom|betray\w*|disown\w*|plain|worthless|humiliat\w+|"
                                    r"discard\w*|disgrace\w*|beggar|mock\w*"),
    "rescue_and_protection": _rx(r"let her go|hands off|protect\w*|stay behind me|stand\w* between|rescue\w*|ring|proposal|"
                                 r"one knee|don't touch"),
    "false_accusation_and_trial": _rx(r"judge|magistrate|court\w*|witness\w*|testimony|accus\w+|constable|theft|stole|"
                                      r"planted|witch\w*|arrest\w*|gavel|renounce"),
    "imprisonment_and_torment": _rx(r"dungeon|cell|chain\w*|shackl\w+|branding|poker|iron|manacl\w+|starv\w+|bars"),
    "physical_ordeal_or_peril": _rx(r"thorns?|briars?|blood\w*|fire|flames?|inferno|smoke|beam|explo\w+|thugs?|club"),
    "hidden_identity_reveal": _rx(r"duke|your grace|rightful|alive|lost son|crest|buried|presumed|imposter|who are you|"
                                  r"ashes|title"),
    "poison_and_conspiracy": _rx(r"poison\w*|arsenic|antidote|plot\w*|scheme\w*|conspir\w+|treason|council|blackmail|"
                                 r"secret\w*|dowager"),
    "forbidden_romance": _rx(r"i love|you love|loved|kiss\w*|embrac\w+|your touch|racing|intimate\w*"),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class Beat:
    intensity: float = 0.0
    hook: float = 0.0
    cliff: float = 0.0
    tags: set[str] = field(default_factory=set)
    tone_level: str = "none"


@dataclass
class ShowContext:
    episodes: dict[int, Episode]
    beats: dict[int, list[Beat]]
    entities: dict[str, int]
    name_variants: dict[str, list[str]]
    authored_cliffs: set[int]
    last_episode: int


def tone_level(tones: list[str]) -> str:
    joined = " ".join(tones)
    if not joined:
        return "none"
    if TONE_HIGH.search(joined):
        return "high"
    if TONE_MID.search(joined):
        return "mid"
    if TONE_LOW.search(joined):
        return "low"
    return "neutral"


def _is_continuation(text: str) -> bool:
    stripped = text.lstrip("\"' ")
    return stripped.startswith(("...", "…")) or (bool(stripped) and stripped[0].islower())


def _trails_off(text: str) -> bool:
    return text.rstrip("\"' ").endswith(("...", "…", "—", "--", "-"))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def _tags(text: str) -> set[str]:
    tags = set()
    for name, pattern in (("violent", VIOLENT_ACT), ("prop", PERIL_PROP), ("pending", PENDING_DECISION), ("threat", THREAT), ("betrayal", BETRAYAL), ("reveal", REVEAL),
                          ("accuse", ACCUSE), ("command", COMMAND), ("plead", PLEAD), ("suspense", SUSPENSE_VISUAL),
                          ("emotive", EMOTIVE), ("explicit_cliff", EXPLICIT_CLIFF), ("resolved", RESOLVED)):
        if pattern.search(text):
            tags.add(name)
    return tags


def _intensity(event: Event, tags: set[str], level: str) -> float:
    if event.is_spoken:
        value = {"high": 0.85, "mid": 0.6, "neutral": 0.45, "none": 0.45, "low": 0.2}[level]
        value += 0.08 if "!" in event.text else 0.0
        value += 0.08 if tags & {"threat", "betrayal", "accuse"} else 0.0
        return clamp(value)
    if event.type == EVENT_CAMERA:
        return clamp(0.15 + 0.3 * ("violent" in tags) + 0.1 * ("prop" in tags) + 0.25 * ("emotive" in tags) - 0.1 * ("resolved" in tags))
    return clamp(0.25 + 0.5 * ("violent" in tags) + 0.2 * ("prop" in tags) + 0.2 * ("emotive" in tags) - 0.15 * ("resolved" in tags))


def _next_spoken(events: list[Event], index: int) -> Event | None:
    return next((e for e in events[index + 1:] if e.is_spoken), None)


def _hook_value(event: Event, beat: Beat) -> float:
    tags = beat.tags
    if event.is_spoken:
        concrete = bool(ROLES.search(event.text) or STAKES.search(event.text))
        semantic = 1.0 if tags & {"betrayal", "reveal", "accuse"} else 0.8 if "threat" in tags else 0.4 if tags & {"command", "plead"} else 0.0
        words = _word_count(event.text)
        value = 0.15 + 0.27 * beat.intensity + 0.24 * semantic + (0.08 if 3 <= words <= 10 else 0.0)
        value += 0.05 if event.duration <= HOOK_WINDOW_S + 0.5 else -0.05
        value += 0.09 if concrete else -0.09  # the line alone has to tell a stranger what is at stake
        if _is_continuation(event.text):
            value -= 0.30
        if words <= 3 and "accuse" not in tags and not concrete:
            value -= 0.15
        if words <= 2 and not semantic:
            value -= 0.20
        if beat.tone_level == "low":
            value -= 0.15
        if words <= 2 and proper_nouns(event.text):
            value -= 0.20
        return clamp(value)
    base = 0.10 if event.type == EVENT_CAMERA else 0.20
    violent = 0.36 if event.type == EVENT_CAMERA else 0.55
    return clamp(base + violent * ("violent" in tags) + 0.14 * ("prop" in tags) + 0.10 * ("emotive" in tags) - 0.10 * ("resolved" in tags))


def _cliff_value(events: list[Event], beats: list[Beat], index: int, near_end: bool, authored: bool) -> float:
    event, beat = events[index], beats[index]
    tags = beat.tags
    if event.is_spoken:
        text = event.text.rstrip("\"' ")
        question, trailing = text.endswith("?"), _trails_off(text)
        value = 0.12 + 0.22 * beat.intensity
        if question or trailing:
            # An open line is only as good as what it withholds, so look at the lines that follow the cut.
            upcoming = [e for e in events[index + 1:] if e.is_spoken and e.start_s - event.end_s <= 6][:2]
            withheld = max((beats[e.index].hook for e in upcoming), default=None)
            value += 0.20 if withheld is None else 0.36 * withheld
            value += 0.05 if question and tags & {"threat", "betrayal", "reveal", "accuse"} else 0.0
        if "threat" in tags:
            value += 0.28 + (0.10 if ROLES.search(event.text) else 0.0)
        if "pending" in tags:
            value += 0.30
        if tags & {"betrayal", "reveal"}:
            value += 0.10
        if beat.tone_level == "low":
            value -= 0.25
        if _word_count(event.text) <= 3 and not tags & {"threat", "pending"}:
            value -= 0.22
        if "resolved" in tags and not tags & {"threat", "pending"}:
            value -= 0.15
        later = next((e for e in events[index + 1:] if e.is_spoken), None)
        if later is None or later.scene != event.scene:
            value += 0.06  # the last line of a scene is usually written as a button
    else:
        value = 0.10 + 0.32 * ("suspense" in tags) + 0.16 * ("violent" in tags) + 0.08 * ("prop" in tags) + 0.14 * ("emotive" in tags)
        if "resolved" in tags:
            value -= 0.30
        if near_end and "explicit_cliff" in tags:
            value += 0.22
    if near_end and authored:
        value += 0.30 if not event.is_spoken else 0.14
    return clamp(value)


def analyze_show(episodes: list[Episode]) -> ShowContext:
    entities = build_entity_index(episodes)
    variants: dict[str, list[str]] = {}
    for group in find_name_variants(entities):
        for name in group:
            variants[name] = group
    numbers = {e.number for e in episodes}
    authored: set[int] = set()
    for episode in episodes:
        notes = episode.continuity_notes.lower()
        previous = episode.number - 1
        # A later episode's notes calling the previous ending a "cliffhanger" is the writers' own label.
        if previous in numbers and "cliffhanger" in notes:
            authored.add(previous)
        tail = " ".join(e.text for e in episode.events[-4:])
        if EXPLICIT_CLIFF.search(tail):
            authored.add(episode.number)
    all_beats: dict[int, list[Beat]] = {}
    for episode in episodes:
        beats: list[Beat] = []
        for event in episode.events:
            tags = _tags(event.text)
            level = tone_level(event.tones)
            beats.append(Beat(intensity=_intensity(event, tags, level), tags=tags, tone_level=level))
        last_index = len(episode.events) - 1
        for i, (event, beat) in enumerate(zip(episode.events, beats)):
            beat.hook = _hook_value(event, beat)
            recent = [b.intensity for e, b in zip(episode.events[:i], beats[:i])
                      if e.type != EVENT_CAMERA and event.start_s - e.start_s <= 15]
            if recent and beat.intensity - sum(recent) / len(recent) >= 0.2:
                beat.hook = clamp(beat.hook + 0.08)  # the moment is a spike, not more of the same
        for i, event in enumerate(episode.events):
            near_end = i >= last_index - 2 or event.end_s >= episode.duration - 1.5
            beats[i].cliff = _cliff_value(episode.events, beats, i, near_end, episode.number in authored)
        all_beats[episode.number] = beats
    return ShowContext({e.number: e for e in episodes}, all_beats, entities, variants, authored,
                       max(numbers) if numbers else 0)


def segment_events(episode: Episode, segment: Segment) -> list[Event]:
    return [e for e in episode.events if segment.start_s - 0.001 <= e.start_s < segment.end_s - 0.001]


def clip_events(candidate: Candidate, ctx: ShowContext) -> list[tuple[Event, Beat]]:
    pairs: list[tuple[Event, Beat]] = []
    for segment in candidate.segments:
        episode = ctx.episodes[segment.episode]
        beats = ctx.beats[segment.episode]
        pairs.extend((e, beats[e.index]) for e in segment_events(episode, segment))
    return pairs


def _short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def score_hook(candidate: Candidate, ctx: ShowContext) -> dict:
    first = candidate.segments[0]
    episode, beats = ctx.episodes[first.episode], ctx.beats[first.episode]
    window = [e for e in segment_events(episode, first) if e.start_s - first.start_s < HOOK_WINDOW_S]
    best, best_value = None, -1.0
    for event in window:
        offset = event.start_s - first.start_s
        value = beats[event.index].hook - 0.05 * max(0.0, offset - 1.0)
        if value > best_value:
            best, best_value = event, value
    if best is None:
        return {"score": 0.0, "event": None, "offset": None, "audible": False}
    audible = any(e.type != EVENT_CAMERA for e in window)
    score = best_value * 100
    if not audible:
        score = min(score, 40.0)
    return {"score": round(clamp(score, 0, 100), 1), "event": best, "offset": round(best.start_s - first.start_s, 2),
            "audible": audible}


def score_cliffhanger(candidate: Candidate, ctx: ShowContext) -> dict:
    last = candidate.segments[-1]
    episode, beats = ctx.episodes[last.episode], ctx.beats[last.episode]
    events = segment_events(episode, last)
    spoken = [e for e in events if e.is_spoken]
    final_line = spoken[-1] if spoken else None
    tail = [e for e in events if e.start_s >= last.end_s - 4.0 and not e.is_spoken]
    options = ([final_line] if final_line else []) + tail
    if not options:
        return {"score": 0.0, "event": None, "dead_air": None, "resolved_ending": False}
    best = max(options, key=lambda e: beats[e.index].cliff)
    value = beats[best.index].cliff
    final_window = [e for e in events if e.start_s >= last.end_s - 5.0]
    resolved = (final_line is not None and final_line.start_s >= last.end_s - 6.0 and beats[final_line.index].tone_level == "low") \
        or (sum("resolved" in beats[e.index].tags for e in final_window) >= max(1, len(final_window) // 2))
    if resolved:
        value -= 0.18
    if final_line is not None and final_line is not best and beats[final_line.index].cliff >= 0.45:
        value += 0.06  # the last line and the last image both leave things open
    dead_air = round(last.end_s - final_line.end_s, 2) if final_line else None
    ends_episode = last.end_s >= episode.duration - 0.6
    if dead_air is not None and dead_air > 3.5 and not ends_episode:
        value -= 0.10
    return {"score": round(clamp(value) * 100, 1), "event": best, "dead_air": dead_air, "resolved_ending": resolved,
            "final_line": final_line, "ends_episode": ends_episode}


def score_emotion(pairs: list[tuple[Event, Beat]]) -> dict:
    audible = [b.intensity for e, b in pairs if e.type != EVENT_CAMERA]
    if not audible:
        return {"score": 0.0, "peak": 0.0, "arc": 0.0}
    ordered = sorted(audible, reverse=True)
    top = ordered[: max(1, round(len(ordered) * 0.4))]
    third = max(1, len(audible) // 3)
    arc = sum(audible[-third:]) / third - sum(audible[:third]) / third
    value = 0.60 * sum(top) / len(top) + 0.20 * ordered[0] + 0.20 * clamp(arc + 0.5)
    return {"score": round(clamp(value) * 100, 1), "peak": round(ordered[0], 2), "arc": round(arc, 2)}


def _names_needing_context(text: str) -> set[str]:
    """Names used inside a sentence. A one-word line is either a shout of a name, which introduces it, or not a name."""
    found: set[str] = set()
    for sentence in re.split(r"(?<=[.!?\u2026])\s+", text):
        if _word_count(sentence) >= 2:
            found |= proper_nouns(sentence)
    return found


def score_standalone(candidate: Candidate, ctx: ShowContext, pairs: list[tuple[Event, Beat]]) -> dict:
    spoken = " ".join(e.text for e, _ in pairs if e.is_spoken)
    visuals = " ".join(e.text for e, _ in pairs if not e.is_spoken)
    first_episode = min(candidate.episodes)
    roles = {m.group(0).lower() for m in ROLES.finditer(spoken)}
    named = set().union(*[_names_needing_context(e.text) for e, _ in pairs if e.is_spoken]) if pairs else set()
    debt = sorted(n for n in named if ctx.entities.get(n, first_episode) < first_episode)
    speakers = {e.character for e, _ in pairs if e.is_spoken}
    opener = next((e for e, _ in pairs if e.is_spoken), None)
    continuation = bool(opener and _is_continuation(opener.text))
    pronoun_open = bool(opener and re.match(r"\s*(he|she|they|it|that|this|then|and|but)\b", opener.text, re.I))
    scenes = {(e.episode, e.scene) for e, _ in pairs}
    joins = len(candidate.segments) - 1
    value = 52.0 + min(22.0, 6.0 * len(roles)) + (10.0 if CLEAR_SETTING.search(visuals + " " + spoken) else 0.0)
    value += 5.0 if 2 <= len(speakers) <= 4 else -4.0
    value -= min(32.0, 8.0 * len(debt))
    value -= 10.0 if continuation else 0.0
    value -= 4.0 if pronoun_open else 0.0
    value -= 4.0 * max(0, len(scenes) - 2)
    value -= (12.0 if len(candidate.episodes) > 1 else 7.0) * joins
    variants = sorted({n for n in named if n in ctx.name_variants})
    return {"score": round(clamp(value, 0, 100), 1), "roles": sorted(roles), "debt": debt, "continuation": continuation,
            "speakers": sorted(speakers), "joins": joins, "name_variants": variants}


def _logline_terms(episode: Episode) -> set[str]:
    terms = {w.lower() for w in re.findall(r"[A-Za-z']{4,}", episode.logline)}
    return {t for t in terms if not STOP.fullmatch(t)}


def score_curiosity(candidate: Candidate, ctx: ShowContext, pairs: list[tuple[Event, Beat]]) -> dict:
    spoken_events = [e for e, _ in pairs if e.is_spoken]
    spoken = " ".join(e.text for e in spoken_events)
    everything = " ".join(e.text for e, _ in pairs)
    questions = sum(e.text.rstrip("\"' ").endswith("?") for e in spoken_events[-6:])
    mystery = len({m.group(0).lower() for m in MYSTERY.finditer(spoken)})
    stakes = {m.group(0).lower() for m in STAKES.finditer(everything)}
    words = {w.lower() for w in re.findall(r"[A-Za-z']{4,}", everything)}
    last_episode = ctx.episodes[candidate.segments[-1].episode]
    terms = _logline_terms(last_episode)
    overlap = len(words & terms) / len(terms) if terms else 0.0
    value = 28.0 + min(15.0, 5.0 * questions) + min(15.0, 3.5 * mystery) + min(24.0, 6.0 * len(stakes)) + 18.0 * clamp(overlap * 1.6)
    return {"score": round(clamp(value, 0, 100), 1), "open_questions": questions, "stakes": sorted(stakes),
            "logline_overlap": round(overlap, 2)}


def classify_situation(pairs: list[tuple[Event, Beat]]) -> str:
    text = " ".join(e.text for e, _ in pairs if e.type != EVENT_CAMERA)
    counts = {name: len(pattern.findall(text)) for name, pattern in SITUATIONS.items()}
    return max(counts, key=counts.get) if any(counts.values()) else "other"


def overall_score(components: dict[str, float]) -> float:
    return round(sum(components[name] * weight for name, weight in WEIGHTS.items()), 1)
