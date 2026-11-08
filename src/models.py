"""Plain data structures shared by every pipeline stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MIN_PROMO_S = 20.0
MAX_PROMO_S = 60.0
HOOK_WINDOW_S = 3.0

EVENT_CAMERA = "camera"
EVENT_ACTION = "action"
EVENT_SFX = "sfx"
EVENT_DIALOGUE = "dialogue"


@dataclass
class Event:
    """One timed block from the screenplay, kept exactly as authored."""

    episode: int
    scene: int
    index: int  # position inside the episode, in file order
    type: str
    text: str
    start_s: float
    end_s: float
    start_raw: str
    end_raw: str
    line_no: int  # 1-based line of the "Start: ... | End: ..." row in the source file
    label_raw: str = ""
    character: str = ""  # canonical speaker, dialogue only
    tones: list[str] = field(default_factory=list)  # performance notes, e.g. "Furious"
    voice: str = ""  # "", "O.S." or "V.O."
    cues: list[str] = field(default_factory=list)  # bracketed cues inside the line

    @property
    def is_spoken(self) -> bool:
        return self.type == EVENT_DIALOGUE

    @property
    def duration(self) -> float:
        return round(self.end_s - self.start_s, 3)


@dataclass
class Scene:
    number: int
    heading: str
    start_s: float | None
    end_s: float | None
    start_raw: str
    end_raw: str


@dataclass
class Episode:
    number: int
    title: str
    external_id: str
    source_file: str
    logline: str
    continuity_notes: str
    scenes: list[Scene]
    events: list[Event]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [e.end_s for e in self.events]
        scene_ends = [s.end_s for s in self.scenes if s.end_s is not None]
        return max(ends + scene_ends) if ends or scene_ends else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = self.duration
        return data


@dataclass
class Segment:
    """One continuous stretch of a promo. Boundaries always come from real events."""

    episode: int
    start_s: float
    end_s: float
    start_raw: str
    end_raw: str
    start_line: int  # source line holding start_raw
    end_line: int  # source line holding end_raw
    first_event: int
    last_event: int

    @property
    def duration(self) -> float:
        return round(self.end_s - self.start_s, 3)


@dataclass
class Candidate:
    id: str
    kind: str  # continuous | same_episode_montage | cross_episode_montage
    segments: list[Segment]
    hook_event: tuple[int, int]  # (episode, event index)
    end_event: tuple[int, int]
    justification: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 3)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})


@dataclass
class Evaluation:
    """What an evaluator returns for one candidate. Any evaluator must fill all of it."""

    candidate_id: str
    method: str
    hook: float
    cliffhanger: float
    emotion: float
    standalone: float
    curiosity: float
    overall: float
    hook_text: str
    cliffhanger_text: str
    hook_explanation: str
    cliffhanger_explanation: str
    standalone_explanation: str
    why_new_viewer: str
    why_it_works: str
    weaknesses: list[str]
    situation: str
    signals: dict[str, Any] = field(default_factory=dict)
