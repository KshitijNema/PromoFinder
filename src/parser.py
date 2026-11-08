"""Parse timestamped screenplay text files into Episode objects.

The parser anchors on the timing row ("Start: (..) | End: (..)") instead of on a fixed
block layout. The non-blank line above a timing row is that event's label, and everything
up to the next label is its text. This keeps it working when label casing, bold markers or
spacing change. Anything it has to repair or cannot read is recorded on Episode.issues.
"""
from __future__ import annotations

import difflib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import (EVENT_ACTION, EVENT_CAMERA, EVENT_DIALOGUE, EVENT_SFX, Episode, Event, Scene)

TIME_ROW = re.compile(
    r"^\s*Start\s*:\s*\(?\s*(?P<start>[\d:.]+)\s*\)?\s*[|,;]\s*End\s*:\s*\(?\s*(?P<end>[\d:.]+)\s*\)?\s*$",
    re.I,
)
HEADER_FIELD = re.compile(r"^\s*(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
SCENE_START = re.compile(r"^\s*SCENE\s+(\d+)\s*$", re.I)
SCENE_FIELD = re.compile(r"^\s*Scene\s+(heading|start|end)\s*:\s*(.*)$", re.I)
ACTIONS_HEADER = re.compile(r"^\s*Actions and dialogue\s*:\s*$", re.I)
SCRIPT_MARK = re.compile(r"^\s*---\s*SCRIPT\s+(START|END)\s*---\s*$", re.I)
LABEL = re.compile(r"^\s*(?:\*\*)?\s*(?P<label>.+?)\s*:?\s*(?:\*\*)?\s*:?\s*$")
PAREN = re.compile(r"\(([^()]*)\)")
FILE_NAME = re.compile(r"episode[-_ ]*0*(\d+)(?:.*?(\d{5,}))?", re.I)
BRACKET_CUE = re.compile(r"\[([^\]]+)\]")
CAPITAL_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")

VOICE_MARKS = {"o.s.": "O.S.", "o.s": "O.S.", "off-screen": "O.S.", "offscreen": "O.S.", "v.o.": "V.O.", "v.o": "V.O."}
CONTINUE_MARKS = {"continues", "continuing", "cont'd", "cont.", "voice continuing"}
TYPE_LABELS = {"CAMERA": EVENT_CAMERA, "ACTION": EVENT_ACTION, "SOUND FX": EVENT_SFX, "SFX": EVENT_SFX,
               "SOUND": EVENT_SFX, "SOUND EFFECT": EVENT_SFX}

# Capitalised words that are not story-specific names.
NOT_NAMES = {
    "The", "And", "But", "You", "Your", "She", "Her", "His", "They", "Their", "That", "This", "What", "Why", "Who",
    "How", "When", "Where", "Then", "Not", "Yes", "Now", "For", "With", "Lady", "Lord", "Duke", "Duchess", "Dowager",
    "Grace", "Honor", "God", "Father", "Mother", "Sister", "Brother", "Sir", "Madam", "Miss", "Please", "Look",
    "Stop", "Good", "Let", "Come", "Enough", "Take", "Get", "Keep", "Tell", "Are", "Did", "Does", "Have", "Has",
    "Had", "Was", "Were", "Will", "Would", "Can", "Could", "Should", "Must", "May", "Might", "Is", "Am", "All", "Any",
    "Because", "Before", "After", "If", "So", "Some", "There", "Here", "Today", "Tomorrow", "Tonight", "Those",
    "These", "One", "Two", "Three", "Ten", "Six", "Four", "Never", "Nothing", "Everyone", "Everything", "Perhaps",
    "Only", "Even", "Still", "Just", "Our", "Their", "Its", "Him", "Them", "Dear", "Crown", "King", "Queen", "Yet",
    "Very", "Much", "Too", "Instead", "Soon", "Once", "Than", "Like", "Don", "Doesn", "Didn", "Won", "Can't",
    "Ahh", "Aaargh", "Bring", "Give", "Show", "Stand", "Wait", "Move", "Leave", "Burn", "Silence", "Guards", "Read",
    "Search", "Sit", "Hold", "Drink", "Rest", "Careful", "Think", "Love", "Home", "Help", "Save", "Thank",
}


class ParseError(ValueError):
    pass


def to_seconds(raw: str) -> float:
    """Convert 'MM:SS.ff' or 'HH:MM:SS.ff' (or plain seconds) to seconds."""
    parts = raw.strip().split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ParseError(f"Unreadable timestamp: {raw!r}") from exc
    if not 1 <= len(numbers) <= 3:
        raise ParseError(f"Unreadable timestamp: {raw!r}")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return round(seconds, 3)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def _split_label(label: str) -> tuple[str, list[str], str, list[str]]:
    """Split 'NAME (O.S.) (Cold, Firm)' into name, tones, voice and bare parentheticals."""
    name = PAREN.sub("", label).strip(" :*")
    tones: list[str] = []
    voice = ""
    extras: list[str] = []
    for chunk in PAREN.findall(label):
        pieces = [p.strip() for p in re.split(r"[,/]", chunk) if p.strip()]
        for piece in pieces:
            low = piece.lower()
            if low in VOICE_MARKS:
                voice = VOICE_MARKS[low]
            elif low.startswith("voice continuing") or low in CONTINUE_MARKS:
                continue
            else:
                tones.append(piece)
        if len(pieces) == 1:
            extras.append(pieces[0])
    return name, tones, voice, extras


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def parse_text(text: str, source_file: str = "<memory>") -> Episode:
    """Parse one screenplay. Raises ParseError only if no timed events can be found."""
    lines = text.splitlines()
    issues: list[dict] = []

    def issue(code: str, line: int, detail: str) -> None:
        issues.append({"code": code, "line": line, "detail": detail})

    header: dict[str, str] = {}
    scenes: list[Scene] = []
    time_rows = [i for i, line in enumerate(lines) if TIME_ROW.match(line)]
    if not time_rows:
        raise ParseError(f"{source_file}: no 'Start: ... | End: ...' rows found")
    label_lines: dict[int, int] = {}
    for row in time_rows:
        cursor = row - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1
        if cursor >= 0:
            label_lines[cursor] = row

    # Structural pass: header fields and scenes.
    current: dict | None = None
    for i, line in enumerate(lines):
        match = HEADER_FIELD.match(line)
        if match:
            header[match.group(1).upper()] = match.group(2).strip()
            continue
        match = SCENE_START.match(line)
        if match:
            if current:
                scenes.append(_finish_scene(current, issue))
            current = {"number": int(match.group(1)), "line": i + 1, "heading": "", "start": "", "end": ""}
            continue
        match = SCENE_FIELD.match(line)
        if match and current is not None:
            current[match.group(1).lower()] = match.group(2).strip()
            if not line.lstrip().startswith("Scene"):
                issue("label_case_variant", i + 1, line.strip()[:40])
    if current:
        scenes.append(_finish_scene(current, issue))
    if not scenes:
        issue("no_scene_headers", 0, "No SCENE blocks found; all events assigned to scene 0")

    number_text = header.get("EPISODE NUMBER", "")
    file_match = FILE_NAME.search(Path(source_file).name)
    number_match = re.search(r"(\d+)", number_text)
    file_number = int(file_match.group(1)) if file_match else None
    if number_match:
        number = int(number_match.group(1))
        if file_number is not None and file_number != number:
            issue("episode_number_mismatch", 1, f"file says {file_number}, header says {number}")
    elif file_number is not None:
        number = file_number
        issue("episode_number_from_filename", 1, "Header missing; used file name")
    else:
        raise ParseError(f"{source_file}: cannot determine episode number")

    # Event pass.
    structural = {i for i, line in enumerate(lines)
                  if HEADER_FIELD.match(line) or SCENE_START.match(line) or SCENE_FIELD.match(line)
                  or ACTIONS_HEADER.match(line) or SCRIPT_MARK.match(line)}
    events: list[Event] = []
    ordered_labels = sorted(label_lines)
    for position, label_line in enumerate(ordered_labels):
        row = label_lines[label_line]
        limit = ordered_labels[position + 1] if position + 1 < len(ordered_labels) else len(lines)
        body = [lines[i].strip() for i in range(row + 1, limit) if i not in structural and lines[i].strip()]
        match = LABEL.match(lines[label_line])
        label = (match.group("label") if match else lines[label_line]).strip().strip("*").rstrip(":").strip()
        time_match = TIME_ROW.match(lines[row])
        start_raw, end_raw = time_match.group("start"), time_match.group("end")
        try:
            start_s, end_s = to_seconds(start_raw), to_seconds(end_raw)
        except ParseError as exc:
            issue("bad_timestamp", row + 1, str(exc))
            continue
        if end_s < start_s:
            issue("end_before_start", row + 1, f"{start_raw} > {end_raw}")
        name, tones, voice, _ = _split_label(label)
        event_type = TYPE_LABELS.get(name.upper(), EVENT_DIALOGUE)
        joined = " ".join(body).strip()
        if not joined:
            issue("empty_event", row + 1, label[:50])
        cues = BRACKET_CUE.findall(joined)
        if event_type == EVENT_DIALOGUE:
            joined = joined.strip().strip('"“”').strip()
            if cues:
                issue("bracket_cue_in_dialogue", row + 1, ", ".join(cues)[:50])
        scene_no = 0
        for scene in scenes:
            if scene.start_s is not None and scene.start_s <= start_s:
                scene_no = scene.number
        events.append(Event(
            episode=number, scene=scene_no, index=len(events), type=event_type, text=joined,
            start_s=start_s, end_s=end_s, start_raw=start_raw, end_raw=end_raw, line_no=row + 1,
            label_raw=label, character=name if event_type == EVENT_DIALOGUE else "",
            tones=tones, voice=voice, cues=cues,
        ))

    events.sort(key=lambda e: (e.start_s, e.index))
    previous_file_order = {e.line_no: e for e in events}
    for new_index, event in enumerate(events):
        event.index = new_index
    _check_structure(events, scenes, issue, previous_file_order)
    title = number_text or f"Episode {number}"
    external = file_match.group(2) if file_match and file_match.group(2) else ""
    return Episode(number=number, title=title, external_id=external, source_file=source_file,
                   logline=header.get("EPISODE LOGLINE", ""), continuity_notes=header.get("CONTINUITY NOTES", ""),
                   scenes=scenes, events=events, issues=issues)


def _finish_scene(data: dict, issue) -> Scene:
    def seconds(raw: str, what: str) -> float | None:
        if not raw:
            issue("scene_missing_time", data["line"], f"Scene {data['number']} has no {what}")
            return None
        try:
            return to_seconds(raw)
        except ParseError as exc:
            issue("bad_timestamp", data["line"], str(exc))
            return None

    return Scene(number=data["number"], heading=data["heading"], start_s=seconds(data["start"], "start"),
                 end_s=seconds(data["end"], "end"), start_raw=data["start"], end_raw=data["end"])


def _check_structure(events: list[Event], scenes: list[Scene], issue, _order) -> None:
    by_line = sorted(events, key=lambda e: e.line_no)
    for earlier, later in zip(by_line, by_line[1:]):
        if later.start_s < earlier.start_s and later.type == earlier.type:
            issue("event_order_not_chronological", later.line_no, f"{later.start_raw} after {earlier.start_raw}")
    spoken = [e for e in events if e.is_spoken]
    for earlier, later in zip(spoken, spoken[1:]):
        if later.start_s < earlier.end_s - 0.01:
            issue("dialogue_overlap", later.line_no, f"{earlier.end_raw} > {later.start_raw}")
    camera = [e for e in events if e.type == EVENT_CAMERA]
    for earlier, later in zip(camera, camera[1:]):
        gap = round(later.start_s - earlier.end_s, 3)
        if gap > 0.05:
            issue("camera_gap", later.line_no, f"{gap}s uncovered before {later.start_raw}")
        elif gap < -0.05:
            issue("camera_overlap", later.line_no, f"{-gap}s overlap at {later.start_raw}")
    for scene, following in zip(scenes, scenes[1:]):
        if scene.end_s is not None and following.start_s is not None and abs(scene.end_s - following.start_s) > 0.01:
            issue("scene_boundary_mismatch", 0, f"Scene {scene.number} end != scene {following.number} start")
    if scenes and scenes[0].number != 1:
        issue("scene_numbering", 0, f"First scene is {scenes[0].number}")
    for event in events:
        for scene in scenes:
            if scene.number == event.scene and scene.end_s is not None and event.end_s > scene.end_s + 0.01:
                issue("event_past_scene_end", event.line_no, f"{event.end_raw} > scene end {scene.end_raw}")
    spoken_times = [e for e in events if e.type != EVENT_CAMERA]
    if camera and spoken_times and max(e.end_s for e in spoken_times) > max(e.end_s for e in camera) + 0.01:
        issue("audio_past_last_shot", 0, "A line or sound runs past the last camera shot")


def load_aliases(path: Path | None) -> dict:
    """Optional show config: {"aliases": {"LABEL": "NAME"}, "episode_aliases": {"LABEL": {"2": "NAME"}}}."""
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {"aliases": {}, "episode_aliases": {}}


def apply_character_aliases(episodes: list[Episode], config: dict) -> None:
    """Map script labels to one canonical speaker name.

    Labels the show's own script ties together, such as 'NAME (Other Name)', are linked
    automatically. Everything else needs the optional config, since the same label can
    mean different people in different episodes.
    """
    aliases = {k.upper(): v for k, v in config.get("aliases", {}).items()}
    per_episode = {k.upper(): v for k, v in config.get("episode_aliases", {}).items()}
    known = {e.character.upper() for ep in episodes for e in ep.events if e.is_spoken}
    for episode in episodes:
        for event in episode.events:
            if not event.is_spoken:
                continue
            key = event.character.upper()
            target = per_episode.get(key, {}).get(str(episode.number)) or aliases.get(key)
            if not target:
                _, _, _, extras = _split_label(event.label_raw)
                for extra in extras:
                    if extra.upper() in known and extra.upper() != key:
                        target = extra
                        event.tones = [t for t in event.tones if t != extra]
            event.character = (target or key).upper()


def proper_nouns(text: str) -> set[str]:
    """Names in a line, skipping sentence-opening words that are only capitalised by position."""
    found: set[str] = set()
    for match in CAPITAL_WORD.finditer(text):
        word = match.group(0)
        if word in NOT_NAMES:
            continue
        before = text[:match.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?\"“-—"
        if sentence_start and not re.match(r"\s*[,!?.]|\s*$", text[match.end():match.end() + 2]):
            continue
        found.add(word)
    return found


def build_entity_index(episodes: list[Episode]) -> dict[str, int]:
    """First episode in which each spoken name appears. Used to measure 'backstory debt'."""
    first: dict[str, int] = {}
    for episode in sorted(episodes, key=lambda e: e.number):
        for event in episode.events:
            if event.is_spoken:
                for name in proper_nouns(event.text):
                    first.setdefault(name, episode.number)
    return first


def find_name_variants(entity_index: dict[str, int]) -> list[list[str]]:
    """Group near-identical spoken names (likely spelling drift between episodes)."""
    names = sorted(entity_index)
    groups: dict[str, set[str]] = defaultdict(set)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if name[:3] == other[:3] and difflib.SequenceMatcher(None, name, other).ratio() >= 0.72:
                groups[name].add(other)
                groups[other].add(name)
    seen: set[str] = set()
    output: list[list[str]] = []
    for name in names:
        if name in seen or name not in groups:
            continue
        stack, cluster = [name], set()
        while stack:
            item = stack.pop()
            if item not in cluster:
                cluster.add(item)
                stack.extend(groups[item])
        seen |= cluster
        output.append(sorted(cluster))
    return output


def safe_extract(zip_path: Path, destination: Path) -> Path:
    """Extract a ZIP, refusing entries that would escape the destination folder."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ParseError(f"Unsafe path in ZIP: {member.filename}")
        archive.extractall(destination)
    return destination


def discover_files(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.txt") if not p.name.startswith("."))


def parse_folder(paths: Iterable[Path] | Path) -> tuple[list[Episode], list[dict]]:
    """Parse every script. A file that cannot be parsed is reported, never silently dropped."""
    files = discover_files(paths) if isinstance(paths, (str, Path)) else list(paths)
    episodes: list[Episode] = []
    failures: list[dict] = []
    for path in files:
        try:
            text, encoding = _read_text(path)
            episode = parse_text(text, str(path))
            if encoding != "utf-8-sig":
                episode.issues.append({"code": "non_utf8_file", "line": 0, "detail": encoding})
            episodes.append(episode)
        except (ParseError, OSError) as exc:
            failures.append({"file": str(path), "error": str(exc)})
    episodes.sort(key=lambda e: e.number)
    seen: set[int] = set()
    for episode in episodes:
        if episode.number in seen:
            failures.append({"file": episode.source_file, "error": f"Duplicate episode number {episode.number}"})
        seen.add(episode.number)
    return episodes, failures
