"""Shared fixtures. Every script here is invented, so no confidential content enters the tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def block(label: str, start: str, end: str, text: str) -> str:
    return f"**{label}:**\nStart: ({start}) | End: ({end})\n{text}\n\n"


def stamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def make_script(number: int, *, upper: bool = False, lines: int = 18, notes: str = "Picks up from the previous cliffhanger.") -> str:
    """A synthetic episode: alternating shots and lines, ending on an unresolved threat."""
    heading, start, end, body = ("SCENE HEADING", "SCENE START", "SCENE END", "ACTIONS AND DIALOGUE") if upper else \
        ("Scene heading", "Scene start", "Scene end", "Actions and dialogue")
    total = lines * 5.0
    text = (f"EPISODE NUMBER: Episode {number}\nEPISODE LOGLINE: A sister is betrayed at her wedding and threatened.\n"
            f"CONTINUITY NOTES: {notes}\n\n--- SCRIPT START ---\n\nSCENE 01\n{heading}: INT. HALL - DAY\n"
            f"{start}: 00:00.00\n{end}: {stamp(total)}\n\n{body}:\n\n")
    for i in range(lines):
        t = i * 5.0
        text += block("CAMERA", stamp(t), stamp(t + 5), f"Close-up number {i}.")
        speaker, tone = ("MARA", "Furious, Shouting") if i % 2 == 0 else ("VICTOR", "Cold, Menacing")
        line = "You knew and you let my husband betray me!" if i % 2 == 0 else "Your sister will hang from the gallows."
        text += block(f"{speaker} ({tone})", stamp(t + 0.5), stamp(t + 4), f'"{line}"')
    text += block("ACTION", stamp(total - 3), stamp(total), "Victor raises the blade as the door swings open.")
    return text + "--- SCRIPT END ---\n"


@pytest.fixture
def script_dir(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    for number in range(1, 5):
        (folder / f"episode-{number:03d}-Episode 90000{number}.txt").write_text(make_script(number, upper=number == 2), encoding="utf-8")
    return folder
