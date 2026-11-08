import zipfile

import pytest

from conftest import block, make_script
from src.parser import (ParseError, apply_character_aliases, find_name_variants, parse_folder, parse_text, proper_nouns,
                        safe_extract, to_seconds)


def test_timestamps_convert():
    assert to_seconds("01:12.80") == 72.8
    assert to_seconds("00:00.00") == 0
    assert to_seconds("01:01:01.50") == 3661.5
    with pytest.raises(ParseError):
        to_seconds("soon")


def test_parses_header_and_events():
    episode = parse_text(make_script(7), "episode-007-Episode 123456.txt")
    assert episode.number == 7 and episode.external_id == "123456"
    assert "betrayed" in episode.logline
    assert {e.type for e in episode.events} == {"camera", "dialogue", "action"}
    line = next(e for e in episode.events if e.is_spoken)
    assert line.character == "MARA" and line.tones == ["Furious", "Shouting"]
    assert line.start_raw == "00:00.50" and line.start_s == 0.5 and not line.text.startswith('"')


def test_uppercase_labels_read_the_same_and_are_reported():
    normal, upper = parse_text(make_script(1), "e1.txt"), parse_text(make_script(1, upper=True), "e1.txt")
    assert upper.scenes[0].heading == normal.scenes[0].heading and len(upper.events) == len(normal.events)
    assert any(i["code"] == "label_case_variant" for i in upper.issues)


def test_line_numbers_point_at_timing_rows():
    text = make_script(3)
    source = text.splitlines()
    for event in parse_text(text, "e3.txt").events:
        assert f"Start: ({event.start_raw})" in source[event.line_no - 1]


def test_voice_marks_are_not_tones():
    text = make_script(1).replace("--- SCRIPT END ---", block("ANNA (O.S. / Continues) (Low, Cold)", "01:30.00", "01:31.00", '"Wait."') + "--- SCRIPT END ---")
    line = next(e for e in parse_text(text, "e.txt").events if e.character == "ANNA")
    assert line.voice == "O.S." and line.tones == ["Low", "Cold"]


def test_bold_markers_are_optional():
    text = make_script(1).replace("**CAMERA:**", "CAMERA:")
    assert any(e.type == "camera" for e in parse_text(text, "e.txt").events)


def test_no_timing_rows_is_an_error():
    with pytest.raises(ParseError):
        parse_text("just some prose", "x.txt")


def test_unreadable_file_is_reported_not_dropped(script_dir):
    (script_dir / "episode-099-Episode 1.txt").write_text("nothing useful", encoding="utf-8")
    episodes, failures = parse_folder(script_dir)
    assert len(episodes) == 4 and len(failures) == 1 and "episode-099" in failures[0]["file"]


def test_aliases_apply_by_episode_and_from_the_label():
    first, second = parse_text(make_script(1), "e1.txt"), parse_text(make_script(2), "e2.txt")
    apply_character_aliases([first, second], {"aliases": {"MARA": "LADY MARA"}, "episode_aliases": {"VICTOR": {"2": "THE DUKE"}}})
    assert {e.character for e in first.events if e.is_spoken} == {"LADY MARA", "VICTOR"}
    assert {e.character for e in second.events if e.is_spoken} == {"LADY MARA", "THE DUKE"}


def test_proper_nouns_skip_sentence_openers():
    assert proper_nouns("Look at Victor, then leave.") == {"Victor"}
    assert proper_nouns("Nobody came to Harrowgate.") == {"Harrowgate"}


def test_finds_spelling_variants():
    assert find_name_variants({"Harrowgate": 1, "Harrowgates": 3, "Victor": 1}) == [["Harrowgate", "Harrowgates"]]


def test_zip_slip_is_refused(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../evil.txt", "x")
    with pytest.raises(ParseError):
        safe_extract(archive, tmp_path / "out")
