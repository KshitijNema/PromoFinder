"""Drives the real Streamlit script headlessly against a run built from synthetic scripts."""
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.pipeline import run_pipeline

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(script_dir, tmp_path):
    out = tmp_path / "out"
    run_pipeline(script_dir, out, aliases_path=None, top_n=4)
    test = AppTest.from_file(APP, default_timeout=60)
    test.session_state["output_dir"] = str(out)
    return test.run()


def test_dashboard_renders_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "PromoFinder"
    assert {t.label for t in app.tabs} == {"Overview", "Promo leaderboard", "Quality checks", "Export"}
    labels = {m.label for m in app.metric}
    assert {"Episodes processed", "Candidates generated", "Ranked promos"} <= labels
    assert next(m for m in app.metric if m.label == "Episodes processed").value == "4"


def test_score_filter_can_empty_the_list(app):
    app.slider[0].set_value(100).run()
    assert not app.exception
    assert any("Showing **0**" in m.value for m in app.markdown)


def test_episode_filter(app):
    app.multiselect[0].select(app.multiselect[0].options[0]).run()
    assert not app.exception


def test_empty_output_folder_shows_guidance(tmp_path):
    test = AppTest.from_file(APP, default_timeout=60)
    test.session_state["output_dir"] = str(tmp_path)
    test.run()
    assert not test.exception and "No results yet" in test.info[0].value
