import copy
import json

import pandas as pd
import pytest

from src.candidate_generator import generate_candidates
from src.evaluator import LLM_SCHEMA, HeuristicEvaluator, LLMEvaluator, OpenAICompatibleClient
from src.models import Candidate, Segment
from src.parser import parse_folder, parse_text
from src.pipeline import run_pipeline
from src.quality_checks import check_candidate, run_quality_checks
from src.ranking import overlap_ratio, rank_candidates
from src.scorer import WEIGHTS, _names_needing_context, analyze_show, overall_score
from conftest import make_script


@pytest.fixture
def world(script_dir):
    episodes, _ = parse_folder(script_dir)
    ctx = analyze_show(episodes)
    candidates = generate_candidates(episodes, ctx)
    return episodes, ctx, candidates, HeuristicEvaluator().evaluate(candidates, ctx)


def test_weights_match_the_brief():
    assert WEIGHTS == {"hook": 0.25, "cliffhanger": 0.30, "emotion": 0.20, "standalone": 0.15, "curiosity": 0.10}
    assert overall_score({k: 100 for k in WEIGHTS}) == 100
    assert overall_score({"hook": 80, "cliffhanger": 60, "emotion": 40, "standalone": 20, "curiosity": 0}) == 49.0


def test_one_word_lines_are_not_treated_as_backstory_names():
    assert _names_needing_context("Poison.") == set()
    assert _names_needing_context("Victor!") == set()
    assert _names_needing_context("Then ask Victor about it.") == {"Victor"}


def test_all_candidates_are_in_range_with_real_boundaries(world):
    episodes, _, candidates, _ = world
    assert candidates and {c.kind for c in candidates} >= {"continuous"}
    index = {e.number: e for e in episodes}
    for candidate in candidates:
        assert 20 <= candidate.duration <= 60
        for segment in candidate.segments:
            events = index[segment.episode].events
            assert events[segment.first_event].start_raw == segment.start_raw
            assert events[segment.last_event].end_raw == segment.end_raw


def test_every_evaluation_is_complete_and_labelled(world):
    _, _, candidates, evaluations = world
    assert len(evaluations) == len(candidates)
    for e in evaluations:
        assert 0 <= e.overall <= 100 and e.hook_explanation and e.cliffhanger_explanation
        assert e.standalone_explanation and e.why_new_viewer and "not validated" in e.method


def test_candidates_pass_checks_on_clean_input(world):
    episodes, _, candidates, evaluations = world
    assert run_quality_checks(candidates, evaluations, episodes)["blocked"] == []


def _broken(candidate, **changes):
    broken = copy.deepcopy(candidate)
    for key, value in changes.items():
        setattr(broken.segments[0], key, value)
    return broken


def test_checks_catch_bad_duration(world):
    episodes, _, candidates, evaluations = world
    bad = _broken(candidates[0], end_s=candidates[0].segments[0].start_s + 5)
    assert check_candidate(bad, evaluations[0], {e.number: e for e in episodes})["duration_20_60s"]["status"] == "fail"


def test_checks_catch_altered_timestamp(world):
    episodes, _, candidates, evaluations = world
    bad = _broken(candidates[0], start_raw="00:00.01", start_s=0.01)
    assert check_candidate(bad, evaluations[0], {e.number: e for e in episodes})["traceable_to_source"]["status"] == "fail"


def test_checks_catch_invalid_episode(world):
    episodes, _, candidates, evaluations = world
    bad = _broken(candidates[0], episode=999)
    assert check_candidate(bad, evaluations[0], {e.number: e for e in episodes})["valid_episode_reference"]["status"] == "fail"


def test_checks_catch_silent_opening(tmp_path):
    text = make_script(1, lines=8)
    path = tmp_path / "episode-001-Episode 1.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_text(text, str(path))
    # Shot 0 opens at 0.0 and the first line is at 0.5, so start where only a camera note begins in the window.
    camera = episode.events[0]
    last = episode.events[-1]
    candidate = Candidate(id="T", kind="continuous", segments=[Segment(1, 5.0, 30.0, "00:05.00", last.end_raw, camera.line_no, last.line_no, 0, last.index)],
                          hook_event=(1, 0), end_event=(1, last.index))
    for event in episode.events:
        if 5.0 <= event.start_s < 8.0 and event.type != "camera":
            event.type, event.text = "camera", "Empty hallway."
    evaluation = HeuristicEvaluator().evaluate([Candidate(id="E", kind="continuous", segments=candidate.segments, hook_event=(1, 0), end_event=(1, last.index))], analyze_show([episode]))[0]
    assert check_candidate(candidate, evaluation, {1: episode})["hook_within_3s"]["status"] == "fail"


def test_ranking_respects_overlap_and_episode_limits(world):
    _, ctx, candidates, evaluations = world
    ranked = rank_candidates(candidates, evaluations, set(), ctx.last_episode, top_n=20)
    by_id = {c.id: c for c in candidates}
    assert ranked and [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    for i, first in enumerate(ranked):
        for second in ranked[i + 1:]:
            assert overlap_ratio(by_id[first["candidate_id"]], by_id[second["candidate_id"]]) <= 0.30


def test_ranking_excludes_blocked_candidates(world):
    _, ctx, candidates, evaluations = world
    top = max(evaluations, key=lambda e: e.overall).candidate_id
    assert top not in {r["candidate_id"] for r in rank_candidates(candidates, evaluations, {top}, ctx.last_episode)}


class FakeClient:
    name = "fake"

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def complete_json(self, system, user, schema):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return {"hook": 91, "cliffhanger": 92, "emotion": 93, "standalone": 94, "curiosity": 95, "hook_explanation": "h", "cliffhanger_explanation": "c",
                "standalone_explanation": "s", "why_new_viewer": "n", "why_it_works": "w", "weaknesses": ["x"]}


def test_llm_rescoring_is_bounded_and_labelled(world):
    _, ctx, candidates, _ = world
    client = FakeClient()
    results = LLMEvaluator(client, top_k=3).evaluate(candidates, ctx)
    changed = [e for e in results if e.method.startswith("llm:fake")]
    assert client.calls == 3 and len(changed) == 3 and len(results) == len(candidates)
    assert changed[0].hook == 91 and "baseline_overall" in changed[0].signals


def test_llm_failure_keeps_rule_score_and_says_so(world):
    _, ctx, candidates, _ = world
    results = LLMEvaluator(FakeClient(fail=True), top_k=2).evaluate(candidates, ctx)
    failed = [e for e in results if "LLM call failed" in e.method]
    assert len(failed) == 2 and "provider down" in failed[0].signals["llm_error"]


def test_openai_adapter_sends_a_valid_request(monkeypatch):
    captured = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"hook": 1}'}}]}).encode()

    def fake_open(request, timeout):
        captured.update(url=request.full_url, body=json.loads(request.data), auth=request.headers["Authorization"])
        return Reply()

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    client = OpenAICompatibleClient("https://example.test/v1/", "secret", "model-x")
    assert client.complete_json("sys", "clip", LLM_SCHEMA) == {"hook": 1}
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["body"]["model"] == "model-x" and captured["auth"] == "Bearer secret"


def test_pipeline_writes_every_output(script_dir, tmp_path):
    out = tmp_path / "out"
    manifest = run_pipeline(script_dir, out, aliases_path=None, top_n=5)
    assert manifest["episodes_parsed"] == 4 and manifest["ranked"] >= 1
    for name in ("01_parsed_episodes.json", "01_parse_issues.json", "02_candidates.json", "03_scores.json", "03_scores.csv",
                 "04_ranked_promos.json", "04_ranked_promos.csv", "05_quality_report.json", "run_manifest.json"):
        assert (out / name).stat().st_size > 0
    ranked = json.loads((out / "04_ranked_promos.json").read_text(encoding="utf-8"))
    assert ranked[0]["segments"][0]["start_raw"] and ranked[0]["excerpt"] and ranked[0]["scores"]["overall"] > 0
    assert len(pd.read_csv(out / "04_ranked_promos.csv")) == len(ranked)


def test_pipeline_reports_gaps_in_episode_numbers(script_dir, tmp_path):
    next(script_dir.glob("episode-002*")).unlink()
    manifest = run_pipeline(script_dir, tmp_path / "out", aliases_path=None)
    assert manifest["missing_episode_numbers"] == [2]


def test_same_input_gives_identical_ranking(script_dir, tmp_path):
    run_pipeline(script_dir, tmp_path / "a", aliases_path=None)
    run_pipeline(script_dir, tmp_path / "b", aliases_path=None)
    assert (tmp_path / "a" / "04_ranked_promos.json").read_bytes() == (tmp_path / "b" / "04_ranked_promos.json").read_bytes()
