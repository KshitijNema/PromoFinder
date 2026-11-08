"""Interchangeable evaluators: the free rule-based baseline, or a hybrid with any LLM.

To add a model, implement LLMClient.complete_json and register it in build_evaluator.
The rest of the pipeline only sees Evaluation objects, so nothing else changes.
"""
from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Protocol

from .models import EVENT_CAMERA, Candidate, Evaluation
from .parser import format_time
from .scorer import (ShowContext, classify_situation, clip_events, overall_score, score_cliffhanger, score_curiosity,
                     score_emotion, score_hook, score_standalone)

BASELINE_METHOD = "heuristic_baseline_v1 (deterministic rules; not an AI model; not validated on viewer data)"

SITUATION_PITCH = {
    "betrayal_and_humiliation": "public betrayal is instantly readable, and viewers want to see the victim hit back",
    "rescue_and_protection": "a stranger stepping in for someone with no allies sets up a protector fantasy",
    "false_accusation_and_trial": "an innocent person being framed creates outrage and a need to see justice",
    "imprisonment_and_torment": "a captive refusing to break against a cruel captor creates sympathy and dread",
    "physical_ordeal_or_peril": "visible danger needs no backstory and makes viewers ask whether she survives",
    "hidden_identity_reveal": "a secret identity about to come out is the genre's strongest reason to keep watching",
    "poison_and_conspiracy": "a hidden plot gives viewers knowledge the victim lacks and a question to resolve",
    "forbidden_romance": "romantic tension invites viewers to stay and see whether the couple comes together",
    "other": "the conflict is direct enough to follow without earlier episodes",
}


def _describe_event(event, offset=None) -> str:
    who = event.character.title() if event.is_spoken else event.type
    when = f" {offset:.1f}s in" if offset is not None else ""
    tone = f", performed {', '.join(event.tones)}" if event.tones else ""
    return f"{who}{when}{tone}"


class BaseEvaluator(ABC):
    name = "base"

    @abstractmethod
    def evaluate(self, candidates: list[Candidate], ctx: ShowContext) -> list[Evaluation]:
        """Return one Evaluation for every candidate, in any order."""


class HeuristicEvaluator(BaseEvaluator):
    name = "heuristic"

    def evaluate(self, candidates: list[Candidate], ctx: ShowContext) -> list[Evaluation]:
        return [self._evaluate_one(c, ctx) for c in candidates]

    def _evaluate_one(self, candidate: Candidate, ctx: ShowContext) -> Evaluation:
        pairs = clip_events(candidate, ctx)
        hook = score_hook(candidate, ctx)
        cliff = score_cliffhanger(candidate, ctx)
        emotion = score_emotion(pairs)
        standalone = score_standalone(candidate, ctx, pairs)
        curiosity = score_curiosity(candidate, ctx, pairs)
        scores = {"hook": hook["score"], "cliffhanger": cliff["score"], "emotion": emotion["score"],
                  "standalone": standalone["score"], "curiosity": curiosity["score"]}
        situation = classify_situation(pairs)
        hook_event, cliff_event = hook["event"], cliff["event"]
        final_line = cliff.get("final_line")
        hook_text = hook_event.text if hook_event else ""
        cliff_text = final_line.text if final_line is not None else (cliff_event.text if cliff_event else "")

        hook_tags = ctx.beats[hook_event.episode][hook_event.index].tags if hook_event else set()
        cliff_tags = ctx.beats[cliff_event.episode][cliff_event.index].tags if cliff_event else set()
        hook_expl = (f"Opens on {_describe_event(hook_event, hook['offset'])}: \"{hook_text[:110]}\". "
                     f"Signals: {', '.join(sorted(hook_tags)) or 'tone only'}.") if hook_event else "No event in the first 3 seconds."
        cliff_expl = (f"Ends at {format_time(candidate.segments[-1].end_s)} on {_describe_event(cliff_event)}: "
                      f"\"{cliff_event.text[:110]}\". Signals: {', '.join(sorted(cliff_tags)) or 'tone only'}"
                      + ("; the writers mark this episode ending as a cliffhanger" if cliff.get("ends_episode") and candidate.segments[-1].episode in ctx.authored_cliffs else "")
                      + ".") if cliff_event else "No closing event found."
        parts = []
        if standalone["roles"]:
            parts.append(f"relationships stated in dialogue: {', '.join(standalone['roles'][:5])}")
        if standalone["debt"]:
            parts.append(f"names introduced in earlier episodes: {', '.join(standalone['debt'][:5])}")
        parts.append(f"{len(standalone['speakers'])} speaker(s)")
        standalone_expl = "; ".join(parts).capitalize() + "."
        why_new = SITUATION_PITCH[situation].capitalize() + "."
        strongest = max(("hook", "cliffhanger", "emotion"), key=lambda k: scores[k])
        why_works = {"hook": "A sharp opening line or action stops the scroll, then ", "cliffhanger": "It cuts before the outcome is known, and ",
                     "emotion": "The emotion climbs across the clip, and "}[strongest] + SITUATION_PITCH[situation] + "."

        weaknesses: list[str] = []
        if hook_event is None or not hook["audible"]:
            weaknesses.append("First 3 seconds rely on visuals only; no line, action or sound.")
        elif hook["offset"] and hook["offset"] > 1.5:
            weaknesses.append(f"Hook lands {hook['offset']:.1f}s in, leaving little margin in the 3-second window.")
        if hook_event is not None and hook_event.type == EVENT_CAMERA:
            weaknesses.append("Hook comes from a camera note and depends on how the shot was rendered.")
        if standalone["debt"]:
            weaknesses.append(f"New viewers will not know: {', '.join(standalone['debt'][:4])}.")
        if standalone["continuation"]:
            weaknesses.append("First spoken line finishes a sentence begun before the clip.")
        if standalone["name_variants"]:
            weaknesses.append(f"Name spelled differently across episodes: {', '.join(standalone['name_variants'][:3])}.")
        if cliff["resolved_ending"]:
            weaknesses.append("Final moments read as warm or settled, which weakens the cliffhanger.")
        if cliff.get("dead_air") and cliff["dead_air"] > 3.5:
            weaknesses.append(f"{cliff['dead_air']:.1f}s with no dialogue after the last line.")
        if standalone["joins"]:
            weaknesses.append("Join between segments is untested; check continuity of costume, location and audio.")
        if max(candidate.episodes) > 0.75 * ctx.last_episode:
            weaknesses.append("Late-season moment: may spoil a major payoff and needs more backstory.")

        return Evaluation(
            candidate_id=candidate.id, method=BASELINE_METHOD, hook=scores["hook"], cliffhanger=scores["cliffhanger"],
            emotion=scores["emotion"], standalone=scores["standalone"], curiosity=scores["curiosity"],
            overall=overall_score(scores), hook_text=hook_text, cliffhanger_text=cliff_text, hook_explanation=hook_expl,
            cliffhanger_explanation=cliff_expl, standalone_explanation=standalone_expl, why_new_viewer=why_new,
            why_it_works=why_works, weaknesses=weaknesses, situation=situation,
            signals={"hook_offset_s": hook["offset"], "hook_event": [hook_event.episode, hook_event.index] if hook_event else None,
                     "dead_air_s": cliff.get("dead_air"), "ends_episode": cliff.get("ends_episode", False),
                     "resolved_ending": cliff["resolved_ending"], "emotion_peak": emotion["peak"], "emotion_arc": emotion["arc"],
                     "backstory_debt": standalone["debt"], "roles": standalone["roles"], "speakers": standalone["speakers"],
                     "open_questions": curiosity["open_questions"], "stakes": curiosity["stakes"],
                     "logline_overlap": curiosity["logline_overlap"]},
        )


LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hook", "cliffhanger", "emotion", "standalone", "curiosity", "hook_explanation", "cliffhanger_explanation",
                 "standalone_explanation", "why_new_viewer", "why_it_works", "weaknesses"],
    "properties": {
        "hook": {"type": "integer"}, "cliffhanger": {"type": "integer"}, "emotion": {"type": "integer"},
        "standalone": {"type": "integer"}, "curiosity": {"type": "integer"},
        "hook_explanation": {"type": "string"}, "cliffhanger_explanation": {"type": "string"},
        "standalone_explanation": {"type": "string"}, "why_new_viewer": {"type": "string"}, "why_it_works": {"type": "string"},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
    },
}

LLM_INSTRUCTIONS = """You are a performance-marketing editor for vertical microdrama apps.
Judge one proposed 20-60 second promo, cut from a screenplay, for someone scrolling a feed who has never seen the show.
Give integers from 0 to 100 for:
- hook: does something in the first 3 seconds make a scroller stop?
- cliffhanger: does it cut at an unresolved moment, so the viewer needs what happens next? A settled or warm ending scores low.
- emotion: how strong and escalating is the feeling on screen?
- standalone: can a new viewer follow who wants what, with no earlier episodes?
- curiosity: how strong is the unanswered story question?
Be strict. Most clips are not good promos. Judge only the clip text given, and quote only from it.
Keep explanations to one sentence each. Return only the JSON object."""


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


class AnthropicClient:
    """Calls Claude through the official SDK. Needs ANTHROPIC_API_KEY and `pip install anthropic`."""

    def __init__(self, model: str | None = None):
        import anthropic  # imported here so the free baseline has no dependency on it

        self.model = model or os.environ.get("PROMOFINDER_MODEL", "claude-opus-5-5")
        self.name = f"anthropic:{self.model}"
        self._client = anthropic.Anthropic()

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        response = self._client.beta.messages.create(
            model=self.model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Model declined this clip")
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


class OpenAICompatibleClient:
    """Any /chat/completions endpoint: a provider's free tier, or an API Kuku supplies."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ["PROMOFINDER_LLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ.get("PROMOFINDER_LLM_API_KEY", "")
        self.model = model or os.environ["PROMOFINDER_MODEL"]
        self.name = f"openai_compatible:{self.model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {"model": self.model, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": f"{system}\nJSON schema: {json.dumps(schema)}"},
                             {"role": "user", "content": user}]}
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return json.loads(payload["choices"][0]["message"]["content"])


def clip_as_text(candidate: Candidate, ctx: ShowContext) -> str:
    lines = []
    for position, segment in enumerate(candidate.segments, 1):
        lines.append(f"[Segment {position}: episode {segment.episode}, {segment.start_raw}-{segment.end_raw}]")
        episode = ctx.episodes[segment.episode]
        for event in (e for e in episode.events if segment.start_s - 0.001 <= e.start_s < segment.end_s - 0.001):
            who = event.character if event.is_spoken else event.type.upper()
            lines.append(f"{event.start_s - segment.start_s:5.1f}s {who}: {event.text}")
    return "\n".join(lines)


class LLMEvaluator(BaseEvaluator):
    """Rules score everything; the model re-scores only the best few, which keeps cost bounded.

    If a call fails, that candidate keeps its rule-based score and is labelled as such.
    """

    name = "llm"

    def __init__(self, client: LLMClient, top_k: int = 60, baseline: BaseEvaluator | None = None):
        self.client, self.top_k = client, top_k
        self.baseline = baseline or HeuristicEvaluator()

    def evaluate(self, candidates: list[Candidate], ctx: ShowContext) -> list[Evaluation]:
        results = {e.candidate_id: e for e in self.baseline.evaluate(candidates, ctx)}
        by_id = {c.id: c for c in candidates}
        shortlist = sorted(results.values(), key=lambda e: e.overall, reverse=True)[: self.top_k]
        for evaluation in shortlist:
            try:
                reply = self.client.complete_json(LLM_INSTRUCTIONS, clip_as_text(by_id[evaluation.candidate_id], ctx), LLM_SCHEMA)
                scores = {k: float(max(0, min(100, reply[k]))) for k in ("hook", "cliffhanger", "emotion", "standalone", "curiosity")}
                results[evaluation.candidate_id] = replace(
                    evaluation, method=f"llm:{self.client.name} on top of heuristic shortlist", overall=overall_score(scores), **scores,
                    hook_explanation=reply["hook_explanation"], cliffhanger_explanation=reply["cliffhanger_explanation"],
                    standalone_explanation=reply["standalone_explanation"], why_new_viewer=reply["why_new_viewer"],
                    why_it_works=reply["why_it_works"], weaknesses=list(reply["weaknesses"]),
                    signals={**evaluation.signals, "baseline_overall": evaluation.overall})
            except Exception as exc:  # one bad response must not lose the whole run
                results[evaluation.candidate_id] = replace(
                    evaluation, method=evaluation.method + " [LLM call failed; rule-based score kept]",
                    signals={**evaluation.signals, "llm_error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return list(results.values())


def build_evaluator(name: str = "heuristic", top_k: int = 60) -> BaseEvaluator:
    if name == "heuristic":
        return HeuristicEvaluator()
    if name == "anthropic":
        return LLMEvaluator(AnthropicClient(), top_k=top_k)
    if name == "openai_compatible":
        return LLMEvaluator(OpenAICompatibleClient(), top_k=top_k)
    raise ValueError(f"Unknown evaluator: {name}. Use heuristic, anthropic or openai_compatible.")
