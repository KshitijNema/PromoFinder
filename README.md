# PromoFinder

PromoFinder reads timestamped microdrama screenplays and finds the best 20–60 second promos. For each promo it gives the episode, the start and end timestamp of every segment, the hook, the cliffhanger, a score breakdown and its known weaknesses. It then ranks them into a varied posting list for an editor to confirm.

The default evaluator uses deterministic rules and needs no API key or paid service. Its scores order clips for an editor. They have not been validated against viewer data and do not predict performance. An optional LLM evaluator can re-score the shortlist.

> **Confidential input.** Screenplays and everything derived from them stay out of version control. `.gitignore` excludes `data/*`, `outputs/*`, and the review, walkthrough and PDF files in `docs/`. Tests use invented scripts only.

## Quick start

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt

# Put the .txt scripts anywhere under data/raw/, or pass a folder or .zip with --input
python run_pipeline.py
python run_pipeline.py --input path/to/new_show.zip --output outputs/new_show

streamlit run app.py     # dashboard at http://localhost:8501
python -m pytest tests   # uses invented scripts, no confidential data needed
```

A new show can also be analysed from the dashboard: choose **Upload a ZIP** in the sidebar, then **Run pipeline**.

## How it works

```
.txt scripts or .zip
   -> parser.py               timed events, scenes, speakers, source line numbers, format issues
   -> scorer.py               per-event signals; hook, cliffhanger, emotion, standalone, curiosity
   -> candidate_generator.py  continuous clips, same-episode montages, cross-episode montages
   -> evaluator.py            rule baseline, or rules plus an LLM on the shortlist
   -> quality_checks.py       re-reads the source files; blocks anything that breaks the brief
   -> ranking.py              greedy varied selection
   -> outputs/ + app.py       JSON, CSV, dashboard
```

**Scoring.** Overall = 25% hook + 30% cliffhanger + 20% emotion + 15% standalone + 10% curiosity. Keywords are only one input. The rules also use:

- the script's own performance notes, such as "Furious" or "Gentle"
- whether a moment is a spike compared with the previous 15 seconds
- how strong the line withheld by the cut is, so an open question is only worth what it hides
- whether names in the clip were introduced in earlier episodes
- whether the final moments read as settled

**Boundaries.** Every start and end comes from a real event in the script, so each can be traced to a line in the source file. Cuts prefer shot boundaries and never split a line of dialogue where that can be avoided.

**Ranking.** The list penalises reused episodes, adjacent episodes, repeated situations and join risk. It gives a small lift to early-story promos and a small penalty to late-season spoilers. Each promo lists the adjustments applied to it.

## Features

- Parser tolerant of label case, bold markers and spacing, which reports what it had to repair
- Continuous and multi-segment promos, with the order and timestamps of each segment
- Five component scores with written explanations and weaknesses for every candidate
- Automated checks: duration, hook window, timestamp order, episode references, missing dialogue or action, traceability to the source, duplicates and overlap, timing precision
- Dashboard: overview, filters, promo cards, script excerpts, score breakdown, checks, CSV and JSON export, rerun, ZIP upload

## Using an LLM instead of the baseline

The rules score every candidate, then the model re-scores the top 60. That keeps cost bounded and gives each failed call a rule-based fallback, labelled as such in the output.

```bash
# Claude, through the official SDK
pip install anthropic
export ANTHROPIC_API_KEY=...
export PROMOFINDER_MODEL=claude-opus-5-5   # optional; this is the default
python run_pipeline.py --evaluator anthropic --llm-top-k 60

# Any OpenAI-compatible /chat/completions endpoint
export PROMOFINDER_LLM_BASE_URL=https://provider.example/v1
export PROMOFINDER_LLM_API_KEY=...
export PROMOFINDER_MODEL=model-name
python run_pipeline.py --evaluator openai_compatible
```

To add another provider, write a class with `name` and `complete_json(system, user, schema)` in `src/evaluator.py`, then register it in `build_evaluator`. To replace the rules entirely, subclass `BaseEvaluator` and return one `Evaluation` per candidate.

The Claude adapter enables server-side model fallback so a refused clip can be retried on another model. Remove `betas` and `fallbacks` in `AnthropicClient.complete_json` to turn that off.

**Status:** both adapters are tested against fake responses only. Neither has been run against a live model.

## Show-specific names (optional)

Scripts often give one character several labels. `data/aliases.json` merges them for variety checks and display. See `data/aliases.example.json`. The pipeline runs without this file.

## Output format

| File | Contents |
|---|---|
| `outputs/01_parsed_episodes.json` | Every scene and timed event, with original timestamps and source lines |
| `outputs/01_parse_issues.json` | Format problems, unreadable files, missing episodes, spelling drift |
| `outputs/02_candidates.json` | All candidates |
| `outputs/03_scores.json`, `.csv` | All scores, explanations and signals |
| `outputs/04_ranked_promos.json`, `.csv` | Final ranked list with excerpts and checks |
| `outputs/05_quality_report.json` | Per-candidate checks and summaries |
| `outputs/06_review_*` | Review packet, answer key, verdicts and results |
| `outputs/run_manifest.json` | Versions, input hash, counts, timing |

Shape of one ranked promo, with invented values:

```json
{
  "rank": 1, "candidate_id": "C001-00001", "kind": "continuous", "duration_s": 42.5,
  "segments": [{"order": 1, "episode": 1, "start_raw": "00:39.00", "end_raw": "01:21.50", "start_line": 210, "end_line": 402}],
  "scores": {"hook": 90.0, "cliffhanger": 80.0, "emotion": 70.0, "standalone": 75.0, "curiosity": 60.0, "overall": 78.5},
  "hook_text": "...", "cliffhanger_text": "...", "why_it_works": "...", "weaknesses": ["..."],
  "method": "heuristic_baseline_v1 (...)", "quality_checks": {"duration_20_60s": {"status": "pass", "detail": "42.50s"}}
}
```

## Folder structure

```
PromoFinder/
  app.py                  dashboard
  run_pipeline.py         command line
  requirements.txt
  src/                    parser, scorer, candidate_generator, evaluator, ranking, quality_checks, pipeline, models
  scripts/                review packet and review scoring (the PDF builder is private)
  tests/                  invented scripts only
  data/                   private input (git-ignored)
  outputs/                private results (git-ignored)
  docs/                   private solution PDF, review notes, walkthrough (git-ignored)
```

## Limitations

- It reads scripts, not video. Timestamps are assumed to match the final cut, and performance, music and picture are invisible to it.
- Rules were written and reviewed on one show. They rely on this format's performance notes and on English vocabulary.
- The hook check confirms that a line or action starts within 3 seconds. Whether it is compelling is estimated by the score and needs human review.
- Joined segments are not checked for visual or audio continuity.
- Scores have not been compared with real promo results.
