"""Command-line entry point. Example: python run_pipeline.py --input path/to/scripts_or.zip"""
import argparse
import json
from pathlib import Path

from src.evaluator import build_evaluator
from src.pipeline import DEFAULT_ALIASES, DEFAULT_INPUT, DEFAULT_OUTPUT, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and rank promos in timestamped microdrama screenplays.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Folder of .txt scripts, or a .zip")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--aliases", default=str(DEFAULT_ALIASES), help="Optional JSON mapping speaker labels to one name")
    parser.add_argument("--evaluator", default="heuristic", choices=["heuristic", "anthropic", "openai_compatible"])
    parser.add_argument("--llm-top-k", type=int, default=60, help="How many top candidates the LLM re-scores")
    parser.add_argument("--top", type=int, default=30, help="Length of the final ranked list")
    args = parser.parse_args()
    aliases = args.aliases if Path(args.aliases).exists() else None
    manifest = run_pipeline(args.input, args.output, build_evaluator(args.evaluator, args.llm_top_k), aliases, args.top,
                            progress=lambda message, fraction: print(f"[{fraction:>4.0%}] {message}"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
