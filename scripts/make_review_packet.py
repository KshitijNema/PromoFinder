"""Build a shuffled, score-free packet so reviewers judge clips without seeing the system's ranking.

The packet mixes the top ranked promos with random candidates from the rest of the pool.
The answer key is written separately so the two groups can be compared afterwards.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import Candidate, Segment  # noqa: E402
from src.parser import apply_character_aliases, load_aliases, parse_folder  # noqa: E402
from src.pipeline import DEFAULT_ALIASES, DEFAULT_INPUT, DEFAULT_OUTPUT  # noqa: E402

TOP, RANDOM_COUNT, SEED = 20, 20, 2026


def main() -> None:
    out = DEFAULT_OUTPUT
    ranked = json.loads((out / "04_ranked_promos.json").read_text(encoding="utf-8"))
    pool = json.loads((out / "02_candidates.json").read_text(encoding="utf-8"))
    episodes, _ = parse_folder(DEFAULT_INPUT)
    apply_character_aliases(episodes, load_aliases(DEFAULT_ALIASES))
    index = {e.number: e for e in episodes}
    top_ids = [p["candidate_id"] for p in ranked[:TOP]]
    rng = random.Random(SEED)
    rest = [c for c in pool if c["id"] not in set(top_ids)]
    random_items = rng.sample(rest, RANDOM_COUNT)
    by_id = {c["id"]: c for c in pool}
    items = [(cid, "top") for cid in top_ids] + [(c["id"], "random") for c in random_items]
    rng.shuffle(items)
    lines, key = [], []
    for number, (cid, group) in enumerate(items, 1):
        code = f"R{number:02d}"
        key.append({"code": code, "candidate_id": cid, "group": group})
        lines.append(f"## {code}")
        for segment in by_id[cid]["segments"]:
            lines.append(f"[segment: ep {segment['episode']}]")
            for event in index[segment["episode"]].events:
                if segment["start_s"] - 0.001 <= event.start_s < segment["end_s"] - 0.001 and event.type != "camera":
                    who = event.character.title() if event.is_spoken else event.type.upper()
                    lines.append(f"{event.start_s - segment['start_s']:.1f} {who}: {event.text[:140]}")
        lines.append("")
    (out / "06_review_packet.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "06_review_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} clips written")


if __name__ == "__main__":
    main()
