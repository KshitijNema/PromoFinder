"""Unblind the review: compare verdicts for top-ranked clips with verdicts for random candidates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "outputs"
read = lambda name: json.loads((ROOT / name).read_text(encoding="utf-8"))
SCORE = {"post": 2, "fix": 1, "reject": 0}


def main() -> None:
    review, key = read("06_review_verdicts.json"), read("06_review_key.json")
    scores = {s["candidate_id"]: s for s in read("03_scores.json")}
    ranked = {r["candidate_id"]: r for r in read("04_ranked_promos.json")}
    rows = []
    for item in (k for k in key if k["group"] != "former_top"):
        verdict, score = review["verdicts"][item["code"]], scores[item["candidate_id"]]
        rows.append({"code": item["code"], "group": item["group"], "candidate_id": item["candidate_id"], "rank": ranked.get(item["candidate_id"], {}).get("rank"),
                     "system_score": score["overall"], "system_hook": score["hook"], "system_cliffhanger": score["cliffhanger"], "system_standalone": score["standalone"], **verdict})

    def stats(group: str) -> dict:
        items = [r for r in rows if r["group"] == group]
        total = len(items)
        return {"clips": total, "post": sum(r["verdict"] == "post" for r in items), "fix": sum(r["verdict"] == "fix" for r in items), "reject": sum(r["verdict"] == "reject" for r in items),
                "hook_pass": sum(r["hook"] for r in items), "cliff_pass": sum(r["cliff"] for r in items), "standalone_pass": sum(r["standalone"] for r in items),
                "mean_system_score": round(sum(r["system_score"] for r in items) / total, 1)}

    top, random_group = stats("top"), stats("random")
    ordered = sorted(rows, key=lambda r: r["system_score"])
    ranks = {r["code"]: i for i, r in enumerate(ordered)}
    by_verdict = sorted(rows, key=lambda r: SCORE[r["verdict"]])
    tie_rank: dict[str, float] = {}
    for level in SCORE.values():
        members = [i for i, r in enumerate(by_verdict) if SCORE[r["verdict"]] == level]
        for i in members:
            tie_rank[by_verdict[i]["code"]] = sum(members) / len(members)
    xs, ys = [ranks[r["code"]] for r in rows], [tie_rank[r["code"]] for r in rows]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = (sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)) ** 0.5
    spearman = round(numerator / denominator, 2)

    false_positive = sorted((r for r in rows if r["group"] == "top" and r["verdict"] != "post"), key=lambda r: -r["system_score"])
    false_negative = [r for r in rows if r["group"] == "random" and r["verdict"] == "post"]
    headline = (f"Top 20 by the system: {top['post']} usable as cut, {top['fix']} need new boundaries or context, {top['reject']} rejected. "
                f"20 random candidates: {random_group['post']} usable, {random_group['fix']} need fixes, {random_group['reject']} rejected. "
                f"Rank correlation between system score and verdict: {spearman}. Reviewed by Claude from script text, partly blind, one show, no viewer data.")
    results = {"reviewer": review["reviewer"], "rubric": review["rubric"], "headline": headline, "top": top, "random": random_group, "spearman_score_vs_verdict": spearman,
               "false_positives_in_top": false_positive, "good_clips_the_ranker_missed": false_negative,
               "table": [{"Code": r["code"], "Group": r["group"], "Rank": r["rank"], "Score": r["system_score"], "Hook": r["hook"], "Cliffhanger": r["cliff"], "Standalone": r["standalone"], "Verdict": r["verdict"], "Note": r["note"]} for r in sorted(rows, key=lambda r: -r["system_score"])]}
    (ROOT / "06_review_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(headline)
    print("top", top)
    print("random", random_group)
    for r in results["table"]:
        print(r["Code"], r["Group"], r["Rank"], r["Score"], r["Hook"], r["Cliffhanger"], r["Standalone"], r["Verdict"])


if __name__ == "__main__":
    main()
