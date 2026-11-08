"""A/B experiment summary: one table for both models, including HumanEval on a clean subset.

HumanEval "clean" = problems whose entry-point name never appears as a call (name followed by "(")
in either training file (A or B). Dataset A's templates contain e.g. fib / is_prime / is_palindrome /
greatest_common_divisor, which are HumanEval tasks; scoring only the clean subset keeps the comparison fair.

  python scripts/ab_summary.py --dir /content/drive/MyDrive/LocalBrain-200m/ab \
      --train-a sft-200m.jsonl --train-b ab/sft-200m-B.jsonl --humaneval raw-hf/eval/humaneval.parquet
"""
import argparse
import json
import re
from pathlib import Path


def flagged_names(paths, names):
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\s*\(")
    found = set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            for text in (row.get("question", ""), row.get("answer", "")):
                found.update(pattern.findall(text))
    return found


def counts(report):
    return {k: (round(v["pass@1_greedy"] * v["tasks"]), v["tasks"]) for k, v in report.items()
            if isinstance(v, dict) and "tasks" in v and v.get("pass@1_greedy") is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--train-a", required=True)
    parser.add_argument("--train-b", required=True)
    parser.add_argument("--humaneval", required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    he = pq.read_table(args.humaneval).to_pylist()
    by_id = {p["task_id"]: p["entry_point"] for p in he}
    dirty = flagged_names([args.train_a, args.train_b], set(by_id.values()))
    clean_ids = {t for t, name in by_id.items() if name not in dirty}
    root, summary = Path(args.dir), {"humaneval_clean_problems": len(clean_ids),
                                     "humaneval_flagged_names": sorted(dirty)}
    for model in ("A", "B"):
        out = {}
        for name in ("code-eval", "code-eval-plain", "exam-eval", "exam-eval-plain"):
            path = root / model / f"{name}.json"
            if path.exists():
                rep = json.loads(path.read_text(encoding="utf-8"))
                out[name] = {"step": rep["checkpoint_step"], **{k: f"{p}/{n}" for k, (p, n) in counts(rep).items()}}
        for name in ("humaneval", "humaneval-plain"):
            path = root / model / f"{name}.json"
            if path.exists():
                rep = json.loads(path.read_text(encoding="utf-8"))
                res = {"step": rep["checkpoint_step"]}
                for fmt in ("complete", "instruct"):
                    rows = [r for r in rep["rows"] if r["format"] == fmt]
                    if rows:
                        res[fmt] = f"{sum(r['status'] == 'pass' for r in rows)}/{len(rows)}"
                        clean = [r for r in rows if r["task_id"] in clean_ids]
                        res[fmt + "_clean"] = f"{sum(r['status'] == 'pass' for r in clean)}/{len(clean)}"
                out[name] = res
        summary[model] = out
    text = json.dumps(summary, indent=2)
    (root / "summary.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
