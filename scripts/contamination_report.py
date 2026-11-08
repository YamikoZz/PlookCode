"""Text-only contamination report: does an SFT jsonl overlap HumanEval, MBPP, our exam or held-out tasks?

Checks per row (question text; answers are not compared because code boilerplate is shared everywhere):
  - shares a word 13-gram with a benchmark task text (HumanEval prompt incl. docstring, MBPP text,
    exam task description)
  - calls a benchmark function name, e.g. "has_close_elements(" (HumanEval entry points, exam and
    held-out family names), in question or answer
No code is executed.

  python scripts/contamination_report.py data.jsonl [more.jsonl ...] --raw workspace/raw-hf
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from import_code_sft import ngrams, words  # noqa: E402


def benchmarks(raw):
    import pyarrow.parquet as pq
    from localbrain.exam import FAMILIES as EXAM, exam
    from localbrain.tasks import FAMILIES, HELD_OUT
    he = pq.read_table(raw / "eval" / "humaneval.parquet").to_pylist()
    mbpp = []
    for split in ("train", "test", "validation", "prompt"):
        mbpp += [r["text"] for r in pq.read_table(raw / "eval" / f"mbpp-{split}.parquet").to_pylist()]
    sets = {
        "humaneval": ({g for p in he for g in ngrams(p["prompt"])}, {p["entry_point"] for p in he}),
        "mbpp": ({g for t in mbpp if len(words(t)) >= 13 for g in ngrams(t)}, set()),
        "exam": ({g for t in exam() for g in ngrams(t["prompt"].split(" that ", 1)[1].split("\n")[0])},
                 set(EXAM)),
        "held_out": (set(), {n for fam in HELD_OUT for n in FAMILIES[fam]["names"]}),
    }
    return sets


def report(path, sets):
    patterns = {k: re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\s*\(") if names else None
                for k, (_, names) in sets.items()}
    hits, samples, total = collections.Counter(), collections.defaultdict(list), 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row, total = json.loads(line), total + 1
        q = row.get("question", "")
        grams = ngrams(q)
        for key, (bench_grams, _) in sets.items():
            reason = None
            if bench_grams and grams & bench_grams:
                reason = "13gram"
            elif patterns[key] and (patterns[key].search(q) or patterns[key].search(row.get("answer", ""))):
                reason = "name"
            if reason:
                hits[f"{key}/{reason}"] += 1
                if len(samples[key]) < 3:
                    samples[key].append(q[:160])
    return {"file": str(path), "rows": total, "hits": dict(hits), "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--raw", default="workspace/raw-hf")
    parser.add_argument("--output")
    args = parser.parse_args()
    sets = benchmarks(Path(args.raw))
    out = [report(Path(f), sets) for f in args.files]
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
