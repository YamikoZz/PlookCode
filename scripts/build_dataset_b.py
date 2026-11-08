"""A/B experiment: build SFT dataset B from dataset A by swapping ONLY the templated coding tasks.

Dataset A = the 200M SFT mix (sft-200m.jsonl). Its templated coding tasks (rows with a "family" key or a
"code-*" source: 41 task families with different numbers) are removed and the same number of slots is
filled with verified teacher exercises (scripts/distill_verify.py). Every other row (general chat,
public code instructions, arithmetic, ...) is copied unchanged, so each training step sees the same
share of coding-task rows vs everything else. Teacher rows are taken round-robin over concepts
(variant 1 of every exercise first, then variant 2, ...) so a shortfall never removes whole concepts.

Dataset A is never modified. Reports rows, tokens and distinct concepts for both.

  python scripts/build_dataset_b.py --a sft-200m.jsonl --teacher distill2-verified.jsonl \
      --out sft-200m-B.jsonl --tokenizer run-200m/tokenizer.json
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def is_template(row):
    return "family" in row or str(row.get("source", "")).startswith("code-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    a_rows = [json.loads(l) for l in open(args.a, encoding="utf-8") if l.strip()]
    keep = [r for r in a_rows if not is_template(r)]
    removed = [r for r in a_rows if is_template(r)]
    slots = len(removed)

    by_concept = collections.defaultdict(list)
    for r in (json.loads(l) for l in open(args.teacher, encoding="utf-8") if l.strip()):
        by_concept[r["concept"]].append(r)
    concepts = list(by_concept)
    random.Random(args.seed).shuffle(concepts)
    teacher, seen_q, depth = [], set(), 0
    while len(teacher) < slots and any(len(v) > depth for v in by_concept.values()):
        for c in concepts:
            if depth < len(by_concept[c]) and len(teacher) < slots:
                r = by_concept[c][depth]
                key = "".join(r["question"].casefold().split())
                if key not in seen_q:
                    seen_q.add(key)
                    teacher.append(r)
        depth += 1

    # Fewer distinct teacher rows than slots: give each a "repeat" weight (train split only, see
    # localbrain.instruction) so the coding-task share of training examples matches A.
    base, extra = (slots // len(teacher), slots % len(teacher)) if teacher else (0, 0)
    repeats = [base + (i < extra) for i in range(len(teacher))]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r, n in zip(teacher, repeats):
            f.write(json.dumps({"question": r["question"], "answer": r["answer"], "source": r["source"],
                                "kind": r["kind"], "language": r["language"], "repeat": n},
                               ensure_ascii=False) + "\n")

    def tokens(rows):
        if not args.tokenizer:
            return None
        from tokenizers import Tokenizer
        from localbrain.instruction import prompt_text
        tok = Tokenizer.from_file(args.tokenizer)
        return sum(len(tok.encode(prompt_text(r["question"]) + r["answer"]).ids) for r in rows)

    tk_keep = tokens(keep)
    report = {
        "A": {"rows": len(a_rows), "coding_task_rows": slots,
              "coding_task_concepts": len({r.get("family") or r.get("source") for r in removed}),
              "coding_task_tokens": tokens(removed)},
        "B": {"rows": len(keep) + len(teacher), "coding_task_rows": len(teacher),
              "coding_task_concepts": len({r["concept"] for r in teacher}),
              "coding_task_tokens": tokens(teacher),
              "kinds": dict(collections.Counter(r["kind"] for r in teacher)),
              "languages": dict(collections.Counter(r["language"] for r in teacher))},
        "shared_rows": len(keep), "shared_tokens": tk_keep,
        "distinct_slot_shortfall": slots - len(teacher),
        "repeat_per_teacher_row": f"{base}-{base + 1}" if extra else str(base),
        "coding_task_slots_after_repeat": sum(repeats),
        "bytes_B": Path(args.out).stat().st_size,
    }
    if tk_keep is not None:
        report["A"]["total_tokens"] = tk_keep + report["A"]["coding_task_tokens"]
        report["B"]["total_tokens_distinct"] = tk_keep + report["B"]["coding_task_tokens"]
        report["B"]["coding_task_tokens_after_repeat"] = sum(
            n * t for n, t in zip(repeats, [report["B"]["coding_task_tokens"] / max(1, len(teacher))] * len(teacher)))
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
