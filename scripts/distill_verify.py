"""Distillation step 2: keep only teacher exercises whose reference solution passes its own tests.

Runs the teacher's code ONLY in a sandboxed environment (Colab), never on the home PC (CLAUDE.md rule 2),
through localbrain.runner.run_tests (static checks + subprocess limits). Then drops:
  - names of exam / held-out tasks, and questions sharing a 13-gram with HumanEval/MBPP/exam text
  - near-duplicates by question skeleton (names and digits removed) or by renamed solution body
and writes {"question", "answer"} SFT rows in the same prompt style as the exam / code tasks.

  python scripts/distill_verify.py --raw /content/distill-raw.jsonl --evalraw /content/raw-hf \
      --out /content/distill-verified.jsonl
"""
import argparse
import collections
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from localbrain.exam import literal  # noqa: E402
from localbrain.runner import run_tests  # noqa: E402

LANGS = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript", "lua": "Lua"}
FENCE = {"python": "python", "javascript": "javascript", "typescript": "typescript", "lua": "lua"}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def parse(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def plain_value(v, depth=0):
    if isinstance(v, bool) or isinstance(v, int) or isinstance(v, str):
        return not isinstance(v, str) or len(v) <= 200
    if isinstance(v, list) and depth < 2 and len(v) <= 40:
        return all(plain_value(x, depth + 1) for x in v)
    return False


def shape_ok(item, language):
    if not isinstance(item, dict):
        return "not_object"
    fn, desc, sol, tests = item.get("function"), item.get("description"), item.get("solution"), item.get("tests")
    params = item.get("params")
    if not (isinstance(fn, str) and NAME.fullmatch(fn)):
        return "bad_name"
    if not (isinstance(desc, str) and 10 <= len(desc) <= 400):
        return "bad_description"
    if not (isinstance(sol, str) and fn in sol and len(sol) <= 4000):
        return "bad_solution"
    if not (isinstance(params, list) and all(isinstance(p, dict) and isinstance(p.get("name"), str)
                                             and NAME.fullmatch(p["name"]) for p in params)):
        return "bad_params"
    if not (isinstance(tests, list) and 4 <= len(tests) <= 12):
        return "bad_tests"
    for t in tests:
        if not (isinstance(t, dict) and isinstance(t.get("args"), list) and "expected" in t
                and len(t["args"]) == len(params) and plain_value(t["args"]) and plain_value(t["expected"])):
            return "bad_test_values"
    if language == "lua" and re.search(r"\b(require|io|os|load|dofile|debug)\b", sol):
        return "lua_unsafe"
    return None


def question(item, language):
    fn, params = item["function"], item["params"]
    if language == "typescript":
        sig = f"{fn}({', '.join(p['name'] + ': ' + str(p.get('type', 'any')) for p in params)}): {item.get('returns', 'any')}"
    else:
        sig = f"{fn}({', '.join(p['name'] for p in params)})"
    desc = item["description"].strip().rstrip(".")
    desc = re.sub(r"^(write a function that|a function that)\s+", "", desc, flags=re.I)
    note = " Lists are Lua array tables." if language == "lua" else ""
    plain = [t for t in item["tests"] if all(a not in ([], "") for a in t["args"])] or item["tests"]
    ex = plain[0]
    shown = ", ".join(literal(a, language) for a in ex["args"])
    return (f"Write a {LANGS[language]} function `{sig}` that {desc}.{note}\n"
            f"Example: {fn}({shown}) returns {literal(ex['expected'], language)}")


def skeleton(item):
    text = item["description"].casefold().replace(item["function"].casefold(), "F")
    text = re.sub(r"\d+", "N", text)
    return " ".join(re.findall(r"[a-z]+|N", text))


def body_key(item):
    return re.sub(r"\s+", " ", item["solution"].replace(item["function"], "F")).strip()


def check(row):
    item = parse(row["raw"])
    language = row["language"]
    reason = shape_ok(item, language)
    if reason:
        return row, item, reason, None
    result = run_tests(language, item["solution"], item["function"], item["tests"], timeout=5)
    if result["status"] != "pass":
        return row, item, "tests_" + result["status"], result.get("message", "")[:200]
    return row, item, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/content/distill-raw.jsonl")
    parser.add_argument("--evalraw", default="/content/raw-hf")
    parser.add_argument("--out", default="/content/distill-verified.jsonl")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    from import_code_sft import eval_ngrams, held_out_names, overlaps_eval
    contaminated, names = eval_ngrams(Path(args.evalraw)), held_out_names()
    rows = [json.loads(line) for line in open(args.raw, encoding="utf-8") if line.strip()]
    dropped, kept, seen_sk, seen_body = collections.Counter(), [], set(), set()
    samples = collections.defaultdict(list)
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for row, item, reason, detail in pool.map(check, rows):
            language = row["language"]
            if not reason and item["function"] in names:
                reason = "held_out_name"
            if not reason:
                q = question(item, language)
                if overlaps_eval(q, contaminated):
                    reason = "eval_overlap"
                elif skeleton(item) in seen_sk or body_key(item) in seen_body:
                    reason = "near_duplicate"
            if reason:
                dropped[(language, reason)] += 1
                if detail and len(samples[reason]) < 3:
                    samples[reason].append(detail)
                continue
            seen_sk.add(skeleton(item))
            seen_body.add(body_key(item))
            kept.append({"question": q, "answer": f"```{FENCE[language]}\n{item['solution'].strip()}\n```",
                         "source": "distill-qwen2.5-coder-32b", "language": language, "topic": row["topic"],
                         "function": item["function"], "tests": item["tests"]})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"raw": len(rows), "kept": len(kept),
              "kept_by_language": dict(collections.Counter(r["language"] for r in kept)),
              "dropped": {f"{l}/{r}": n for (l, r), n in sorted(dropped.items())},
              "error_samples": samples}
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    main()
