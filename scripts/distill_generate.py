"""Distillation data, step 1 (runs on a Colab GPU): an open-source teacher writes coding exercises.

Teacher: Qwen/Qwen2.5-Coder-32B-Instruct (Apache-2.0). Only its *outputs* are kept; its weights never
enter PlookCode (CLAUDE.md rule 1). Every exercise is later checked by running code on its tests
(scripts/distill_verify.py), so the teacher's word is never trusted on its own.

Four exercise kinds (dataset B of the A/B experiment, see STATE.md):
  write   : function from a description                      -> code (tests must pass)
  plan    : function from a description                      -> short plan, then code (tests must pass)
  debug   : description + buggy code                          -> bug explanation + fixed code
                                                                (fixed passes, buggy fails a test)
  predict : code + one call                                   -> short trace + return value
                                                                (the code must reproduce every value)
Answers come in tagged sections (META / PLAN / BUGGY / BUG / TRACE / CODE / TESTS) instead of one JSON
object, because code inside JSON strings broke ~20% of the pilot outputs.

  python scripts/distill_generate.py --n 20000 --out /content/distill2-raw.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LANGS = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript", "lua": "Lua"}
LANG_WEIGHTS = {"python": 0.30, "javascript": 0.25, "typescript": 0.20, "lua": 0.25}
KIND_WEIGHTS = {"write": 0.35, "plan": 0.30, "debug": 0.20, "predict": 0.15}
TOPICS = [
    "counting elements that satisfy a condition", "running / prefix sums", "sliding window over a list",
    "two pointers on a sorted list", "grouping consecutive equal items", "rotating or shifting a list",
    "merging two sorted lists", "local extremes in a sequence", "pairs that meet a condition",
    "matrix (list of lists) row/column processing", "grid neighbours and borders", "diagonals of a square grid",
    "string character frequency", "string compression or expansion", "word-level string processing",
    "case conversion and identifier formatting", "parsing digits inside a string", "palindromes and symmetry",
    "simple ciphers and character shifting", "letter classes (vowels, consonants, digits)",
    "bracket / balance checking", "stack-based processing", "queue / round-robin simulation",
    "intervals: overlap, merge, cover", "number theory: divisors, multiples, remainders", "digits of an integer",
    "binary representation and bit counts", "base conversion", "integer sequences defined by a rule",
    "rounding to multiples", "greedy choice on a list", "small dynamic programming on a list",
    "longest run / streak", "deduplication keeping order", "ranking and ordering by a key", "top-k selection",
    "simulation of a simple game or process", "scoring rules (points per item)", "calendar-like integer math",
    "inventory or stock bookkeeping with lists", "histogram / bucket counting", "interleaving and zipping lists",
    "chunking a list into groups", "filtering and transforming with rules", "set operations on lists",
    "checking ordering properties of a list", "distance and differences between elements",
    "encoding a list as a string and back", "voting / majority counting", "path walking on a 1D track",
    "time and duration arithmetic with integers", "price, discount and tax arithmetic with integers (cents)",
    "string templating and padding", "run-length style encodings", "counting substrings or patterns",
]
DIFFICULTY = ["easy", "easy", "medium", "medium", "medium", "harder"]
SEEDS = ["river", "orbit", "lantern", "harbor", "meadow", "circuit", "glacier", "market", "puzzle", "garden",
         "signal", "ladder", "mosaic", "tunnel", "compass", "festival", "orchard", "beacon", "canyon", "relay",
         "quarry", "tide", "pixel", "caravan", "ember", "summit", "ferry", "hive", "loom", "prism", "atlas",
         "bakery", "chess", "dune", "elevator", "forest", "guitar", "island", "jungle", "kettle", "library"]
LANG_NOTES = {
    "python": "Plain Python 3. Imports allowed only from: math, itertools, functools, collections, string, re, "
              "heapq, bisect, operator.",
    "javascript": "A plain JavaScript function declaration (`function name(...) {...}`). No require/import, "
                  "no process, no console output.",
    "typescript": "A TypeScript function declaration with type annotations on every parameter and on the return "
                  "type. No imports, classes, namespaces or enums.",
    "lua": "Lua 5.4, written as `local function name(...) ... end`. Lists are Lua array tables indexed from 1. "
           "No require, io, os, load or debug.",
}
KIND_TASK = {
    "write": "Write a function exercise: a clear description and a correct solution.",
    "plan": ("Write a function exercise whose solution needs a little thinking. Before the code, give a short "
             "plan of 2-4 numbered steps that explains HOW to solve it (the idea, not a restatement)."),
    "debug": ("Write a function exercise, then a BUGGY version of the same function containing exactly one "
              "realistic bug (off-by-one, wrong comparison, wrong variable, missing case...). The buggy version "
              "must fail at least one of your tests. CODE must be the corrected function."),
    "predict": ("Write a short function (5-15 lines, with loops or conditions, not a one-liner) for a "
                "code-reading exercise. In TRACE, explain in 1-3 sentences how the function computes its result "
                "for the FIRST test (do not just restate the answer)."),
}
SECTIONS = {
    "write": ["META", "CODE", "TESTS"],
    "plan": ["META", "PLAN", "CODE", "TESTS"],
    "debug": ["META", "BUGGY", "BUG", "CODE", "TESTS"],
    "predict": ["META", "TRACE", "CODE", "TESTS"],
}
SECTION_HELP = {
    "META": ('### META\n{"function": "<name>", "params": [{"name": "<param>", "type": "<TypeScript type>"}], '
             '"returns": "<TypeScript type>", "description": "<completes the sentence \'Write a function that '
             '...\', e.g. \'returns the number of ...\'>"}'),
    "PLAN": "### PLAN\n1. ...\n2. ...",
    "BUGGY": "### BUGGY\n```<language>\n<the buggy function>\n```",
    "BUG": "### BUG\n<one sentence: what is wrong and how it was fixed>",
    "TRACE": "### TRACE\n<1-3 sentences>",
    "CODE": "### CODE\n```<language>\n<the correct function>\n```",
    "TESTS": ('### TESTS\n[{"args": [<arg1>, <arg2>], "expected": <value>}, ...]   '
              '(6 tests, JSON, args in parameter order)'),
}


def banned():
    """Exam + held-out families (name and description) and HumanEval entry points: never to be imitated."""
    from localbrain.exam import F as EXAM
    from localbrain.tasks import FAMILIES, HELD_OUT
    lines = [f"- {f['name']}: {f['desc']}" for f in EXAM]
    for fam in sorted(HELD_OUT):
        f = FAMILIES[fam]
        desc = f["descs"][0] if f.get("descs") else fam
        lines.append(f"- {', '.join(f['names'])}: {desc.replace('$', '')}")
    names = []
    path = ROOT / "workspace" / "raw-hf" / "eval" / "humaneval.parquet"
    for candidate in (path, Path("/content/raw-hf/eval/humaneval.parquet")):
        if candidate.exists():
            import pyarrow.parquet as pq
            names = sorted({r["entry_point"] for r in pq.read_table(candidate).to_pylist()})
            break
    return lines, names


def system_prompt():
    lines, he_names = banned()
    return ("You are an expert programming teacher. You write original, self-contained coding exercises with "
            "correct solutions and exact, carefully checked test cases. Follow the requested section format "
            "exactly.\n\nNEVER write an exercise that is the same task as (or a trivial variation of) any of "
            "these reserved test tasks:\n" + "\n".join(lines) +
            "\nAlso never reuse these function names or their tasks: " + ", ".join(he_names) +
            "\nand never reproduce well-known benchmark problems (HumanEval, MBPP, LeetCode classics).")


def user_prompt(kind, lang, topic, difficulty, seed):
    fmt = "\n\n".join(SECTION_HELP[s] for s in SECTIONS[kind]).replace("<language>", lang.lower())
    return f"""Exercise kind: {kind}. {KIND_TASK[kind]}
Language: {LANGS[lang]}
Topic: {topic}
Difficulty: {difficulty}
Inspiration word: {seed} (only to make the exercise different from textbook problems)

Rules:
- Exactly one function; snake_case name for Python/Lua, camelCase or snake_case for JS/TS.
- Parameters and return value may only be integers, strings, booleans, lists of integers, lists of strings,
  or lists of lists of integers. No floats, no dictionaries/objects, no null/None.
- {LANG_NOTES[lang]}
- Pure function: no input/output, no printing, no randomness.
- 6 tests with exact expected values, including edge cases (empty list or string, single element, negatives
  where they make sense). Compute every expected value carefully.

Answer with exactly these sections and nothing else:

{fmt}"""


def requests(n, seed):
    rng = random.Random(seed)
    langs, lw = zip(*LANG_WEIGHTS.items())
    kinds, kw = zip(*KIND_WEIGHTS.items())
    system = system_prompt()
    for i in range(n):
        meta = {"id": i, "kind": rng.choices(kinds, kw)[0], "language": rng.choices(langs, lw)[0],
                "topic": rng.choice(TOPICS), "difficulty": rng.choice(DIFFICULTY), "seed": rng.choice(SEEDS)}
        text = user_prompt(meta["kind"], meta["language"], meta["topic"], meta["difficulty"], meta["seed"])
        yield meta, [{"role": "system", "content": system}, {"role": "user", "content": text}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", default="/content/distill2-raw.jsonl")
    parser.add_argument("--max-model-len", type=int, default=6144)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1500, seed=args.seed)
    reqs = list(requests(args.n, args.seed))
    outputs = llm.chat([m for _, m in reqs], params, use_tqdm=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for (meta, _), out in zip(reqs, outputs):
            f.write(json.dumps({**meta, "raw": out.outputs[0].text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(reqs)} rows to {args.out}")


if __name__ == "__main__":
    main()
