"""Distillation step 1 (runs on Colab GPU): an open-source teacher invents coding exercises.

Teacher: Qwen/Qwen2.5-Coder-32B-Instruct (Apache-2.0). Only its *outputs* (exercises + reference
solutions + tests) are kept; its weights never enter PlookCode (CLAUDE.md rule 1). Every row is
checked later by actually running the solution on its tests (scripts/distill_verify.py).

  python scripts/distill_generate.py --n 6000 --out /content/distill-raw.jsonl
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
TOPICS = [
    "counting elements that satisfy a condition", "running / prefix sums", "sliding window over a list",
    "two pointers on a sorted list", "grouping consecutive equal items", "rotating or shifting a list",
    "merging two sorted lists", "finding peaks, valleys or local extremes", "pairs that meet a condition",
    "matrix (list of lists) row/column processing", "grid neighbours and borders", "diagonals of a square grid",
    "string character frequency", "string compression or expansion", "word-level string processing",
    "case conversion and identifier formatting", "parsing digits inside a string", "palindromes and symmetry",
    "simple ciphers and character shifting", "vowels, consonants and letter classes", "bracket / balance checking",
    "stack-based processing", "queue / round-robin simulation", "intervals: overlap, merge, cover",
    "number theory: divisors, primes, gcd", "digits of an integer", "binary representation and bit counts",
    "base conversion", "integer sequences (Fibonacci-like, arithmetic, geometric)", "rounding to multiples",
    "greedy choice on a list", "small dynamic programming on a list", "longest run / streak",
    "deduplication keeping order", "ranking and ordering by a key", "top-k selection",
    "simulation of a simple game or process", "scoring rules (points per item)", "calendar-like integer math",
    "inventory or stock bookkeeping with lists", "histogram / bucket counting", "interleaving and zipping lists",
    "chunking a list into groups", "filtering and transforming with rules", "set operations on lists",
    "checking ordering properties of a list", "distance and differences between elements",
    "encoding a list as a string and back", "voting / majority counting", "path walking on a 1D track",
]
DIFFICULTY = ["easy", "easy", "medium", "medium", "medium", "harder"]
SEEDS = ["river", "orbit", "lantern", "harbor", "meadow", "circuit", "glacier", "market", "puzzle", "garden",
         "signal", "ladder", "mosaic", "tunnel", "compass", "festival", "orchard", "beacon", "canyon", "relay",
         "quarry", "tide", "pixel", "caravan", "ember", "summit", "ferry", "hive", "loom", "prism"]
LANG_NOTES = {
    "python": "Write plain Python 3. You may import only from: math, itertools, functools, collections, "
              "string, re, heapq, bisect, operator.",
    "javascript": "Write a plain JavaScript function declaration (`function name(...) {...}`). "
                  "No require/import, no process, no console output.",
    "typescript": "Write a TypeScript function declaration with type annotations on every parameter and the "
                  "return type. No imports, no classes, no namespaces, no enums.",
    "lua": "Write Lua 5.4 as `local function name(...) ... end`. Lists are Lua array tables indexed from 1. "
           "Do not use require, io, os, load, string.format with %q, or global state.",
}
SYSTEM = ("You are an expert programming teacher. You write original, self-contained coding exercises with "
          "correct reference solutions and exact, carefully checked test cases. You answer with JSON only.")
TEMPLATE = """Invent ONE original coding exercise.
Language: {lang}
Topic: {topic}
Difficulty: {difficulty}
Inspiration word: {seed} (use it only to make the exercise different from common textbook problems)

Rules:
- Exactly one function. Do not use any of these names: {banned}.
- Parameters and return value may only be: integers, strings, booleans, lists of integers, lists of strings,
  or lists of lists of integers. No floats, no dictionaries/objects, no null/None.
- {note}
- The function must be pure: no input/output, no printing, no randomness.
- Give 6 test cases with exact expected values, including edge cases (empty list or empty string,
  single element, negative numbers where they make sense). Compute each expected value carefully.
- Do not reproduce well-known benchmark problems.

Return ONLY this JSON object (no markdown fences, no comments):
{{"function": "<name>", "params": [{{"name": "<param>", "type": "<TypeScript type, e.g. number[]>"}}],
 "returns": "<TypeScript return type>",
 "description": "<completes the sentence 'Write a function that ...', e.g. 'returns the number of ...'>",
 "solution": "<complete code of the function in {lang}>",
 "tests": [{{"args": [<arg1>, <arg2>], "expected": <value>}}, ...]}}
"args" lists the arguments in parameter order; "expected" is the exact return value."""


def banned_names():
    from localbrain.exam import FAMILIES as EXAM
    from localbrain.tasks import FAMILIES, HELD_OUT
    names = set(EXAM) | {n for fam in HELD_OUT for n in FAMILIES[fam]["names"]}
    return sorted(names)


def requests(n, seed):
    rng = random.Random(seed)
    banned = ", ".join(banned_names())
    langs, weights = zip(*LANG_WEIGHTS.items())
    for i in range(n):
        lang = rng.choices(langs, weights)[0]
        meta = {"id": i, "language": lang, "topic": rng.choice(TOPICS), "difficulty": rng.choice(DIFFICULTY),
                "seed": rng.choice(SEEDS)}
        text = TEMPLATE.format(lang=LANGS[lang], topic=meta["topic"], difficulty=meta["difficulty"],
                               seed=meta["seed"], banned=banned, note=LANG_NOTES[lang])
        yield meta, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="/content/distill-raw.jsonl")
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1200, seed=args.seed)
    reqs = list(requests(args.n, args.seed))
    outputs = llm.chat([m for _, m in reqs], params, use_tqdm=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for (meta, _), out in zip(reqs, outputs):
            f.write(json.dumps({**meta, "raw": out.outputs[0].text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(reqs)} rows to {args.out}")


if __name__ == "__main__":
    main()
