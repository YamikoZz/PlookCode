"""Distillation data, step 2: keep only teacher exercises that survive execution and decontamination.

Teacher code runs ONLY in a sandbox (Colab), never on the home PC (CLAUDE.md rule 2), through
localbrain.runner.run_tests (static checks + subprocess limits). Per kind:
  write / plan : CODE passes all TESTS
  debug        : CODE passes all TESTS and BUGGY fails at least one
  predict      : CODE reproduces every expected value in TESTS (so the teacher's prediction was right)
Then drops anything that resembles a reserved evaluation task:
  - function name of an exam / held-out family or a HumanEval entry point
  - 13-gram overlap with HumanEval / MBPP / exam text (scripts/import_code_sft.py)
  - concept_equal: the reference solution of an exam or held-out family reproduces ALL of the
    exercise's tests (same task under a new name)
  - concept_similar: description word overlap (Jaccard >= 0.6) with an exam / held-out / HumanEval task
  - near duplicates of another kept exercise (question skeleton or renamed solution body)

  python scripts/distill_verify.py --raw /content/distill2-raw.jsonl --evalraw /content/raw-hf \
      --out /content/distill2-verified.jsonl
"""
import argparse
import ast
import collections
import concurrent.futures as cf
import copy
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
STOP = set("a an the of in on to and or is are be by for with from that this which its it as at each "
           "every all any given list string number numbers integer integers return returns value values "
           "function write element elements".split())


# ---------- parsing ----------
def sections(raw):
    parts, current = {}, None
    for line in raw.replace("\r", "").split("\n"):
        m = re.match(r"^\s*#{2,4}\s*([A-Z]+)\s*$", line)
        if m:
            current = m.group(1)
            parts[current] = []
        elif current:
            parts[current].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


def code_block(text):
    m = re.search(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", text or "", flags=re.S)
    return (m.group(1) if m else (text or "")).strip("\n")


def loose_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    end = max(text.rfind("}"), text.rfind("]"))
    body = text[start:end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([\]}])", r"\1", re.sub(r"//[^\n]*", "", body))
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    try:  # Python-style literals (True/False/None, single quotes)
        return ast.literal_eval(fixed.replace("true", "True").replace("false", "False").replace("null", "None"))
    except Exception:  # noqa: BLE001 - any malformed teacher text just means "unparseable"
        return None


def plain_value(v, depth=0):
    if isinstance(v, (bool, int)):
        return True
    if isinstance(v, str):
        return len(v) <= 200
    if isinstance(v, list) and depth < 2 and len(v) <= 40:
        return all(plain_value(x, depth + 1) for x in v)
    return False


def parse(row):
    """-> (item dict, None) or (None, reason)."""
    s = sections(row["raw"])
    kind = row["kind"]
    meta = loose_json(s.get("META"))
    tests = loose_json(s.get("TESTS"))
    if not isinstance(meta, dict):
        return None, "bad_meta"
    fn, params, desc = meta.get("function"), meta.get("params"), meta.get("description")
    if isinstance(params, list) and all(isinstance(p, str) for p in params):
        params = [{"name": p, "type": "any"} for p in params]
    if not (isinstance(fn, str) and NAME.fullmatch(fn)):
        return None, "bad_name"
    if not (isinstance(params, list) and all(isinstance(p, dict) and isinstance(p.get("name"), str)
                                             and NAME.fullmatch(p["name"]) for p in params)):
        return None, "bad_params"
    if not (isinstance(desc, str) and 10 <= len(desc) <= 400):
        return None, "bad_description"
    code = code_block(s.get("CODE"))
    if not code or fn not in code or len(code) > 4000:
        return None, "bad_code"
    if not (isinstance(tests, list) and 3 <= len(tests) <= 12):
        return None, "bad_tests"
    clean = []
    for t in tests:
        if not (isinstance(t, dict) and isinstance(t.get("args"), list) and "expected" in t):
            return None, "bad_test_shape"
        args = t["args"]
        if len(params) == 1 and len(args) != 1:
            args = [args]  # teacher wrote the single list argument unwrapped
        if len(args) != len(params) or not plain_value(args) or not plain_value(t["expected"]):
            return None, "bad_test_values"
        clean.append({"args": args, "expected": t["expected"]})
    item = {"kind": kind, "function": fn, "params": params, "returns": meta.get("returns", "any"),
            "description": desc.strip(), "code": code, "tests": clean}
    if kind == "plan":
        item["plan"] = s.get("PLAN", "").strip()
        if not re.search(r"^\s*1[.)]", item["plan"], flags=re.M):
            return None, "bad_plan"
    if kind == "debug":
        item["buggy"], item["bug"] = code_block(s.get("BUGGY")), s.get("BUG", "").strip()
        if not item["buggy"] or fn not in item["buggy"] or not item["bug"] or item["buggy"].strip() == code.strip():
            return None, "bad_debug"
    if kind == "predict":
        item["trace"] = s.get("TRACE", "").strip()
        if not 10 <= len(item["trace"]) <= 600:
            return None, "bad_trace"
    return item, None


# ---------- execution ----------
def check(row):
    try:
        item, reason = parse(row)
    except Exception:  # noqa: BLE001 - one odd teacher answer must never stop the whole run
        return row, None, "parse_crash", None
    if reason:
        return row, None, reason, None
    language = row["language"]
    if language == "lua" and re.search(r"\b(require|io|os|load|dofile|debug)\b", item["code"] + item.get("buggy", "")):
        return row, item, "lua_unsafe", None
    result = run_tests(language, item["code"], item["function"], item["tests"], timeout=5)
    if result["status"] != "pass":
        return row, item, "tests_" + result["status"], result.get("message", "")[:200]
    if item["kind"] == "debug":
        buggy = run_tests(language, item["buggy"], item["function"], item["tests"], timeout=5)
        if buggy["status"] == "pass":
            return row, item, "bug_not_caught", None
    return row, item, None, None


# ---------- decontamination ----------
def reserved():
    from localbrain.exam import F as EXAM
    from localbrain.tasks import FAMILIES, HELD_OUT
    refs = [(f["name"], len(f["params"]), f["ref"]) for f in EXAM]
    refs += [(fam, len(FAMILIES[fam]["params"]), FAMILIES[fam]["ref"]) for fam in HELD_OUT]
    texts = [f["desc"] for f in EXAM] + [d.replace("$", "") for fam in HELD_OUT for d in FAMILIES[fam]["descs"]]
    return refs, texts


def same_as_reserved(item, refs):
    for name, arity, ref in refs:
        if arity != len(item["params"]):
            continue
        try:
            if all(ref(*copy.deepcopy(t["args"])) == t["expected"] for t in item["tests"]):
                return name
        except Exception:  # noqa: BLE001 - our own reference code on foreign inputs
            continue
    return None


def bag(text):
    return {w for w in re.findall(r"[a-z]+", text.casefold()) if w not in STOP and len(w) > 2}


def similar(desc, bags):
    b = bag(desc)
    return bool(b) and any(len(b & o) / len(b | o) >= 0.6 for o in bags if o)


def skeleton(item):
    text = item["description"].casefold().replace(item["function"].casefold(), "F")
    return " ".join(re.findall(r"[a-z]+|N", re.sub(r"\d+", "N", text)))


def body_key(item):
    return re.sub(r"\s+", " ", item["code"].replace(item["function"], "F")).strip()


# ---------- SFT rows ----------
def signature(item, language):
    if language == "typescript":
        params = ", ".join(f"{p['name']}: {p.get('type', 'any')}" for p in item["params"])
        return f"{item['function']}({params}): {item['returns']}"
    return f"{item['function']}({', '.join(p['name'] for p in item['params'])})"


def call(item, test, language):
    return f"{item['function']}({', '.join(literal(a, language) for a in test['args'])})"


def rows_for(item, language, topic, max_variants=6):
    """One or more {"question","answer"} rows; variants differ in the example / call shown."""
    fence, lang = FENCE[language], LANGS[language]
    desc = re.sub(r"^(write a function that|a function that)\s+", "", item["description"].rstrip("."), flags=re.I)
    note = " Lists are Lua array tables." if language == "lua" else ""
    code = f"```{fence}\n{item['code'].strip()}\n```"
    shown = [t for t in item["tests"] if all(a not in ([], "") for a in t["args"])] or item["tests"]
    out = []
    if item["kind"] in ("write", "plan"):
        answer = code if item["kind"] == "write" else f"Plan:\n{item['plan']}\n\n{code}"
        for t in shown[:max_variants]:
            q = (f"Write a {lang} function `{signature(item, language)}` that {desc}.{note}\n"
                 f"Example: {call(item, t, language)} returns {literal(t['expected'], language)}")
            out.append((q, answer))
    elif item["kind"] == "debug":
        q = (f"This {lang} function is supposed to work like this: it {desc}. But it has a bug:{note}\n\n"
             f"```{fence}\n{item['buggy'].strip()}\n```\nFind the bug and fix it.")
        out.append((q, f"Bug: {item['bug']}\n\nFixed code:\n{code}"))
    else:  # predict
        for i, t in enumerate(shown[:3]):
            q = f"What does `{call(item, t, language)}` return?{note}\n\n{code}"
            trace = item["trace"] + "\n\n" if i == 0 else ""
            out.append((q, f"{trace}It returns `{literal(t['expected'], language)}`."))
    return [{"question": q, "answer": a, "source": "distill-qwen2.5-coder-32b", "kind": item["kind"],
             "language": language, "topic": topic, "concept": item["function"] + "#" + skeleton(item)[:60]}
            for q, a in out]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/content/distill2-raw.jsonl")
    parser.add_argument("--evalraw", default="/content/raw-hf")
    parser.add_argument("--out", default="/content/distill2-verified.jsonl")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from import_code_sft import eval_ngrams, held_out_names, overlaps_eval
    contaminated = eval_ngrams(Path(args.evalraw))
    he = pq.read_table(Path(args.evalraw) / "eval" / "humaneval.parquet").to_pylist()
    names = held_out_names() | {p["entry_point"] for p in he}
    refs, texts = reserved()
    he_texts = [(p["prompt"].split('"""')[1] if '"""' in p["prompt"] else p["prompt"]).strip().split("\n")[0]
                for p in he]
    bags = [bag(t) for t in texts + he_texts]
    raw = [json.loads(line) for line in open(args.raw, encoding="utf-8") if line.strip()]
    dropped, kept_items, out = collections.Counter(), collections.Counter(), []
    seen_sk, seen_body, samples = set(), set(), collections.defaultdict(list)
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for row, item, reason, detail in pool.map(check, raw):
            language = row["language"]
            if not reason:
                if item["function"] in names:
                    reason = "reserved_name"
                elif same_as_reserved(item, refs):
                    reason = "concept_equal"
                elif similar(item["description"], bags):
                    reason = "concept_similar"
                elif skeleton(item) in seen_sk or body_key(item) in seen_body:
                    reason = "near_duplicate"
            if not reason:
                new = [r for r in rows_for(item, language, row["topic"]) if not overlaps_eval(r["question"], contaminated)]
                if not new:
                    reason = "eval_overlap"
            if reason:
                dropped[(row["kind"], reason)] += 1
                if detail and len(samples[reason]) < 3:
                    samples[reason].append(detail)
                continue
            seen_sk.add(skeleton(item))
            seen_body.add(body_key(item))
            kept_items[(row["kind"], language)] += 1
            out.extend(new)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"raw": len(raw), "kept_items": sum(kept_items.values()), "rows": len(out),
              "kept_by_kind_language": {f"{k}/{l}": n for (k, l), n in sorted(kept_items.items())},
              "dropped": {f"{k}/{r}": n for (k, r), n in sorted(dropped.items())},
              "error_samples": samples}
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "error_samples"}, indent=2))


if __name__ == "__main__":
    main()
