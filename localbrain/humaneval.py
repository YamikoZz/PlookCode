"""HumanEval (164 Python problems, openai/openai_humaneval, MIT) scored by running the official tests.

The official tests are external benchmark code, so they execute ONLY when LOCALBRAIN_SANDBOX=1
(set by the Colab scripts), never on the home PC (CLAUDE.md rule 2). The model's code also passes
the same static check as code-eval (check_python: whitelisted imports, no exec/eval/open).

Two prompt formats, because the model was fine-tuned on "Write a Python function ..." requests
rather than raw code completion:
  complete : the original HumanEval prompt (signature + docstring) inside a "complete this" request
  instruct : the same signature and docstring rewritten as a "Write a Python function ..." request
Greedy decoding, pass@1.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .runner import Rejected, check_python

FORMATS = ("complete", "instruct")


def load(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def split_prompt(problem):
    """header (imports/helpers before the target def), def line, docstring text."""
    prompt, name = problem["prompt"], problem["entry_point"]
    match = re.search(rf"^def {re.escape(name)}\s*\(.*?\).*?:\s*$", prompt, flags=re.M | re.S)
    if not match:
        return prompt, "", ""
    header, def_line = prompt[:match.start()], match.group(0).strip()
    doc = re.search(r'("""|\'\'\')(.*?)\1', prompt[match.end():], flags=re.S)
    return header, def_line, (doc.group(2) if doc else "")


def question(problem, fmt):
    if fmt == "complete":
        return ("Complete the following Python function. Write the whole function.\n\n"
                f"```python\n{problem['prompt'].rstrip()}\n```")
    _, def_line, doc = split_prompt(problem)
    signature = def_line[len("def "):].rstrip(":").strip()
    text = " ".join(line.strip() for line in doc.strip().splitlines())
    return f"Write a Python function `{signature}` that solves this task: {text}"


def program(problem, code):
    header, _, _ = split_prompt(problem)
    name = problem["entry_point"]
    if not re.search(rf"^\s*def {re.escape(name)}\s*\(", code, flags=re.M):
        code = problem["prompt"] + code  # the model wrote only the body
        header = ""
    return f"{header}\n{code}\n\n{problem['test']}\n\ncheck({name})\n"


def execute(source, timeout=10):
    if os.environ.get("LOCALBRAIN_SANDBOX") != "1":
        raise RuntimeError("HumanEval tests are external code: run only in a sandbox (LOCALBRAIN_SANDBOX=1)")
    with tempfile.TemporaryDirectory(prefix="lb-he-") as temp:
        path = Path(temp) / "program.py"
        path.write_text(source, encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", str(path)], cwd=temp, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    if done.returncode == 0:
        return "pass", ""
    last = (done.stderr.strip().splitlines() or [""])[-1]
    return ("fail" if "AssertionError" in done.stderr else "error"), last[:200]


def evaluate(run, data, output, device="auto", plain=False, checkpoint=None, formats=FORMATS,
             max_tokens=512):
    from .coder import CodeModel, clean
    model = CodeModel(run, device, checkpoint)
    if plain:
        model.repetition_penalty, model.no_repeat_ngram = 1.0, 0
    problems, began = load(data), time.time()
    report = {"checkpoint_step": model.step, "run": str(run), "problems": len(problems),
              "decoding": {"repetition_penalty": model.repetition_penalty,
                           "no_repeat_ngram": model.no_repeat_ngram, "greedy": True},
              "max_tokens": max_tokens, "rows": []}
    for fmt in formats:
        passed = 0
        for problem in problems:
            reply = model.sample(question(problem, fmt), 1, greedy=True, max_tokens=max_tokens, raw=True)[0]
            code = clean(reply)
            try:
                check_python(code)
                status, message = execute(program(problem, code))
            except Rejected as error:
                status, message = "rejected", str(error)[:200]
            passed += status == "pass"
            report["rows"].append({"task_id": problem["task_id"], "format": fmt, "status": status,
                                   "message": message, "code": code})
        report[fmt] = {"pass@1": round(passed / len(problems), 3), "passed": passed}
    report["seconds"] = round(time.time() - began, 1)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "rows"}
