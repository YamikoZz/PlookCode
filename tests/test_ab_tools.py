import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import distill_verify as dv  # noqa: E402
from build_dataset_b import is_template  # noqa: E402
from localbrain import humaneval  # noqa: E402

HE = ROOT / "workspace" / "raw-hf" / "eval" / "humaneval.parquet"


def teacher_raw(code, tests, extra=""):
    meta = {"function": "count_long_words", "params": [{"name": "words", "type": "string[]"},
                                                        {"name": "k", "type": "number"}],
            "returns": "number", "description": "returns how many words are longer than k characters"}
    return f"### META\n{json.dumps(meta)}\n{extra}\n### CODE\n```python\n{code}\n```\n### TESTS\n{tests}\n"


GOOD = "def count_long_words(words, k):\n    return sum(1 for w in words if len(w) > k)"
TESTS = ('[{"args": [["aa", "bbb"], 2], "expected": 1}, {"args": [[], 1], "expected": 0},'
         ' {"args": [["x"], 0], "expected": 1,}, {"args": [["ab", "cde"], 1], "expected": 2}]')


class DistillVerifyTests(unittest.TestCase):
    def test_loose_json_accepts_trailing_commas_and_python_literals(self):
        self.assertEqual(dv.loose_json('[{"a": 1,},]'), [{"a": 1}])
        self.assertEqual(dv.loose_json("{'a': True, 'b': None}"), {"a": True, "b": None})
        self.assertIsNone(dv.loose_json("no json here"))

    def test_kinds_are_verified_by_running(self):
        row = lambda kind, raw: {"kind": kind, "language": "python", "topic": "t", "raw": raw}
        self.assertIsNone(dv.check(row("write", teacher_raw(GOOD, TESTS)))[2])
        self.assertEqual(dv.check(row("write", teacher_raw(GOOD.replace("> k", ">= k"), TESTS)))[2], "tests_fail")
        buggy = "### BUGGY\n```python\n" + GOOD.replace("> k", ">= k") + "\n```\n### BUG\nUsed >= instead of >."
        self.assertIsNone(dv.check(row("debug", teacher_raw(GOOD, TESTS, buggy)))[2])
        same = "### BUGGY\n```python\ndef count_long_words(words, k):\n    return len([w for w in words if len(w) > k])\n```\n### BUG\nx"
        self.assertEqual(dv.check(row("debug", teacher_raw(GOOD, TESTS, same)))[2], "bug_not_caught")
        self.assertEqual(dv.check(row("plan", teacher_raw(GOOD, TESTS, "### PLAN\nno steps")))[2], "bad_plan")

    def test_renamed_exam_task_is_caught(self):
        refs, _ = dv.reserved()
        item = {"params": [{"name": "nums"}, {"name": "k"}], "tests": [
            {"args": [[1, 2, 3, 4], 1], "expected": [4, 1, 2, 3]}, {"args": [[1, 2, 3], 2], "expected": [2, 3, 1]},
            {"args": [[], 3], "expected": []}]}
        self.assertEqual(dv.same_as_reserved(item, refs), "rotate_right")

    def test_rows_use_exam_prompt_style_and_vary_the_example(self):
        item, _ = dv.parse({"kind": "write", "raw": teacher_raw(GOOD, TESTS)})
        rows = dv.rows_for(item, "lua", "t")
        self.assertTrue(rows[0]["question"].startswith("Write a Lua function `count_long_words(words, k)`"))
        self.assertIn("Lists are Lua array tables.", rows[0]["question"])
        self.assertEqual(len({r["question"] for r in rows}), len(rows))


class DatasetBTests(unittest.TestCase):
    def test_template_rows_are_detected(self):
        self.assertTrue(is_template({"source": "code-min_value", "family": "min_value"}))
        self.assertTrue(is_template({"source": "code-x"}))
        self.assertFalse(is_template({"source": "oasst2"}))
        self.assertFalse(is_template({"source": "distill-qwen2.5-coder-32b"}))


@unittest.skipUnless(HE.exists(), "HumanEval parquet not downloaded")
class HumanEvalTests(unittest.TestCase):
    def test_prompts_and_program(self):
        problems = humaneval.load(HE)
        self.assertEqual(len(problems), 164)
        p = problems[0]
        self.assertIn("```python", humaneval.question(p, "complete"))
        self.assertTrue(humaneval.question(p, "instruct").startswith("Write a Python function `has_close_elements("))
        body_only = humaneval.program(p, "    return False\n")
        self.assertIn("def has_close_elements", body_only)
        self.assertIn("check(has_close_elements)", body_only)

    def test_tests_never_run_outside_a_sandbox(self):
        old = os.environ.pop("LOCALBRAIN_SANDBOX", None)
        try:
            with self.assertRaises(RuntimeError):
                humaneval.execute("print(1)")
        finally:
            if old is not None:
                os.environ["LOCALBRAIN_SANDBOX"] = old


if __name__ == "__main__":
    unittest.main()
