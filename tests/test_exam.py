import json
import unittest

from localbrain import exam, tasks
from localbrain.runner import lua_expected, run_tests, toolchains

CHAINS = toolchains()


class ExamSuiteTests(unittest.TestCase):
    def test_exam_is_fixed_and_complete(self):
        first, second = exam.exam(), exam.exam()
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(len(first), len(exam.F) * len(exam.LANGUAGES))
        for task in first:
            self.assertEqual(len(task["tests"]), 8, task["family"])
            self.assertEqual(task["split"], "exam")

    def test_exam_families_are_not_training_families(self):
        self.assertFalse(set(exam.FAMILIES) & set(tasks.FAMILIES))
        trained_names = {n for f in tasks.F for n in f["names"]}
        self.assertFalse(set(exam.FAMILIES) & trained_names)

    def test_prompt_example_is_not_a_scored_test(self):
        for task in exam.exam():
            shown = task["prompt"].split("Example: ", 1)[1]
            args = shown.split(") returns ", 1)[0].split("(", 1)[1]
            scored = {", ".join(exam.literal(a, task["language"]) for a in t["args"]) for t in task["tests"]}
            self.assertNotIn(args, scored, task["prompt"])

    def test_lua_prompts_use_lua_tables(self):
        task = next(t for t in exam.exam() if t["language"] == "lua" and t["family"] == "zip_sum")
        self.assertIn("{", task["prompt"].split("Example: ")[1])
        self.assertNotIn("[", task["prompt"].split("Example: ")[1])

    def _check_references(self, language):
        for task in exam.exam(languages=(language,)):
            result = run_tests(language, task["solutions"][0], task["function"], task["tests"])
            self.assertEqual(result["status"], "pass", (task["family"], result["message"]))

    @unittest.skipUnless("javascript" in CHAINS, "node not installed")
    def test_javascript_references_pass(self):
        self._check_references("javascript")

    @unittest.skipUnless("typescript" in CHAINS, "node without TypeScript type stripping")
    def test_typescript_references_pass(self):
        self._check_references("typescript")

    @unittest.skipUnless("lua" in CHAINS, "lua not installed")
    def test_lua_references_pass(self):
        self._check_references("lua")


class RunnerLanguageTests(unittest.TestCase):
    cases = [{"args": [[1, 2, 3], 2], "expected": [1, 3]}, {"args": [[], 1], "expected": []}]

    @unittest.skipUnless("typescript" in CHAINS, "node without TypeScript type stripping")
    def test_typescript_types_are_stripped_and_wrong_answers_fail(self):
        good = ("function drop_every_kth(nums: number[], k: number): number[] {\n"
                "  const out: number[] = [];\n  nums.forEach((x: number, i: number) => { if ((i + 1) % k) out.push(x); });\n"
                "  return out;\n}\n")
        self.assertEqual(run_tests("typescript", good, "drop_every_kth", self.cases)["status"], "pass")
        wrong = good.replace("(i + 1) % k", "i % k")
        self.assertEqual(run_tests("typescript", wrong, "drop_every_kth", self.cases)["status"], "fail")
        broken = run_tests("typescript", "function drop_every_kth(: {\n", "drop_every_kth", self.cases)
        self.assertEqual(broken["status"], "error")
        blocked = run_tests("typescript", "function drop_every_kth() { return process.exit(1); }",
                            "drop_every_kth", self.cases)
        self.assertEqual(blocked["status"], "rejected")

    def test_lua_blocklist_rejects_before_running(self):
        for code in ("local function f() return os.time() end", "local f = load('return 1')",
                     "local function f() io.write('x') end", "local f = require('socket')"):
            if "lua" in CHAINS:
                self.assertEqual(run_tests("lua", code, "f", self.cases)["status"], "rejected", code)
            else:
                from localbrain.runner import Rejected, check_lua
                with self.assertRaises(Rejected):
                    check_lua(code)

    def test_lua_expected_encoding_matches_json_for_plain_values(self):
        for value in (1, -3, True, "a\"b", [1, [2, 3]], [], ["x", "y"]):
            self.assertEqual(json.loads(lua_expected(value)), value)
        self.assertEqual(lua_expected(2.0), "2")

    @unittest.skipUnless("lua" in CHAINS, "lua not installed")
    def test_lua_runner_passes_fails_and_handles_errors(self):
        good = ("local function drop_every_kth(nums, k)\n  local out = {}\n"
                "  for i, x in ipairs(nums) do if i % k ~= 0 then out[#out + 1] = x end end\n  return out\nend\n")
        self.assertEqual(run_tests("lua", good, "drop_every_kth", self.cases)["status"], "pass")
        global_style = good.replace("local function", "function")
        self.assertEqual(run_tests("lua", global_style, "drop_every_kth", self.cases)["status"], "pass")
        wrong = good.replace("i % k ~= 0", "i % k == 0")
        self.assertEqual(run_tests("lua", wrong, "drop_every_kth", self.cases)["status"], "fail")
        crash = run_tests("lua", "local function drop_every_kth(nums, k) return nums.x.y end",
                          "drop_every_kth", self.cases)
        self.assertEqual(crash["status"], "error")
        loop = run_tests("lua", "local function drop_every_kth(n, k) while true do end end",
                         "drop_every_kth", self.cases, timeout=2)
        self.assertEqual(loop["status"], "timeout")
        floats = run_tests("lua", "local function half(n) return n / 2 end", "half",
                           [{"args": [4], "expected": 2}, {"args": [3], "expected": 1.5}])
        self.assertEqual(floats["status"], "pass")


if __name__ == "__main__":
    unittest.main()
