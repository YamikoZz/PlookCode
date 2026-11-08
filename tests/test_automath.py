import unittest

from localbrain.tools import auto_math


class AutoMathTests(unittest.TestCase):
    def test_arithmetic_is_computed_correctly(self):
        cases = {
            "What is 25 * 10?": ("25 * 10", 250),
            "What is 25 + 17?": ("25 + 17", 42),
            "Add 48 and 76.": ("48+76", 124),
            "25 * 10": ("25 * 10", 250),
            "Calculate (8+4)*2": ("(8+4)*2", 24),
            "What is 100 / 8?": ("100 / 8", 12.5),
            "multiply 12 and 12": ("12*12", 144),
            "sum of 9 and 10": ("9+10", 19),
            "subtract 5 from 20": ("20-5", 15),
            "7 times 8": ("7*8", 56),
            "2 ** 10": ("2 ** 10", 1024),
            "divide 144 by 12": ("144/12", 12),
        }
        for query, expected in cases.items():
            self.assertEqual(auto_math(query), expected, query)

    def test_non_math_is_left_to_the_model(self):
        for query in ["Write a Python function that reverses a string.",
                      "What is a list in Python?", "Hello", "Explain recursion",
                      "sum_list(nums)", "", "a + b"]:
            self.assertIsNone(auto_math(query), query)

    def test_division_by_zero_falls_back(self):
        self.assertIsNone(auto_math("What is 5 / 0?"))


if __name__ == "__main__":
    unittest.main()
