from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scirl.reward import extract_expression, is_correct, score_tool_call
from scirl.tasks import make_dataset
from scirl.tools import CalculatorError, safe_calculate


class CoreTest(unittest.TestCase):
    def test_safe_calculator_allows_arithmetic(self) -> None:
        self.assertAlmostEqual(safe_calculate("0.5*4*3**2"), 18.0)
        self.assertAlmostEqual(safe_calculate("sqrt(16)+1"), 5.0)

    def test_safe_calculator_blocks_unsafe_code(self) -> None:
        with self.assertRaises(CalculatorError):
            safe_calculate("__import__('os').system('echo unsafe')")

    def test_dataset_examples_have_valid_oracle_calls(self) -> None:
        examples = make_dataset(30, seed=123)
        for example in examples:
            self.assertTrue(is_correct(example.ideal_call, example))
            self.assertIsNotNone(extract_expression(example.ideal_call))

    def test_reward_prefers_correct_call(self) -> None:
        example = make_dataset(1, seed=5)[0]
        good = score_tool_call(example.ideal_call, example).reward
        bad = score_tool_call("CALL: CALC[1+1]\n", example).reward
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()

