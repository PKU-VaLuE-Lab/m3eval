import importlib.util
import unittest
from pathlib import Path

_PARSER_PATH = Path(__file__).resolve().parents[3] / "lmms_eval/tasks/m3eval/parsing_utils.py"
_SPEC = importlib.util.spec_from_file_location("m3eval_parser_for_tests", _PARSER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PARSER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PARSER)

ANSWER_RESOLUTION_SINGLE_LETTER = _PARSER.ANSWER_RESOLUTION_SINGLE_LETTER
ANSWER_RESOLUTION_STRICT = _PARSER.ANSWER_RESOLUTION_STRICT
STRICT_FINAL_ANSWER_INVALID = _PARSER.STRICT_FINAL_ANSWER_INVALID
STRICT_FINAL_ANSWER_MISSING = _PARSER.STRICT_FINAL_ANSWER_MISSING
parse_mc_response_with_details = _PARSER.parse_mc_response_with_details


class TestAnswerResolutionParsing(unittest.TestCase):
    def test_strict_final_answer_success(self):
        parsed = parse_mc_response_with_details("A\nThe final answer is A", allowed_letters=["A", "B", "C", "D"], strict_final_answer=True)
        self.assertEqual(parsed["answer"], "A")
        self.assertEqual(parsed["format_status"], "ok")
        self.assertTrue(parsed["format_compliant"])
        self.assertEqual(parsed["resolution_status"], ANSWER_RESOLUTION_STRICT)
        self.assertTrue(parsed["resolution_success"])
        self.assertEqual(parsed["deepseek_input_messages"], [])
        self.assertEqual(parsed["deepseek_raw_output"], "")
        self.assertEqual(parsed["deepseek_error"], "")

    def test_strict_final_answer_allows_trailing_punctuation(self):
        parsed = parse_mc_response_with_details("The final answer is B.", allowed_letters=["A", "B", "C", "D"], strict_final_answer=True)
        self.assertEqual(parsed["answer"], "B")
        self.assertEqual(parsed["format_status"], "ok")
        self.assertTrue(parsed["format_compliant"])
        self.assertEqual(parsed["resolution_status"], ANSWER_RESOLUTION_STRICT)
        self.assertTrue(parsed["resolution_success"])
        self.assertFalse(parsed["deepseek_checked"])

    def test_single_uppercase_letter_fallback(self):
        parsed = parse_mc_response_with_details("A", allowed_letters=["A", "B", "C", "D"], strict_final_answer=True)
        self.assertEqual(parsed["answer"], "A")
        self.assertEqual(parsed["format_status"], STRICT_FINAL_ANSWER_MISSING)
        self.assertFalse(parsed["format_compliant"])
        self.assertEqual(parsed["resolution_status"], ANSWER_RESOLUTION_SINGLE_LETTER)
        self.assertTrue(parsed["resolution_success"])

    def test_single_lowercase_letter_is_not_accepted(self):
        parsed = parse_mc_response_with_details(" b ", allowed_letters=["A", "B", "C", "D"], strict_final_answer=True)
        self.assertEqual(parsed["answer"], "")
        self.assertEqual(parsed["format_status"], STRICT_FINAL_ANSWER_MISSING)
        self.assertEqual(parsed["resolution_status"], STRICT_FINAL_ANSWER_MISSING)
        self.assertFalse(parsed["resolution_success"])

    def test_invalid_letter_in_strict_answer_stops_resolution(self):
        parsed = parse_mc_response_with_details("The final answer is Z", allowed_letters=["A", "B", "C", "D"], strict_final_answer=True)
        self.assertEqual(parsed["answer"], "")
        self.assertEqual(parsed["format_status"], STRICT_FINAL_ANSWER_INVALID)
        self.assertEqual(parsed["resolution_status"], STRICT_FINAL_ANSWER_INVALID)
        self.assertFalse(parsed["resolution_success"])


if __name__ == "__main__":
    unittest.main()
