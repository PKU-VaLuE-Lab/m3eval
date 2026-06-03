from __future__ import annotations

import unittest

from lmms_eval.tasks.m3eval.m3eval_parsing import parse_strict_final_answer, parse_strict_final_yes_no_answer
from lmms_eval.tasks.m3eval.m3eval_utils import m3eval_memory_interference_process_results, public_nback_process_results


class M3EvalParsingTest(unittest.TestCase):
    def test_strict_multiple_choice_answer(self):
        self.assertEqual(parse_strict_final_answer("Reasoning\nThe final answer is B", ["A", "B"]), ("B", "ok"))

    def test_bare_multiple_choice_answer(self):
        self.assertEqual(parse_strict_final_answer("A", ["A", "B"]), ("A", "ok_bare_answer"))

    def test_lowercase_multiple_choice_answer_remains_invalid(self):
        self.assertEqual(parse_strict_final_answer("a", ["A", "B"]), ("", "missing_required_final_answer_format"))

    def test_disallowed_bare_multiple_choice_answer_remains_invalid(self):
        self.assertEqual(parse_strict_final_answer("C", ["A", "B"]), ("", "invalid_final_answer_option_letter"))

    def test_extra_text_multiple_choice_answer_remains_invalid(self):
        self.assertEqual(parse_strict_final_answer("Answer: A", ["A", "B"]), ("", "missing_required_final_answer_format"))

    def test_strict_yes_no_answer(self):
        self.assertEqual(parse_strict_final_yes_no_answer("Reasoning\nThe final answer is no"), ("No", "ok"))

    def test_bare_yes_no_answer(self):
        self.assertEqual(parse_strict_final_yes_no_answer("no"), ("No", "ok_bare_answer"))

    def test_extra_text_yes_no_answer_remains_invalid(self):
        self.assertEqual(parse_strict_final_yes_no_answer("No, because it matches."), ("", "missing_required_final_answer_format"))

    def test_multiple_choice_process_results_accepts_bare_answer(self):
        doc = {"answer": "A", "options": ["A. red", "B. blue"]}
        scores = m3eval_memory_interference_process_results(doc, ["A"])
        self.assertIs(scores["m3eval_memory_interference_accuracy"], True)

    def test_memory_interference_intrusion_rate_from_option_roles(self):
        doc = {
            "answer": "A",
            "options": ["A. correct", "B. intrusion", "C. distractor", "D. intrusion"],
            "paper_v4_metric_group": "proactive",
            "option_role_by_letter": {"A": "correct", "B": "intrusion", "C": "irrelevant_distractor", "D": "intrusion"},
            "intrusion_option_letter": "B",
        }
        scores = m3eval_memory_interference_process_results(doc, ["D"])
        self.assertIs(scores["m3eval_memory_interference_accuracy"], False)
        self.assertIs(scores["m3eval_memory_interference_proactive_accuracy"], False)
        self.assertIs(scores["m3eval_memory_interference_proactive_intrusion_rate"], True)

    def test_memory_interference_intrusion_rate_missing_metadata_is_none(self):
        doc = {
            "answer": "A",
            "options": ["A. correct", "B. wrong"],
            "paper_v4_metric_group": "retroactive",
        }
        scores = m3eval_memory_interference_process_results(doc, ["B"])
        self.assertIs(scores["m3eval_memory_interference_retroactive_intrusion_rate"], None)

    def test_nback_process_results_accepts_bare_answer(self):
        doc = {"answer": "No", "attribute": "action", "n_value": 1}
        scores = public_nback_process_results(doc, ["no"])
        self.assertIs(scores["m3eval_nback_accuracy"], True)
        self.assertIs(scores["m3eval_nback_action_accuracy"], True)
        self.assertIs(scores["m3eval_nback_n1_accuracy"], True)


if __name__ == "__main__":
    unittest.main()
