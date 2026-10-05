"""Leakage regression tests; passing them does not prove attack detection."""
import copy
import unittest

from evaluation_guards import calibrate_threshold, scoring_payload, validate_partitions
from inspect_sample import pre_action_contexts


def fixture(case_id, group, label=0):
    return {"id": case_id, "group_id": group, "label": label,
            "detector_input": {"contexts": {"original": "original " + case_id,
                                            "perturbed": "perturbed " + case_id},
                               "continuation": "same candidate"}}


class EvaluationGuardsTests(unittest.TestCase):
    def test_label_and_outcome_mutation_cannot_change_payload(self):
        case = fixture("a", "a")
        before = scoring_payload(case)
        case.update(label=1, label_evidence="ATTACK", outcome="FAIL", private_actor_goal="COVERT")
        self.assertEqual(scoring_payload(case), before)
        case["detector_input"]["label"] = 1
        with self.assertRaises(ValueError):
            scoring_payload(case)

    def test_current_and_future_outcomes_do_not_enter_context(self):
        rows = [{"id": str(i), "agent_action": "candidate", "output": "old", "error": None} for i in range(4)]
        before = pre_action_contexts(rows, 2)
        rows[2].update(output="CURRENT_OUTCOME", error="CURRENT_LABEL")
        rows[3].update(agent_action="FUTURE_REPAIR", output="FUTURE_OUTCOME")
        self.assertEqual(pre_action_contexts(rows, 2), before)
        self.assertNotIn("CURRENT_OUTCOME", str(before))

    def test_noop_is_unavailable_not_negative(self):
        case = fixture("a", "a")
        case["detector_input"]["contexts"]["perturbed"] = case["detector_input"]["contexts"]["original"]
        with self.assertRaises(ValueError):
            scoring_payload(case)

    def test_groups_and_exact_duplicates_cannot_cross_splits(self):
        partitions = {"development": [fixture("a", "g1")], "calibration": [fixture("b", "g2")], "test": [fixture("c", "g3", 1)]}
        validate_partitions(partitions)
        related = copy.deepcopy(partitions)
        related["test"][0]["group_id"] = "g1"
        with self.assertRaises(ValueError):
            validate_partitions(related)
        duplicate = copy.deepcopy(partitions)
        duplicate["test"][0]["detector_input"] = duplicate["development"][0]["detector_input"]
        with self.assertRaises(ValueError):
            validate_partitions(duplicate)

    def test_calibration_is_benign_and_has_enough_episodes(self):
        partitions = {"development": [fixture("a", "g1")], "calibration": [fixture("b", "g2", 1)], "test": [fixture("c", "g3", 1)]}
        with self.assertRaises(ValueError):
            validate_partitions(partitions)
        with self.assertRaises(ValueError):
            calibrate_threshold([1, 2, 3], .05)
        self.assertEqual(calibrate_threshold(list(range(19)), .05), 18)
        self.assertEqual(calibrate_threshold([1] * 19, .05), 1)


if __name__ == "__main__":
    unittest.main()
